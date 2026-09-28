"""DM0.5 LoRA variant micro-benchmark (single GPU). Variants:
A        : stock DM05LoraConfig (all-linear LoRA on SigLIP+LLM+AE) + modules_to_save
B        : LoRA only on action-expert linears + same modules_to_save (stock apply path,
           so HF enable_input_require_grads hooks stay active)
B_nohook : B, then delete every make_inputs_require_grad* forward hook (5 in our run) so VLM activations do not require grad
Trainable params upcast to fp32 after wrapping (as in the plan).
"""
import argparse
import json
import os
import statistics
import subprocess
import threading
import time

import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True, choices=["A", "B", "B_nohook"])
ap.add_argument("--bs", type=int, required=True)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--iters", type=int, default=15)
ap.add_argument("--out", required=True)
ap.add_argument("--phys_gpu", default="1")
ap.add_argument("--ddp", action="store_true")
args = ap.parse_args()

# 실험 루트(opendm/) = 이 파일에서 세 단계 위. DM05_MODEL 로 덮어쓴다.
MODEL = os.environ.get("DM05_MODEL", str(__import__("pathlib").Path(__file__).resolve().parents[3] / "models" / "dm05"))
SCRATCH = os.environ.get("BENCH_OUT", os.path.dirname(os.path.abspath(__file__)))
result = {"variant": args.variant, "bs": args.bs, "ddp_ws1": args.ddp}

# ---------------- nvidia-smi peak poller ----------------
smi_peak = [0]
stop_evt = threading.Event()


def poll():
    while not stop_evt.is_set():
        try:
            o = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                 "-i", args.phys_gpu], capture_output=True, text=True, timeout=5).stdout
            smi_peak[0] = max(smi_peak[0], int(o.strip()))
        except Exception:
            pass
        stop_evt.wait(0.2)


threading.Thread(target=poll, daemon=True).start()

from opendm.exp.dm05_exp import DM05ModelConfig  # noqa: E402
from opendm.model.dm05.dm05_lora import DM05LoraConfig, unwrap_dm05_model  # noqa: E402
from opendm.data.transforms import ChatTokenization, PadAction  # noqa: E402
from opendm.data.collator import TrainingCollator  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

AE_REGEX = (r"model\.action_expert\.layers\.\d+\."
            r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)")

lora_kwargs = dict(dump_trainable_path=os.path.join(SCRATCH, f"trainable_{args.variant}.json"))
if args.variant in ("B", "B_nohook"):
    lora_kwargs["target_modules"] = AE_REGEX


class BenchModelConfig(DM05ModelConfig):
    def _wrap_new_lora_for_training(self, model):
        m = super()._wrap_new_lora_for_training(model)
        n = 0
        for p in m.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
                n += 1
        result["upcast_param_tensors"] = n
        return m


mc = BenchModelConfig(
    model_name_or_path=MODEL,
    chunk_size=32,
    precision_policy="bf16_mixed",
    bf16=True,
    llm_attn_implementation="eager",
    vision_attn_implementation="sdpa",
    action_attn_implementation="sdpa",
    liger_kernel=True,
    freeze_vlm_embedding=True,
    vlm_gradient_checkpointing=True,
    ae_gradient_checkpointing=True,
    lora_config=DM05LoraConfig(**lora_kwargs),
)
t0 = time.time()
model = mc.build_model(use_lora=True)
result["build_s"] = time.time() - t0
dm = unwrap_dm05_model(model)
dm.config.use_cache = False
dm.model.vlm.config.use_cache = False
result["n_require_grads_hooks"] = len(getattr(dm, "_require_grads_hooks", []) or [])
# list every "make inputs require grad" forward hook (HF GC enable, OpenDM apply_lora, PEFT)
req_hooks = []
for mname, mod in model.named_modules():
    for hid, fn in list(mod._forward_hooks.items()):
        if getattr(fn, "__name__", "").startswith("make_inputs_require_grad"):
            req_hooks.append((mname, hid))
result["require_grad_hooks_found"] = sorted(set(m for m, _ in req_hooks))
if args.variant == "B_nohook":
    removed = 0
    for mname, mod in model.named_modules():
        for hid, fn in list(mod._forward_hooks.items()):
            if getattr(fn, "__name__", "").startswith("make_inputs_require_grad"):
                del mod._forward_hooks[hid]
                removed += 1
    result["require_grad_hooks_removed"] = removed


def group_of(name):
    if "vision_tower" in name:
        return "siglip"
    if "language_model" in name or "multi_modal_projector" in name or "lm_head" in name:
        return "llm"
    if "action_expert.layers" in name:
        return "ae_layers"
    return "action_other"


trainable = {}
dtypes = set()
for n, p in model.named_parameters():
    if p.requires_grad:
        g = group_of(n)
        trainable[g] = trainable.get(g, 0) + p.numel()
        dtypes.add(str(p.dtype))
result["trainable_by_group"] = trainable
result["trainable_total"] = sum(trainable.values())
result["trainable_dtypes"] = sorted(dtypes)

model.to("cuda")
model.train()
torch.cuda.synchronize()
result["mem_after_load_GB"] = torch.cuda.memory_allocated() / 1e9

# ---------------- synthetic batch through OpenDM transforms ----------------
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
tok = ChatTokenization(processor=processor, n_bins=256, max_length=1024,
                       image_prompts=["Head", "Left wrist", "Right wrist"], add_state=True)
pad = PadAction(32)
rng = np.random.default_rng(0)
samples = []
for i in range(args.bs):
    imgs = [Image.fromarray(rng.integers(0, 256, (448, 448, 3), dtype=np.uint8)) for _ in range(3)]
    data = {
        "meta_data": {"robot_type": "R1Pro", "control_mode": "joint", "speed": "0.5"},
        "prompt": "Thaw the frozen chicken breast. Subtask: open the door of the fridge",
        "images": imgs,
        "state": rng.uniform(-1, 1, 23).astype(np.float32),
        "action": rng.standard_normal((1, 32, 23)).astype(np.float32),
        "action_mask": np.ones((1, 32, 23), dtype=bool),
    }
    out = pad(tok(data))
    samples.append(out)
result["prompt_tokens_unpadded"] = int(samples[0]["input_ids"].shape[1])
coll = TrainingCollator(pad_token_id=processor.tokenizer.pad_token_id, max_length=1024)
batch = coll(samples)
batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
result["batch_shapes"] = {k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)}
result["n_image_tokens_per_sample"] = int((batch["token_type_ids"][0] == 1).sum())

params = [p for p in model.parameters() if p.requires_grad]
fwd_model = model
if args.ddp:
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29577")
    dist.init_process_group("nccl", rank=0, world_size=1)
    from torch.nn.parallel import DistributedDataParallel as DDP
    fwd_model = DDP(model, device_ids=[0], find_unused_parameters=True)
    torch.cuda.synchronize()
    result["mem_after_ddp_wrap_GB"] = torch.cuda.memory_allocated() / 1e9
opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-10, fused=True)


import copy
torch.backends.cudnn.deterministic = True
def run_once(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**batch)
    out.loss.backward()
    g = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    for p in model.parameters(): p.grad = None
    return out.loss.item(), g
# run 1: hooks active (stock B)
l1, g1 = run_once(123)
l1b, g1b = run_once(123)  # repeat to measure run-to-run nondeterminism
removed = 0
for mname, mod in model.named_modules():
    for hid, fn in list(mod._forward_hooks.items()):
        if getattr(fn, "__name__", "").startswith("make_inputs_require_grad"):
            del mod._forward_hooks[hid]; removed += 1
l2, g2 = run_once(123)
def cmp(a, b):
    num = sum(((a[k]-b[k])**2).sum().item() for k in a); den = sum((a[k]**2).sum().item() for k in a)
    mx = max((a[k]-b[k]).abs().max().item() for k in a)
    return {"rel_l2": (num/den)**0.5, "max_abs": mx, "same_keys": sorted(a)==sorted(b), "n": len(a)}
res = {"variant": args.variant, "loss_hook": l1, "loss_hook_repeat": l1b, "loss_nohook": l2, "hooks_removed": removed,
       "hook_vs_hook_repeat": cmp(g1, g1b), "hook_vs_nohook": cmp(g1, g2)}
stop_evt.set()
print(json.dumps(res, indent=1)); json.dump(res, open(args.out, "w"), indent=1)
