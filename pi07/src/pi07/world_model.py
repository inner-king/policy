"""π0.7 재현 — 서브골 이미지 월드 모델 (π0.7 의 BAGEL-14B 자리).

π0.7 V-B / App. C:  g* = g_ψ(o_t, ℓ̂_t, m)  — 현재 3뷰 관측과 subtask 문장으로 세그먼트 끝 3뷰를 생성한다.
flow matching, 정답 = 세그먼트 끝 프레임. 행동 조건 동역학 모델이 아니다.

서버 제약에 맞춘 대체 (조사: memory subgoal-world-model-landscape, 2026-09-18):
- BAGEL-14B → FLUX.2 [klein] base 4B (Apache-2.0) + LoRA. 증류되지 않은 base 를 쓴다(LoRA 학습용).
- 3뷰를 가로로 이어 붙인 한 장(머리 | 왼손목 | 오른손목, 각 224×224 → 672×224)을 생성한다.
  현재 3뷰 타일을 FLUX.2 의 참조 이미지(T 좌표 10)로 넣는다 — 이미지 편집 경로 그대로.
- 텍스트 = "Task: … Subtask: …" 를 Qwen3-4B 층 9/18/27 은닉 상태로 인코딩(파이프라인과 같음).
  문장 종류가 닫혀 있어(태스크×subtask ≈ 600) 미리 계산해 캐시한다. 텍스트 인코더(8 GB)는 학습 중
  메모리에 올리지 않는다.
- π0.7 은 메타데이터 m 도 넣지만 여기서는 뺐다 — 세그먼트 끝 장면은 속도와 거의 무관하고,
  넣으면 캐시할 문장 수가 속도 구간 수만큼 늘어난다.
- CFG 를 쓰려고 학습 때 10% 확률로 빈 문장("")으로 바꾼다.

CLI:
  python src/pi07/world_model.py cache-text       # data/pi07/wm_text_cache.pt 생성 (GPU 8 GB, 1~2분)
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import LoRALinear  # noqa: E402

CAPSTONE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FLUX_ROOT = os.path.join(CAPSTONE, "models", "flux2_klein_base_4b")
TEXT_CACHE = os.path.join(CAPSTONE, "data", "pi07", "wm_text_cache.pt")
TEXT_LAYERS = (9, 18, 27)
REF_T = 10  # FLUX.2 파이프라인의 첫 참조 이미지 T 좌표 (scale + scale*0)


def wm_prompt(instruction: str, subtask: str) -> str:
    return f"Task: {instruction.rstrip('.')}. Subtask: {subtask}."


def tile(views: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W, 3] uint8 → [B, 3, H, 3W] float [-1, 1] (머리 | 왼손목 | 오른손목)."""
    x = views.permute(0, 4, 2, 1, 3).float()  # B, C, H, V, W
    B, C, H, V, W = x.shape
    return x.reshape(B, C, H, V * W).div_(127.5).sub_(1.0)


def untile(img: torch.Tensor, n_views: int = 3) -> torch.Tensor:
    """[B, 3, H, 3W] [-1,1] → [B, 3, H, W, 3] uint8."""
    B, C, H, VW = img.shape
    W = VW // n_views
    x = img.reshape(B, C, H, n_views, W).permute(0, 3, 2, 4, 1)
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


# ---------------------------------------------------------------- 텍스트 캐시

@torch.no_grad()
def build_text_cache(device: str = "cuda:0", max_len: int | None = None):
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    from dataset import B1KIndex

    idx = B1KIndex()
    prompts = sorted({wm_prompt(e["instruction"], g["text"]) for e in idx.episodes for g in e["segments"]})
    prompts = [""] + prompts  # 0번 = CFG 무조건 문장
    tok = AutoTokenizer.from_pretrained(os.path.join(FLUX_ROOT, "tokenizer"))
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for p in prompts]
    lens = [len(tok(t)["input_ids"]) for t in texts]
    if max_len is None:
        max_len = int(math.ceil(max(lens) / 16) * 16)
    print(f"prompts={len(prompts)} token_len max={max(lens)} mean={np.mean(lens):.1f} → max_len={max_len}")
    te = Qwen3ForCausalLM.from_pretrained(os.path.join(FLUX_ROOT, "text_encoder"), torch_dtype=torch.bfloat16,
                                          device_map={"": device})
    te.eval()
    out = torch.empty(len(prompts), max_len, len(TEXT_LAYERS) * te.config.hidden_size, dtype=torch.bfloat16)
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i + 16], return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
        o = te(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device),
               output_hidden_states=True, use_cache=False)
        h = torch.stack([o.hidden_states[k] for k in TEXT_LAYERS], dim=1)  # B, 3, L, D
        B, C, L, D = h.shape
        out[i:i + B] = h.permute(0, 2, 1, 3).reshape(B, L, C * D).cpu()
    torch.save(dict(prompts=prompts, embeds=out, max_len=max_len, layers=TEXT_LAYERS), TEXT_CACHE)
    print(f"saved {TEXT_CACHE} {tuple(out.shape)} {out.numel() * 2 / 1e9:.2f} GB")


# ---------------------------------------------------------------- 모델

def add_lora_flux(model: nn.Module, r: int, alpha: float) -> int:
    names = {"to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out", "linear_in",
             "linear_out", "to_qkv_mlp_proj"}
    n = 0
    for pname, parent in list(model.named_modules()):
        if not ("transformer_blocks" in pname or "single_transformer_blocks" in pname):
            continue
        for cname, child in list(parent.named_children()):
            ok = isinstance(child, nn.Linear) and (
                cname in names or (cname == "to_out" and "single_transformer_blocks" in pname)
                or (cname == "0" and pname.endswith("to_out")))
            if ok:
                setattr(parent, cname, LoRALinear(child, r, alpha))
                n += 1
    return n


class WorldModel(nn.Module):
    def __init__(self, device: torch.device, lora_r: int = 64, lora_alpha: float = 64.0, grad_ckpt: bool = True,
                 text_drop: float = 0.1):
        super().__init__()
        from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel

        self.device = device
        self.text_drop = text_drop
        self.transformer = Flux2Transformer2DModel.from_pretrained(
            os.path.join(FLUX_ROOT, "transformer"), torch_dtype=torch.bfloat16, device_map={"": str(device)})
        self.vae = AutoencoderKLFlux2.from_pretrained(os.path.join(FLUX_ROOT, "vae"), torch_dtype=torch.float32,
                                                      device_map={"": str(device)})
        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.n_lora = add_lora_flux(self.transformer, lora_r, lora_alpha)
        if grad_ckpt:
            self.transformer.enable_gradient_checkpointing()
        tc = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
        self.prompt_index = {p: i for i, p in enumerate(tc["prompts"])}
        self.text_embeds = tc["embeds"].to(device)  # [N, L, 7680] bf16 — ~0.7 GB
        bn = self.vae.bn
        self.register_buffer("bn_mean", bn.running_mean.view(1, -1, 1, 1).float().clone(), persistent=False)
        self.register_buffer("bn_std", torch.sqrt(bn.running_var.view(1, -1, 1, 1).float() + self.vae.config.batch_norm_eps),
                             persistent=False)

    # --- 잠재 공간 ---
    @torch.no_grad()
    def encode(self, img: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """[B, 3, H, W] [-1,1] → 패킹된 잠재 [B, h*w, 128] (파이프라인 _encode_vae_image 와 같은 정규화)."""
        lat = self.vae.encode(img.to(self.device, torch.float32)).latent_dist.mode()
        B, C, H, W = lat.shape
        lat = lat.view(B, C, H // 2, 2, W // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(B, C * 4, H // 2, W // 2)
        lat = (lat - self.bn_mean) / self.bn_std
        h, w = lat.shape[-2:]
        return lat.reshape(B, C * 4, h * w).permute(0, 2, 1).contiguous(), (h, w)

    @torch.no_grad()
    def decode(self, packed: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        B, N, C = packed.shape
        h, w = hw
        lat = packed.permute(0, 2, 1).reshape(B, C, h, w).float() * self.bn_std + self.bn_mean
        lat = lat.reshape(B, C // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(B, C // 4, h * 2, w * 2)
        return self.vae.decode(lat).sample

    def _ids(self, B: int, hw: tuple[int, int], L_txt: int):
        h, w = hw
        hh, ww = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        base = torch.stack([torch.zeros_like(hh), hh, ww, torch.zeros_like(hh)], -1).reshape(-1, 4)
        ref = base.clone()
        ref[:, 0] = REF_T
        img_ids = torch.cat([base, ref], 0)[None].expand(B, -1, -1).to(self.device)
        txt_ids = torch.stack([torch.zeros(L_txt, dtype=torch.long)] * 3 + [torch.arange(L_txt)], -1)
        return img_ids, txt_ids[None].expand(B, -1, -1).to(self.device)

    def text(self, prompts: list[str], drop: float = 0.0) -> torch.Tensor:
        ids = []
        for p in prompts:
            i = self.prompt_index.get(p)
            if i is None:
                raise KeyError(f"텍스트 캐시에 없는 문장: {p!r} — cache-text 를 다시 돌릴 것")
            if drop > 0 and np.random.random() < drop:
                i = 0
            ids.append(i)
        return self.text_embeds[torch.tensor(ids, device=self.device)]

    def velocity(self, x_t, ref, sigma, txt, hw):
        B, N, _ = x_t.shape
        img_ids, txt_ids = self._ids(B, hw, txt.shape[1])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.transformer(hidden_states=torch.cat([x_t, ref], 1).to(torch.bfloat16),
                                   encoder_hidden_states=txt, timestep=sigma.to(torch.bfloat16),
                                   img_ids=img_ids, txt_ids=txt_ids, guidance=None, return_dict=False)[0]
        return out[:, :N].float()

    @staticmethod
    def shift_sigma(s: torch.Tensor, seq_len: int, steps: int = 50) -> torch.Tensor:
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
        mu = compute_empirical_mu(seq_len, steps)
        return math.exp(mu) / (math.exp(mu) + (1 / s.clamp(1e-4, 1 - 1e-4) - 1))

    def loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        ref, hw = self.encode(tile(batch["images"]))
        x0, _ = self.encode(tile(batch["goal_images"]))
        prompts = [wm_prompt(i, s) for i, s in zip(batch["instruction"], batch["subtask"])]
        txt = self.text(prompts, drop=self.text_drop if self.training else 0.0)
        B, N, _ = x0.shape
        s = torch.sigmoid(torch.randn(B, device=self.device))  # logit-normal (SD3/FLUX 학습 관행)
        sigma = self.shift_sigma(s, N)
        noise = torch.randn_like(x0)
        x_t = (1 - sigma[:, None, None]) * x0 + sigma[:, None, None] * noise
        v = self.velocity(x_t, ref, sigma, txt, hw)
        loss = F.mse_loss(v, noise - x0)
        return loss, {"loss": loss.detach()}

    @torch.no_grad()
    def generate(self, images: torch.Tensor, instructions: list[str], subtasks: list[str], steps: int = 28,
                 cfg: float = 4.0, seed: int = 0) -> torch.Tensor:
        """현재 3뷰 [B,3,H,W,3] uint8 → 서브골 3뷰 [B,3,H,W,3] uint8."""
        ref, hw = self.encode(tile(images))
        B, N, C = ref.shape
        txt = self.text([wm_prompt(i, s) for i, s in zip(instructions, subtasks)])
        unc = self.text([""] * B)
        g = torch.Generator(device=self.device).manual_seed(seed)
        x = torch.randn(B, N, C, generator=g, device=self.device)
        sig = self.shift_sigma(torch.linspace(1.0, 1.0 / steps, steps), N, steps)
        sig = torch.cat([torch.as_tensor(sig, dtype=torch.float32), torch.zeros(1)]).to(self.device)
        for i in range(steps):
            s = sig[i].expand(B)
            v = self.velocity(x, ref, s, txt, hw)
            if cfg > 1.0:
                vu = self.velocity(x, ref, s, unc, hw)
                v = vu + cfg * (v - vu)
            x = x + (sig[i + 1] - sig[i]) * v
        return untile(self.decode(x, hw))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cache-text":
        build_text_cache()
    else:
        print(__doc__)
