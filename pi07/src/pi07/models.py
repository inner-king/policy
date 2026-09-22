"""π0.7 재현 — π0.5 base(openpi → LeRobot 포트) 위에 LL 정책과 HL 정책을 얹는다.

LL (저수준 VLA)
  π0.5 구조 그대로: PaliGemma(SigLIP 400M + Gemma 2B) + Gemma 300M action expert, flow matching,
  adaRMS 시간 주입, 50스텝 청크. π0.7 에서 바꾼 것:
  - 프롬프트 = Task / Subtask / Speed / Quality / Mistake / Control Mode (+ π0.5 식 이산 state 토큰)
  - 서브골 3뷰를 텍스트 뒤 block-causal 블록으로 붙인다 (π0.7 Fig.19: 서브골 토큰은 앞의 전부를 보고,
    관측·텍스트는 서브골을 보지 않는다 → 서브골이 없을 때 prefix 는 π0.5 사전학습과 동일)
  - 추론 denoising 5 스텝 (π0.7 VII)
  서버 제약으로 바꾼 것: VLM 동결, action expert 만 학습. prefix 는 no_grad 로 한 번 돌려 KV 캐시만
  expert 에 넘긴다. π0.7 의 knowledge insulation(expert → VLM 그래디언트 차단)의 극단형이다.

HL (고수준 정책)
  π0.7 은 HL 을 같은 구조의 모델로 만든다. 여기서는 LL 과 같은 π0.5 base 의 PaliGemma 를 쓰고
  언어 모델 층에 LoRA 를 달아 subtask 문장을 생성하도록 학습한다 (prefix-LM: 이미지+프롬프트 양방향,
  목표 문장은 인과).

메모리: 가중치는 safetensors 에서 텐서 하나씩 GPU 로 바로 읽는다. CPU 에 14.5 GB 사본을 만들지 않는다.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch, make_att_2d_masks

CAPSTONE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PI05_WEIGHTS = os.path.join(CAPSTONE, "models", "pi05_base", "model.safetensors")
TOKENIZER_DIR = os.path.join(CAPSTONE, "models", "paligemma_tokenizer")
MASK_VALUE = -2.3819763e38  # openpi OPENPI_ATTENTION_MASK_VALUE


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR)


def build_pi05(device: torch.device, with_expert: bool = True, attn: str = "sdpa") -> PI05Pytorch:
    """π0.5 base 를 GPU 에 직접 만든다. VLM 은 bf16, expert·투영층은 fp32 (학습 대상)."""
    from safetensors import safe_open

    cfg = PI05Config(dtype="float32", chunk_size=50, n_action_steps=50, max_state_dim=32, max_action_dim=32,
                     num_inference_steps=5)
    with torch.device(device):
        model = PI05Pytorch(cfg)
    pwe = model.paligemma_with_expert
    pwe.gemma_expert.lm_head = None  # expert 는 토큰을 내지 않는다 (263M 절약)
    # PaliGemma 는 lm_head 와 embed_tokens 를 묶는다(tie_word_embeddings). 직접 생성하면 안 묶여 있어
    # 527M 짜리 사본이 둘이 된다 — 묶어서 하나로 만든다. 체크포인트에는 lm_head 만 있다.
    pwe.paligemma.lm_head.weight = pwe.paligemma.model.language_model.embed_tokens.weight
    torch.cuda.empty_cache()
    params = dict(model.named_parameters())
    loaded = set()
    with safe_open(PI05_WEIGHTS, framework="pt", device=str(device)) as f:
        for k in f.keys():
            if k.startswith("paligemma_with_expert.gemma_expert.lm_head"):
                continue
            if not with_expert and (k.startswith("paligemma_with_expert.gemma_expert") or
                                    k.split(".")[0] in ("action_in_proj", "action_out_proj", "time_mlp_in",
                                                        "time_mlp_out")):
                continue
            t = f.get_tensor(k)
            targets = [k]
            if k == "paligemma_with_expert.paligemma.lm_head.weight":
                # 묶인 파라미터는 named_parameters 에 이름 하나로만 나온다 (모듈 등록 순서에 따라 어느 쪽인지 바뀐다)
                targets = [k, "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"]
            for tk in targets:
                if tk not in params:
                    continue
                p = params[tk]
                assert p.shape == t.shape, (tk, p.shape, t.shape)
                with torch.no_grad():
                    p.copy_(t.to(p.dtype))
                loaded.add(tk)
            del t
    missing = [k for k in params if k not in loaded]
    if not with_expert:
        missing = [k for k in missing if "gemma_expert" not in k and not k.startswith(("action_", "time_mlp"))]
    assert not missing, f"가중치 누락 {len(missing)}: {missing[:5]}"

    pwe.paligemma.to(torch.bfloat16)
    for p in pwe.paligemma.parameters():
        p.requires_grad_(False)
    pwe.paligemma.eval()
    for m in (pwe.paligemma.model.language_model, pwe.gemma_expert.model):
        m.config._attn_implementation = attn
    if not with_expert:
        pwe.gemma_expert = None
        model.action_in_proj = model.action_out_proj = model.time_mlp_in = model.time_mlp_out = None
    torch.cuda.empty_cache()
    return model


def images_to_model(x: torch.Tensor) -> list[torch.Tensor]:
    """[B, V, H, W, 3] uint8 → V 개의 [B, 3, H, W] float, SigLIP 입력 범위 [-1, 1]."""
    x = x.permute(1, 0, 4, 2, 3).float().div_(127.5).sub_(1.0)
    return list(x.unbind(0))


def state_string(state: np.ndarray) -> str:
    """π0.5 식 이산 state: [-1,1] → 256 구간 (openpi PaligemmaTokenizer)."""
    s = np.clip(state, -1.0, 1.0)
    d = np.digitize(s, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    return " ".join(map(str, d.tolist()))


class LLPolicy(nn.Module):
    """π0.7 컨텍스트 저수준 정책 (π0.5 base, expert 학습)."""

    def __init__(self, device: torch.device, max_text_len: int = 256):
        super().__init__()
        self.model = build_pi05(device, with_expert=True)
        self.tok = load_tokenizer()
        self.device = device
        self.max_text_len = max_text_len
        self.action_dim = 23

    def trainable_parameters(self):
        return [p for n, p in self.model.named_parameters() if p.requires_grad]

    def tokenize(self, prompts: list[str], states: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        texts = [f"{p} State: {state_string(s)};\nAction: " for p, s in zip(prompts, states)]
        enc = self.tok(texts, padding="longest", truncation=True, max_length=self.max_text_len,
                       return_tensors="pt")
        if int(enc["attention_mask"].sum(1).max()) >= self.max_text_len and not getattr(self, "_warned", False):
            # 잘리면 끝의 "State: …;\nAction: " 가 사라진다. 실측 최대 170 토큰 (2026-09-18) 이라 지금은 안 걸린다.
            import logging
            logging.warning(f"LL 프롬프트가 {self.max_text_len} 토큰에서 잘렸다 — max_text_len 을 늘릴 것")
            self._warned = True
        return enc["input_ids"].to(self.device), enc["attention_mask"].bool().to(self.device)

    @torch.no_grad()
    def encode_prefix(self, images: torch.Tensor, prompts: list[str], states: np.ndarray,
                      goal_images: torch.Tensor | None = None):
        """prefix(관측 3뷰 + 텍스트 [+ 서브골 3뷰]) 를 VLM 에 한 번 통과시켜 KV 캐시를 만든다."""
        pwe = self.model.paligemma_with_expert
        tokens, tmask = self.tokenize(prompts, states)
        B = tokens.shape[0]
        embs, pads, att = [], [], []
        for img in images_to_model(images.to(self.device, non_blocking=True)):
            e = pwe.embed_image(img.to(torch.bfloat16))
            embs.append(e)
            pads.append(torch.ones(B, e.shape[1], dtype=torch.bool, device=self.device))
            att += [0] * e.shape[1]
        le = pwe.embed_language_tokens(tokens)
        embs.append(le)
        pads.append(tmask)
        att += [0] * le.shape[1]
        if goal_images is not None:
            for j, img in enumerate(images_to_model(goal_images.to(self.device, non_blocking=True))):
                e = pwe.embed_image(img.to(torch.bfloat16))
                embs.append(e)
                pads.append(torch.ones(B, e.shape[1], dtype=torch.bool, device=self.device))
                att += ([1] if j == 0 else [0]) + [0] * (e.shape[1] - 1)
        embs = torch.cat(embs, 1).to(torch.bfloat16)
        pad = torch.cat(pads, 1)
        att = torch.tensor(att, dtype=torch.bool, device=self.device)[None].expand(B, -1)
        att2d = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, 1) - 1
        mask4d = torch.where(att2d[:, None], 0.0, MASK_VALUE).to(torch.bfloat16)
        _, kv = pwe.forward(attention_mask=mask4d, position_ids=pos, past_key_values=None,
                            inputs_embeds=[embs, None], use_cache=True)
        return pad, kv

    def expert_velocity(self, pad: torch.Tensor, kv, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """캐시된 prefix 위에서 action expert 한 번 — v(x_t, t). 학습·추론 공용."""
        return self.model.denoise_step(prefix_pad_masks=pad, past_key_values=kv, x_t=x_t, timestep=t)

    def loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        images = batch["images"]
        goal = batch.get("goal_images")
        pad, kv = self.encode_prefix(images, batch["prompt"], batch["state"].numpy(), goal)
        a = batch["actions"].to(self.device)  # [B, H, 23]
        is_pad = batch["action_is_pad"].to(self.device)
        B, H, D = a.shape
        actions = F.pad(a, (0, 32 - D))
        noise = torch.randn_like(actions)
        t = self.model.sample_time(B, self.device)
        x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
        u_t = noise - actions
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_t = self.expert_velocity(pad, kv, x_t, t)
        err = (v_t.float() - u_t)[..., :D].pow(2)  # 패딩 차원(23..31)은 손실에서 뺀다
        w = (~is_pad).float()[..., None]
        loss = (err * w).sum() / (w.sum() * D).clamp_min(1.0)
        return loss, {"loss": loss.detach()}

    @torch.no_grad()
    def sample_actions(self, images, prompts, states, goal_images=None, num_steps: int = 5,
                       noise: torch.Tensor | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
        """noise([B, 50, 32]) 나 generator 를 주면 초기 노이즈가 고정된다 (검증용).
        둘 다 None 이면 전역 RNG 에서 뽑는다 — 서빙 경로(serve_pi07.py)가 쓰는 기본 동작."""
        pad, kv = self.encode_prefix(images, prompts, states, goal_images)
        B = pad.shape[0]
        if noise is None:
            x = torch.randn(B, 50, 32, device=self.device, generator=generator)
        else:
            assert noise.shape == (B, 50, 32), noise.shape
            x = noise.to(self.device, torch.float32)
        dt = -1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), 1.0 + i * dt, device=self.device)
            import copy
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = self.expert_velocity(pad, copy.deepcopy(kv), x, t)
            x = x + dt * v.float()
        return x[..., :self.action_dim]

    def trainable_state_dict(self) -> dict:
        return {n: p.detach().cpu() for n, p in self.model.named_parameters() if p.requires_grad}

    def load_trainable(self, sd: dict):
        params = dict(self.model.named_parameters())
        for n, v in sd.items():
            params[n].data.copy_(v.to(params[n].device, params[n].dtype))


# ---------------------------------------------------------------- HL

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / r
        dev = base.weight.device
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True  # 서빙 때 HL(LoRA 켬)과 LL(끔)이 같은 PaliGemma 를 나눠 쓴다

    @property
    def weight(self):  # pi_gemma 가 q_proj.weight.dtype 을 읽는다
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        z = F.linear(F.linear(self.drop(x).to(self.lora_a.dtype), self.lora_a), self.lora_b)
        return y + (z * self.scale).to(y.dtype)


def set_lora(module: nn.Module, enabled: bool):
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.enabled = enabled


def add_lora(module: nn.Module, names: tuple[str, ...], r: int, alpha: float, dropout: float) -> int:
    n = 0
    for full, m in list(module.named_modules()):
        for child_name, child in list(m.named_children()):
            if isinstance(child, nn.Linear) and child_name in names:
                setattr(m, child_name, LoRALinear(child, r, alpha, dropout))
                n += 1
    return n


class HLPolicy(nn.Module):
    """고수준 정책: (3뷰, 태스크, 이력) → subtask 문장. PaliGemma(π0.5 base) + LoRA."""

    def __init__(self, device: torch.device, lora_r: int = 32, lora_alpha: float = 32.0, lora_dropout: float = 0.05,
                 grad_ckpt: bool = True, vlm: nn.Module | None = None):
        super().__init__()
        if vlm is None:
            pi05 = build_pi05(device, with_expert=False)
            vlm = pi05.paligemma_with_expert.paligemma  # PaliGemmaForConditionalGenerationWithPiGemma
            del pi05
        self.vlm = vlm
        self.tok = load_tokenizer()
        self.device = device
        lm = self.vlm.model.language_model
        n = add_lora(lm, ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
                     lora_r, lora_alpha, lora_dropout)
        self.n_lora = n
        if grad_ckpt:
            # HF 는 층 단위 플래그다 (transformers.modeling_layers.GradientCheckpointingLayer)
            import functools
            import torch.utils.checkpoint as cp
            for layer in lm.layers:
                layer.gradient_checkpointing = True
                layer._gradient_checkpointing_func = functools.partial(cp.checkpoint, use_reentrant=False)
        self.eos = self.tok.eos_token_id

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _embed(self, images: torch.Tensor, ids: torch.Tensor, ids_mask: torch.Tensor, ar: torch.Tensor):
        B = ids.shape[0]
        embs, pads, att = [], [], []
        with torch.no_grad():
            for img in images_to_model(images.to(self.device, non_blocking=True)):
                e = self.vlm.model.get_image_features(img.to(torch.bfloat16)).pooler_output
                embs.append(e.to(torch.bfloat16))
                pads.append(torch.ones(B, e.shape[1], dtype=torch.bool, device=self.device))
                att.append(torch.zeros(B, e.shape[1], dtype=torch.bool, device=self.device))
            le = self.vlm.model.language_model.embed_tokens(ids)
        embs.append(le.to(torch.bfloat16))
        pads.append(ids_mask)
        att.append(ar)
        return torch.cat(embs, 1), torch.cat(pads, 1), torch.cat(att, 1)

    def build_ids(self, prompts: list[str], targets: list[str] | None):
        """[BOS] prompt (양방향) | target + EOS (인과). 오른쪽 패딩."""
        seqs, ars = [], []
        for i, p in enumerate(prompts):
            pid = self.tok(p, add_special_tokens=True)["input_ids"]
            tid = []
            if targets is not None:
                tid = self.tok(" " + targets[i], add_special_tokens=False)["input_ids"] + [self.eos]
            seqs.append(pid + tid)
            ars.append([0] * len(pid) + [1] * len(tid))
        L = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), L, dtype=torch.long)
        m = torch.zeros(len(seqs), L, dtype=torch.bool)
        ar = torch.zeros(len(seqs), L, dtype=torch.bool)
        plen = []
        for i, (s, a) in enumerate(zip(seqs, ars)):
            ids[i, :len(s)] = torch.tensor(s)
            m[i, :len(s)] = True
            ar[i, :len(a)] = torch.tensor(a, dtype=torch.bool)
            plen.append(len(s) - sum(a))
        return ids.to(self.device), m.to(self.device), ar.to(self.device), plen

    def loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        ids, m, ar, plen = self.build_ids(batch["prompt"], batch["target"])
        embs, pad, att = self._embed(batch["images"], ids, m, ar)
        n_img = embs.shape[1] - ids.shape[1]
        att2d = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, 1) - 1
        mask4d = torch.where(att2d[:, None], 0.0, MASK_VALUE).to(torch.bfloat16)
        lm = self.vlm.model.language_model
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = lm(inputs_embeds=embs, attention_mask=mask4d, position_ids=pos, use_cache=False).last_hidden_state
        # 목표 토큰 j 는 위치 j-1 의 은닉 상태로 예측한다. 목표 위치에서만 LM head 를 계산한다.
        B = ids.shape[0]
        rows, cols, labels = [], [], []
        for b in range(B):
            L = int(m[b].sum())
            for j in range(plen[b], L):
                rows.append(b)
                cols.append(n_img + j - 1)
                labels.append(int(ids[b, j]))
        rows = torch.tensor(rows, device=self.device)
        cols = torch.tensor(cols, device=self.device)
        labels = torch.tensor(labels, device=self.device)
        hs = h[rows, cols]
        W = lm.embed_tokens.weight
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = hs @ W.t()
        loss = F.cross_entropy(logits.float(), labels)
        acc = (logits.argmax(-1) == labels).float().mean()
        return loss, {"loss": loss.detach(), "tok_acc": acc.detach()}

    def build_trie(self, texts: list[str]) -> dict:
        """허용 문장 집합의 토큰 트라이 (정답 토큰화와 같게: " " + 문장 + EOS)."""
        root: dict = {}
        for t in texts:
            node = root
            for tid in self.tok(" " + t, add_special_tokens=False)["input_ids"] + [self.eos]:
                node = node.setdefault(tid, {})
        return root

    @torch.no_grad()
    def generate(self, images: torch.Tensor, prompts: list[str], max_new: int = 32,
                 allowed: list[dict | None] | None = None) -> list[str]:
        """탐욕 디코딩. 표본마다 따로 돈다 (평가·서빙용, 배치 크기 작음).

        allowed[b] 가 트라이(build_trie)면 그 문장들 중 하나만 나오도록 매 스텝 허용 토큰 밖을 막는다.
        런타임에서 쓴다 — 어휘 밖 문장은 LL 이 학습한 적 없고 WM 텍스트 캐시에도 없다."""
        outs = []
        lm = self.vlm.model.language_model
        W = lm.embed_tokens.weight
        for b in range(len(prompts)):
            node = allowed[b] if allowed is not None else None
            ids, m, ar, _ = self.build_ids([prompts[b]], None)
            embs, pad, att = self._embed(images[b:b + 1], ids, m, ar)
            T = embs.shape[1]
            mask4d = torch.zeros(1, 1, T, T, dtype=torch.bfloat16, device=self.device)
            pos = torch.arange(T, device=self.device)[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = lm(inputs_embeds=embs, attention_mask=mask4d, position_ids=pos, use_cache=True)
            kv, h = o.past_key_values, o.last_hidden_state[:, -1]
            gen = []
            for k in range(max_new):
                logits = (h.float() @ W.float().t())[0]
                if node is not None:
                    cand = torch.tensor(list(node.keys()), device=self.device)
                    nxt = int(cand[logits[cand].argmax()])
                    node = node[nxt]
                else:
                    nxt = int(logits.argmax())
                if nxt == self.eos:
                    break
                gen.append(nxt)
                e = lm.embed_tokens(torch.tensor([[nxt]], device=self.device)).to(torch.bfloat16)
                Lk = T + k + 1
                mask4d = torch.zeros(1, 1, 1, Lk, dtype=torch.bfloat16, device=self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = lm(inputs_embeds=e, attention_mask=mask4d, position_ids=torch.tensor([[T + k]], device=self.device),
                           past_key_values=kv, use_cache=True)
                kv, h = o.past_key_values, o.last_hidden_state[:, -1]
            outs.append(self.tok.decode(gen, skip_special_tokens=True).strip())
        return outs

    def trainable_state_dict(self) -> dict:
        return {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable(self, sd: dict):
        params = dict(self.named_parameters())
        for n, v in sd.items():
            params[n].data.copy_(v.to(params[n].device, params[n].dtype))
