"""지시문을 바꾸면 정책의 예측이 실제로 달라지는가 — 시뮬레이터 없이 재는 검사.

    JAX_PLATFORMS=cuda HF_LEROBOT_HOME=$PWD/data/lerobot HF_HUB_OFFLINE=1 \
    third_party/openpi/.venv/bin/python scripts/probe_language.py \
        --ckpt checkpoints/pi05_cooking_top20_v3/run3/12499

## 왜 이걸 먼저 하나

이 프로젝트의 실패는 늘 "손실은 잘 내려가는데 지시문을 무시한다" 였다.
손실 곡선으로는 그것을 알 수 없고, 시뮬레이터 rollout 은 준비가 오래 걸린다.
그 사이를 메우는 **몇 분짜리 사전 검사**다. rollout 을 대신하지 않는다 —
결과가 나쁘면 rollout 을 돌릴 필요가 없어진다는 뜻에서 앞에 둔다.

## 어떻게 재나

진짜 학습 데이터에서 배치를 하나 뽑아, **관측(카메라 3장 + 상태)은 그대로 두고
지시문 토큰만 배치 안에서 서로 바꿔 끼운다.** 그러면 각 샘플이 "다른 샘플의
지시문" 을 받는다. 그 상태로 예측한 동작을 정답과 비교한다.

- 정책이 지시문을 쓴다면: 남의 지시문을 받았을 때 **오차가 커져야** 한다.
- 지시문을 무시한다면: 오차가 그대로다.

**잡음은 완전히 고정한다.** pi0.5 는 잡음에서 동작을 만들어 내므로 실행마다
결과가 다른데, `sample_actions(noise=...)` 로 같은 잡음을 넣어 그 변동을 없앤다.
그래야 차이를 전부 지시문 탓으로 돌릴 수 있다. 비교를 위해 **잡음만 바꿨을 때**의
변동도 따로 잰다 — 지시문 효과가 잡음 효과보다 확실히 커야 의미가 있다.

## 🔴 2026-09-25 — "남의 지시문" 은 **남의 로봇 자세까지** 넣고 있었다

pi0.5 는 상태를 연속값으로 넣지 않는다(`pi0.py:151` 의 `state_proj` 는 pi0 전용).
상태 25개를 정수로 바꿔 **지시문과 같은 문자열**에 넣는다(`tokenizer.py:26-29`):

    "Task: {지시문}, State: {정수 25개};\\nAction: "

그래서 토큰을 통째로 밀면 지시문과 함께 **자세도 남의 것**이 된다. 예전의 "지시문
효과"(run7 +277.9%, HANDOFF 40.2) 는 "지시문 + 자세를 바꿨을 때" 의 값이다.
이제는 토큰을 글자로 되돌려 **지시문만** 바꿔 끼우고 자세는 제 것을 둔다(아래
`instr_swapped`). 옛 방식도 "통째 교체" 로 같이 재서 과거 값과 이어 볼 수 있게 한다.
되돌린 글자로 다시 만든 토큰이 원래 토큰과 **한 개라도 다르면 멈춘다** — 그래야
바꿔 끼운 결과를 믿을 수 있다.

배치 캐시도 고쳤다. 예전에는 파일 이름이 고정(`batch.npz`)이고 **어느 설정에서
뽑았는지 확인하지 않아서**, 32종 설정으로 돌려도 20종 시절 배치를 조용히 썼다.
이제 이름에 설정과 배치 크기를 넣고, 설정 이름을 파일 안에 적어 두고 대조한다.
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--config", default="pi05_cooking_top20_v3")
ap.add_argument("--ckpt", default="checkpoints/pi05_cooking_top20_v3/run3/12499")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--num-steps", type=int, default=10, help="flow matching 적분 스텝")
ap.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
ap.add_argument("--chunk", type=int, default=1,
                help=("한 번에 GPU 로 보낼 샘플 수. 8GB 카드는 가중치만 6.81GB 라 "
                      "남는 자리가 0.6GB 뿐이다. 배치 8 을 통째로 넣으면 SigLIP 에서 "
                      "OOM 이 난다(실측: 121.5MiB 요청 실패). 나눠 넣고 이어 붙인다."))
ap.add_argument("--batch-cache", default=None,
                help=("뽑은 배치를 저장해 두고 다시 쓴다. 데이터 로더를 세우는 데만 "
                      "4분이 걸려서, 설정을 바꿔 다시 돌릴 때마다 그 값을 치르지 않게 한다. "
                      "기본값은 outputs/probe/batch_<설정>_b<배치>.npz — 설정마다 따로 둔다."))
ap.add_argument("--json-out", default=None,
                help="결과를 JSON 으로도 남긴다 (여러 체크포인트를 표로 모을 때)")
ap.add_argument("--dry-run", action="store_true",
                help="배치와 지시문 바꿔 끼우기 검사까지만 하고 모델은 안 돌린다")
A = ap.parse_args()

import os                     # noqa: E402
if A.device == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import lerobot_patch          # noqa: E402
import task_subset            # noqa: E402
import cooking_config         # noqa: E402

lerobot_patch.apply()
task_subset.apply()
cooking_config.register()

import dataclasses            # noqa: E402
import json                   # noqa: E402
import re                     # noqa: E402
import jax                    # noqa: E402
import jax.numpy as jnp       # noqa: E402
from openpi.models import model as _model            # noqa: E402
from openpi.models import tokenizer as _tokenizer    # noqa: E402
from openpi.shared import nnx_utils                  # noqa: E402
from openpi.training import config as _config        # noqa: E402
from openpi.training import data_loader as _dl       # noqa: E402

ACT_DIM = 23        # 패딩(32) 중 실제로 쓰는 차원


def main() -> int:
    cfg = dataclasses.replace(_config.get_config(A.config), batch_size=A.batch)
    print(f"설정 {cfg.name} / 배치 {A.batch} / 적분 스텝 {A.num_steps}")

    # --- 1. 진짜 학습 경로로 배치 하나 (한 번 뽑아 두고 재사용) ---
    cache = pathlib.Path(A.batch_cache or f"outputs/probe/batch_{A.config}_b{A.batch}.npz")
    if cache.exists():
        z = np.load(cache)
        # 설정마다 정규화 통계와 지시문 범위가 다르다. 다른 설정의 배치를 쓰면
        # 에러 없이 틀린 오차가 나온다 — 그래서 뽑은 설정을 적어 두고 대조한다.
        saved = str(z["config_name"]) if "config_name" in z.files else None
        if saved is None:
            print(f"⚠️ 옛 캐시다 — 어느 설정에서 뽑았는지 기록이 없다: {cache}")
        elif saved != A.config:
            print(f"🔴 이 배치는 '{saved}' 에서 뽑았는데 지금 설정은 '{A.config}' 다: {cache}")
            return 1
        # ⚠️ `image` / `image_mask` 는 **중첩 딕셔너리**다. 평평한 "image/xxx"
        # 키로 넣으면 `KeyError: 'image'` 가 난다. 그리고 jaxtyping 이 numpy
        # int64 를 거부하므로 jnp 로 바꿔서 넣는다.
        obs = _model.Observation.from_dict(jax.tree.map(jnp.asarray, {
            "image": {k[6:]: z[k] for k in z.files if k.startswith("image/")},
            "image_mask": {k[11:]: z[k] for k in z.files if k.startswith("image_mask/")},
            "state": z["state"],
            "tokenized_prompt": z["tokenized_prompt"],
            "tokenized_prompt_mask": z["tokenized_prompt_mask"],
        }))
        act = z["actions"]
        print(f"배치 캐시 사용: {cache}")
    else:
        loader = _dl.create_data_loader(cfg, num_batches=1, shuffle=True)
        obs, act = next(iter(loader))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, config_name=np.asarray(A.config), actions=np.asarray(act),
            state=np.asarray(obs.state),
            tokenized_prompt=np.asarray(obs.tokenized_prompt),
            tokenized_prompt_mask=np.asarray(obs.tokenized_prompt_mask),
            **{f"image/{k}": np.asarray(v) for k, v in obs.images.items()},
            **{f"image_mask/{k}": np.asarray(v) for k, v in obs.image_masks.items()},
        )
        print(f"배치 캐시 저장: {cache}")
    gt = np.asarray(act, dtype=np.float64)[:, :, :ACT_DIM]
    b = gt.shape[0]

    tok = np.asarray(obs.tokenized_prompt)
    msk = np.asarray(obs.tokenized_prompt_mask)
    L = tok.shape[1]

    # --- 1-1. 토큰을 글자로 되돌려 지시문과 자세(State)를 떼어 낸다 ---
    sp = _tokenizer.PaligemmaTokenizer(max_len=L)._tokenizer
    pat = re.compile(r"^Task: (.*), State: ([-0-9 ]*);\nAction: $", re.S)
    instr, states = [], []
    for i in range(b):
        text = sp.decode([int(t) for t in tok[i][msk[i]]])
        m = pat.match(text)
        if not m:
            print(f"🔴 샘플 {i} 의 프롬프트가 예상한 형식이 아니다: {text!r}")
            return 1
        instr.append(m.group(1))
        states.append(m.group(2))

    def encode(ins, st):
        """`PaligemmaTokenizer.tokenize` 와 같은 모양(BOS + 뒤쪽 0 채움)으로 만든다."""
        ids = sp.encode(f"Task: {ins}, State: {st};\nAction: ", add_bos=True)
        if len(ids) > L:
            print(f"⚠️ 바꿔 끼운 프롬프트가 {len(ids)}토큰 > {L} — 뒤가 잘린다")
            ids = ids[:L]
        t = np.zeros(L, dtype=tok.dtype)
        t[:len(ids)] = ids
        m_ = np.zeros(L, dtype=msk.dtype)
        m_[:len(ids)] = True
        return t, m_

    # 되돌린 글자로 원래 토큰이 **한 개도 틀리지 않고** 다시 나와야 한다.
    # 하나라도 다르면 바꿔 끼운 프롬프트가 학습 때와 다른 모양이라 결과를 못 믿는다.
    for i in range(b):
        t, m_ = encode(instr[i], states[i])
        if not (np.array_equal(t, tok[i]) and np.array_equal(m_, msk[i])):
            print(f"🔴 샘플 {i}: 되돌린 글자로 만든 토큰이 원래와 다르다 — 지시문만 바꾸기를 믿을 수 없다")
            return 1
    print(f"토큰 왕복 검사 통과 — {b}개 전부 원래 토큰을 그대로 다시 만든다")

    new12 = {s.lower() for s in cooking_config.NEW12}
    top20 = {s.lower() for s in cooking_config.TOP20}
    kind = ["신규" if s.lower() in new12 else "기존" if s.lower() in top20 else "그밖"
            for s in instr]
    is_new = np.array([k == "신규" for k in kind])
    is_old = np.array([k == "기존" for k in kind])
    for i in range(b):
        print(f"  [{i:2d}] {kind[i]}  {instr[i]}")
    print(f"  → 신규 12종 {is_new.sum()}개 / 기존 20종 {is_old.sum()}개 / 그 밖 {b - is_new.sum() - is_old.sum()}개")

    # 배치 안에서 지시문이 실제로 서로 다른지부터 본다. 다 같으면 이 검사가
    # 무의미해진다. (예전에는 토큰 전체로 셌는데, 자세 숫자가 샘플마다 달라서
    # 지시문이 같아도 "다르다" 로 셌다.)
    differs = {k: np.array([instr[(i - k) % b] != instr[i] for i in range(b)]) for k in (1, 2)}
    print(f"배치 안 서로 다른 지시문 {len(set(instr))}/{b}개, 밀었을 때 지시문이 바뀌는 샘플 "
          f"한 칸 {differs[1].sum()}/{b}개 · 두 칸 {differs[2].sum()}/{b}개")
    if differs[1].sum() < b // 2:
        print("🔴 배치에 같은 지시문이 너무 많다 — 배치를 키워서 다시 돌릴 것")
        return 1

    def instr_swapped(k: int):
        """지시문만 k 칸 민 **남의 것**으로, 자세(State)는 **제 것** 그대로.
        `jnp.roll(x, k)[i] == x[i-k]` 와 같은 짝을 쓴다 — 통째 교체와 상대가 같다."""
        pairs = [encode(instr[(i - k) % b], states[i]) for i in range(b)]
        return dataclasses.replace(
            obs,
            tokenized_prompt=jnp.asarray(np.stack([p[0] for p in pairs])),
            tokenized_prompt_mask=jnp.asarray(np.stack([p[1] for p in pairs])),
        )

    if A.dry_run:
        t, m_ = encode(instr[(0 - 1) % b], states[0])
        print("\n예) 샘플 0 에 넣을 '지시문만 교체' 프롬프트:")
        print("   ", repr(sp.decode([int(x) for x in t[m_]])))
        print("(--dry-run: 모델은 돌리지 않았다)")
        return 0

    # --- 2. 모델 ---
    params = _model.restore_params(ROOT / A.ckpt / "params",
                                   restore_type=np.ndarray, dtype=jnp.bfloat16)
    params = jax.tree.map(jnp.asarray, params)
    model = cfg.model.load(params)
    # 🔴 반드시 한 번 묶어 컴파일해서 쓴다 (정책 서버와 같은 방식, openpi `policy.py:64`).
    # `sample_actions` 는 부를 때마다 안에서 `step`/`cond` 를 새로 만들어
    # `lax.while_loop` 에 넘기므로, 그냥 부르면 **호출마다 추론부 전체를 새로
    # 컴파일**하고 그 결과가 메모리에 쌓인다. 2026-09-25 에 57번째 호출에서
    # `LLVM compilation error: Cannot allocate memory` 로 죽었다(종료코드 139).
    sample = nnx_utils.module_jit(model.sample_actions)
    print(f"체크포인트 {A.ckpt}  ({jax.devices()[0].platform})")

    # --- 3. 조건들 ---
    def shifted(k: int):
        """지시문 토큰만 k 칸 민 관측. 나머지(이미지·상태)는 그대로."""
        return dataclasses.replace(
            obs,
            tokenized_prompt=jnp.roll(obs.tokenized_prompt, k, axis=0),
            tokenized_prompt_mask=jnp.roll(obs.tokenized_prompt_mask, k, axis=0),
        )

    # 잡음을 명시적으로 고정한다 — 이걸 안 하면 조건별 차이에 잡음이 섞인다.
    noise0 = jax.random.normal(jax.random.key(0), (b, cfg.model.action_horizon,
                                                   cfg.model.action_dim))
    noise1 = jax.random.normal(jax.random.key(1), noise0.shape)

    def _slice(o, s):
        return dataclasses.replace(
            o,
            images={k: v[s] for k, v in o.images.items()},
            image_masks={k: v[s] for k, v in o.image_masks.items()},
            state=o.state[s], tokenized_prompt=o.tokenized_prompt[s],
            tokenized_prompt_mask=o.tokenized_prompt_mask[s],
        )

    def run(o, noise):
        """`--chunk` 개씩 잘라 추론하고 이어 붙인다. 잘라도 결과는 같다 —
        샘플끼리 섞이는 연산이 없다(어텐션은 샘플 안에서만 일어난다)."""
        outs = []
        for i in range(0, b, A.chunk):
            s = slice(i, min(i + A.chunk, b))
            out = sample(jax.random.key(0), _slice(o, s),
                         num_steps=A.num_steps, noise=noise[s])
            outs.append(np.asarray(out, dtype=np.float64))
            print(f"    {s.stop}/{b}", end="\r", flush=True)
        return np.concatenate(outs, 0)[:, :, :ACT_DIM]

    print("\n예측 중 (조건 6개)...", flush=True)
    pred_true = run(obs, noise0)             # 제 지시문 + 제 자세
    pred_i1 = run(instr_swapped(1), noise0)  # 남의 지시문 + 제 자세   ← 새 기준
    pred_i2 = run(instr_swapped(2), noise0)
    pred_r1 = run(shifted(1), noise0)        # 남의 지시문 + 남의 자세 ← 옛 방식(통째)
    pred_r2 = run(shifted(2), noise0)
    pred_noise = run(obs, noise1)            # 제 것 그대로 + 다른 잡음

    def err_s(p, sl=slice(None)):
        """샘플별 오차."""
        return np.abs(p[:, sl] - gt[:, sl]).mean(axis=(1, 2))

    def err(p, sl=slice(None)):
        return float(err_s(p, sl).mean())

    def diff(p, q, sl=slice(None)):
        return float(np.abs(p[:, sl] - q[:, sl]).mean())

    front = slice(0, 8)
    print("\n" + "=" * 74)
    print("정답과의 오차 (작을수록 잘 맞힌 것)")
    print("=" * 74)
    print(f"{'조건':<34}{'전체':>12}{'청크앞8':>12}")
    for name, p in (("제 지시문 (정상)", pred_true),
                    ("지시문만 교체 (한 칸) — 자세는 제 것", pred_i1),
                    ("지시문만 교체 (두 칸)", pred_i2),
                    ("통째 교체 (한 칸) — 옛 방식", pred_r1),
                    ("통째 교체 (두 칸)", pred_r2),
                    ("제 지시문 + 다른 잡음", pred_noise)):
        print(f"{name:<30}{err(p):>12.5f}{err(p, front):>12.5f}")

    e0_s = err_s(pred_true)

    def instr_effect(group):
        """지시문이 **실제로 바뀐** 샘플만 모아 오차가 얼마나 늘었나.
        (밀어도 옆 샘플과 지시문이 같으면 아무것도 안 바뀐 것이라 뺀다.)"""
        d, base = [], []
        for k, p in ((1, pred_i1), (2, pred_i2)):
            sel = differs[k] & group
            d += list(err_s(p)[sel] - e0_s[sel])
            base += list(e0_s[sel])
        if not d:
            return None, None, 0
        return float(np.mean(d)), float(np.mean(d) / np.mean(base)), len(d)

    everyone = np.ones(b, dtype=bool)
    lang, lang_rel, lang_n = instr_effect(everyone)
    lang_new = instr_effect(is_new)
    lang_old = instr_effect(is_old)
    e0, e1, e2, en = err(pred_true), err(pred_r1), err(pred_r2), err(pred_noise)
    full = (e1 + e2) / 2 - e0        # 옛 "지시문 효과" 와 같은 식 (지시문 + 자세를 바꿨을 때)
    noise_eff = abs(en - e0)         # 잡음만 바꿔서 달라진 양

    print("\n" + "=" * 74)
    print("예측끼리의 차이 (제 지시문 예측 대비)")
    print("=" * 74)
    print(f"{'지시문만 바꿨을 때':<30}{(diff(pred_true, pred_i1) + diff(pred_true, pred_i2)) / 2:>12.5f}")
    print(f"{'통째로 바꿨을 때':<30}{(diff(pred_true, pred_r1) + diff(pred_true, pred_r2)) / 2:>12.5f}")
    print(f"{'잡음만 바꿨을 때':<30}{diff(pred_true, pred_noise):>12.5f}")

    def pct(x):
        return "   —" if x is None else f"{x * 100:+.1f}%"

    print("\n" + "=" * 74)
    print(f"지시문 효과 (지시문만 교체 — 새 기준)  {lang:+.5f}   ({pct(lang_rel)})   [{lang_n}쌍]")
    print(f"   신규 12종 샘플                       {pct(lang_new[1]):>18}   [{lang_new[2]}쌍]")
    print(f"   기존 20종 샘플                       {pct(lang_old[1]):>18}   [{lang_old[2]}쌍]")
    print(f"통째 교체 효과 (옛 기준 — 지시문+자세)  {full:+.5f}   ({full / e0 * 100:+.1f}%)")
    print(f"잡음 효과 (오차 변동)                   {noise_eff:+.5f}   ({noise_eff / e0 * 100:+.1f}%)")
    print("=" * 74)
    if lang <= 0:
        print("🔴 남의 지시문을 줘도 오차가 안 늘었다 — **지시문을 안 보고 있다.**")
    elif lang < noise_eff:
        print("🟡 오차는 늘었지만 잡음 변동보다 작다 — 언어의 영향이 미약하다.")
    else:
        print("🟢 남의 지시문을 주면 오차가 뚜렷이 는다 — **지시문을 보고 있다.**")
    print("\n주의: 이 검사는 '지시문에 반응하는가'만 본다. '태스크를 해내는가'는"
          "\n      시뮬레이터 rollout 으로만 알 수 있다. 쌍 수가 적은 칸은 흔들린다.")

    if A.json_out:
        out = pathlib.Path(A.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "config": A.config, "ckpt": A.ckpt, "batch_cache": str(cache), "batch": b,
            "err": {"true": e0, "instr1": err(pred_i1), "instr2": err(pred_i2),
                    "full1": e1, "full2": e2, "noise": en},
            "instr_effect": {"all": [lang, lang_rel, lang_n],
                             "new12": list(lang_new), "top20": list(lang_old)},
            "full_effect": [full, full / e0], "noise_effect": [noise_eff, noise_eff / e0],
            "instructions": instr, "kind": kind,
        }, ensure_ascii=False, indent=1))
        print(f"결과 저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
