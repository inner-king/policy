"""`configs/pi0_patch.py` 검증 — CU 를 쓰기 전에 로컬에서 끝낸다.

    JAX_PLATFORMS=cpu third_party/openpi/.venv/bin/python scripts/check_pi0_patch.py

세 가지를 본다.

1. **프리픽스를 쪼개도 값이 같은가.** 다표본 패치의 전제다. 프리픽스의 `ar_mask`
   가 전부 False 라 프리픽스 출력이 서픽스에 의존하지 않는다는 것이 근거인데,
   말로만 두지 않고 수치로 대조한다. 여기가 틀리면 학습이 조용히 망가진다.
2. **패치된 `compute_loss` 의 반환 모양이 openpi 규약과 같은가** (`[b, ah]`).
   `scripts/train.py:150` 이 여기에 `jnp.mean` 을 걸기 때문에 모양이 다르면
   손실 스케일이 어긋난다.
3. **N 을 늘려도 손실의 기댓값이 안 변하는가.** N 은 분산을 줄이는 장치이지
   손실 정의를 바꾸는 것이 아니다.

모델은 `dummy` 변형(width 64 / depth 4)이라 CPU 에서 몇 초면 돈다.

⚠️ **시각 인코더(SigLIP 400M)는 고정 토큰으로 대체한다.** `dummy` 변형은 LLM 만
줄이고 `PaliGemma.img` 는 그대로라, CPU 에서 이것 때문에 2분을 넘겨 못 돈다.
이 검사의 대상은 **LLM 의 프리픽스/서픽스 분리**이지 시각 인코더가 아니므로,
이미지 토큰을 고정 배열로 두면 검사의 뜻은 그대로면서 몇 초로 줄어든다.
"""
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))

import einops
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at

import pi0_patch

AH, AD, TOK = 8, 32, 24


def build():
    cfg = pi0_config.Pi0Config(
        pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
        action_horizon=AH, action_dim=AD, max_token_len=TOK,
    )
    model = cfg.create(jax.random.key(0))
    model.eval()
    rng = jax.random.key(1)
    b = 2

    # 시각 인코더를 고정 토큰으로 대체한다 (위 주석 참조).
    # 두 경로가 **같은** 프리픽스 토큰을 받아야 비교가 성립하므로 값을 고정한다.
    width = model.PaliGemma.llm.config.width if hasattr(model.PaliGemma.llm, "config") else 64
    img_tokens = jax.random.normal(jax.random.key(7), (b, 8, width)) * 0.1
    model.PaliGemma.img = lambda x, train=False: (img_tokens, None)
    obs = _model.Observation(
        images={k: jax.random.uniform(jax.random.key(i), (b, 224, 224, 3), minval=-1, maxval=1)
                for i, k in enumerate(["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"])},
        image_masks={k: jnp.ones((b,), dtype=bool)
                     for k in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]},
        state=jax.random.normal(jax.random.key(9), (b, AD)),
        tokenized_prompt=jax.random.randint(jax.random.key(10), (b, TOK), 1, 100),
        tokenized_prompt_mask=jnp.ones((b, TOK), dtype=bool),
    )
    actions = jax.random.normal(jax.random.key(11), (b, AH, AD))
    return cfg, model, obs, actions, rng


def v_joint(model, obs, x_t, time):
    """openpi 원본 방식 — 프리픽스와 서픽스를 한 번에 통과시킨다."""
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms = model.embed_suffix(obs, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions,
        adarms_cond=[None, adarms])
    return model.action_out_proj(suffix_out[:, -model.action_horizon:])


def v_split(model, obs, x_t, time):
    """패치 방식 — 프리픽스로 KV 캐시를 만들고 서픽스만 따로 통과시킨다."""
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    _, kv = model.PaliGemma.llm([prefix_tokens, None],
                                mask=_pi0.make_attn_mask(prefix_mask, prefix_ar_mask),
                                positions=jnp.cumsum(prefix_mask, axis=1) - 1)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms = model.embed_suffix(obs, x_t, time)
    pre = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full = jnp.concatenate([pre, _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)], axis=-1)
    pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (po, suffix_out), _ = model.PaliGemma.llm([None, suffix_tokens], mask=full, positions=pos,
                                              kv_cache=kv, adarms_cond=[None, adarms])
    assert po is None
    return model.action_out_proj(suffix_out[:, -model.action_horizon:])


def main() -> int:
    cfg, model, obs, actions, rng = build()
    obs_p = _model.preprocess_observation(None, obs, train=False)
    time = jnp.full((actions.shape[0],), 0.37)
    x_t = jax.random.normal(jax.random.key(12), actions.shape)

    print("=" * 66)
    print("1) 프리픽스를 쪼개도 값이 같은가")
    a = np.asarray(v_joint(model, obs_p, x_t, time), dtype=np.float64)
    b = np.asarray(v_split(model, obs_p, x_t, time), dtype=np.float64)
    absd = np.abs(a - b).max()
    rel = absd / max(np.abs(a).max(), 1e-12)
    print(f"   한번에 통과 vs 캐시 재사용   최대 절대차 {absd:.3e}  상대차 {rel:.3e}")
    # bfloat16 연산이라 완전 동일은 기대하지 않는다. 상대차 1e-2 면 같은 값으로 본다.
    ok1 = rel < 1e-2
    print(f"   -> {'통과' if ok1 else '🔴 실패 — 다표본 패치를 쓰면 안 된다'}")

    print()
    print("2) 반환 모양이 openpi 규약과 같은가")
    orig = _pi0.Pi0.compute_loss
    loss_orig = orig(model, rng, obs, actions, train=False)
    patched1 = pi0_patch._make_compute_loss(1, None)
    patched8 = pi0_patch._make_compute_loss(8, None)
    l1 = patched1(model, rng, obs, actions, train=False)
    l8 = patched8(model, rng, obs, actions, train=False)
    print(f"   원본 {loss_orig.shape}  패치 N=1 {l1.shape}  패치 N=8 {l8.shape}"
          f"   (기대 {(actions.shape[0], AH)})")
    ok2 = loss_orig.shape == l1.shape == l8.shape == (actions.shape[0], AH)
    print(f"   -> {'통과' if ok2 else '🔴 실패'}")

    print()
    print("3) N 을 늘려도 손실의 기댓값이 그대로인가 (분산만 줄어야 한다)")
    keys = jax.random.split(jax.random.key(99), 16)
    m1 = np.array([float(jnp.mean(patched1(model, k, obs, actions))) for k in keys])
    m8 = np.array([float(jnp.mean(patched8(model, k, obs, actions))) for k in keys])
    mo = np.array([float(jnp.mean(orig(model, k, obs, actions))) for k in keys])
    print(f"   원본     평균 {mo.mean():.4f}  표준편차 {mo.std():.4f}")
    print(f"   패치 N=1 평균 {m1.mean():.4f}  표준편차 {m1.std():.4f}")
    print(f"   패치 N=8 평균 {m8.mean():.4f}  표준편차 {m8.std():.4f}  <- 표준편차가 줄어야 정상")
    ok3 = abs(m8.mean() - mo.mean()) < 0.5 * mo.std() and m8.std() < m1.std()
    print(f"   -> {'통과' if ok3 else '🔴 실패'}")

    print()
    print("=" * 66)
    ok = ok1 and ok2 and ok3
    print("전체:", "통과" if ok else "🔴 실패")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
