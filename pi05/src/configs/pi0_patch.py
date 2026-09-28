"""pi0.5 학습 손실에 붙이는 두 가지 — openpi 는 건드리지 않는다.

`third_party/openpi` 는 gitignore 대상(다시 clone 하는 대상)이라 그쪽 파일을
고치면 재현이 안 된다. 그래서 `Pi0.compute_loss` 를 런타임에 갈아 끼운다.
`scripts/train.py` 래퍼가 openpi 를 실행하기 전에 `apply()` 를 부른다.

## 1. 다표본 flow matching (`num_flow_samples`)

flow matching 학습은 매 스텝 **잡음 하나와 시간 하나**를 뽑아 "이 잡음에서 정답
쪽으로 가는 속도"를 맞힌다. 그런데 우리 설정에서 연산의 **99.5%** 는 프리픽스
(이미지 768 토큰 + 텍스트 150 토큰을 2.5B Gemma 에 통과시키는 것)이고, 액션
전문가는 32 토큰만 처리해 **0.5%** 다. 즉 비싼 계산을 해 놓고 그 결과를
잡음 표본 **하나**에 쓰고 버린다.

프리픽스를 한 번만 돌려 KV 캐시를 만들고 서픽스만 N 번 돌리면, 같은 시간에
액션 전문가의 기울기 표본이 N 배가 된다 (실측 아님, 계산: N=8 에서 +0.2%).

**쪼개도 값이 수학적으로 같다.** 프리픽스의 `ar_mask` 가 전부 False 라
(`pi0.py:126`, `pi0.py:132`) 프리픽스 토큰은 서픽스를 attend 하지 않는다.
따라서 프리픽스 출력은 서픽스와 무관하고, 한 번 계산해 재사용해도 된다.
`sample_actions` 가 이미 같은 경로를 쓴다 (`pi0.py:232-265`).

근거: 2025 우승팀이 `num_flow_samples=15` 를 쓴다 (`docs/winner-config-2025.md` 4절).

## 2. 차원별 손실 로깅

액션 23차원 중 **그리퍼 2개가 학습 신호(정규화된 E[a²])의 51%** 를 차지한다
(docs/HANDOFF.md 15.3). 이게 실제 문제인지 — 즉 학습 뒤에도 잔차가 거기 남는지 —
는 **차원별 손실을 따로 재야만** 알 수 있다.

⚠️ **openpi 의 `train_step` 을 복제하지 않는다.** 그쪽 `info` 딕셔너리에 키를
더하려면 `train_step` 을 통째로 베껴야 하는데, 그러면 EMA 갱신 같은 것을 조용히
빠뜨릴 위험이 생긴다. 대신 `jax.debug.callback` 으로 값만 호스트로 흘려보낸다.
**학습 계산 자체는 한 글자도 안 바뀐다.** 콜백이 터져도 학습이 죽지 않도록
호스트 쪽 전체를 try/except 로 감싼다.
"""
import threading

import einops
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.shared import array_typing as at

_applied = False


class _DimLogger:
    """차원별 / 청크위치별 손실을 호스트에서 모았다가 주기적으로 찍는다."""

    def __init__(self, interval: int, action_dim: int = 23):
        self.interval = interval
        self.action_dim = action_dim
        self.lock = threading.Lock()
        self.n = 0
        self.step = 0
        self.dim_sum = None
        self.pos_sum = None

    def __call__(self, per_dim, per_pos):
        try:
            with self.lock:
                import numpy as np
                d = np.asarray(per_dim, dtype=np.float64)[: self.action_dim]
                p = np.asarray(per_pos, dtype=np.float64)
                self.dim_sum = d if self.dim_sum is None else self.dim_sum + d
                self.pos_sum = p if self.pos_sum is None else self.pos_sum + p
                self.n += 1
                if self.n < self.interval:
                    return
                d, p = self.dim_sum / self.n, self.pos_sum / self.n
                self.step += self.n
                self.dim_sum = self.pos_sum = None
                self.n = 0
            h = len(p)
            names = (["base_vx", "base_vy", "base_wz"] + [f"trunk{i}" for i in range(4)]
                     + [f"L_arm{i}" for i in range(7)] + ["L_grip"]
                     + [f"R_arm{i}" for i in range(7)] + ["R_grip"])
            arm = [i for i in range(3, 22) if i != 14]          # 팔·토르소 (그리퍼 제외)
            log = {f"dim/{names[i]}": float(d[i]) for i in range(min(len(names), len(d)))}
            log["group/arm_trunk"] = float(d[arm].mean())
            log["group/gripper"] = float((d[14] + d[22]) / 2)
            log["group/base"] = float(d[:3].mean())
            log["chunk/first8"] = float(p[: h // 4].mean())
            log["chunk/last8"] = float(p[-h // 4 :].mean())
            # VRAM 실사용량. 리소스 창의 "71.8/80.0 GB" 는 JAX 가 **시작할 때 미리
            # 잡아 둔 양**이라 실사용이 아니다(HANDOFF 7.1/15.7). `memory_stats()`
            # 는 프리얼로케이션과 무관하게 진짜 쓰는 양을 준다. 이 숫자를 남겨 둬야
            # "더 작은(싼) GPU 로 되나" 를 다음에 계산으로 답할 수 있다.
            try:
                import jax as _jax
                ms = _jax.local_devices()[0].memory_stats() or {}
                peak = ms.get("peak_bytes_in_use")
                mem = f"  VRAM 최대 {peak / 2**30:.1f}GiB" if peak else ""
                if peak:
                    log["mem/peak_gib"] = peak / 2**30
            except Exception:
                mem = ""
            print(f"[dim] step~{self.step}  팔·토르소 {log['group/arm_trunk']:.5f}  "
                  f"그리퍼 {log['group/gripper']:.5f}  베이스 {log['group/base']:.5f}  "
                  f"청크앞8 {log['chunk/first8']:.5f}  청크뒤8 {log['chunk/last8']:.5f}"
                  f"{mem}", flush=True)
            try:
                import wandb
                if wandb.run is not None:
                    # ⚠️ `step=` 을 주면 안 된다. JAX 는 비동기라 콜백이 파이썬
                    # 루프보다 뒤처지는데, openpi 쪽이 이미 더 큰 step 으로 로그했으면
                    # wandb 가 경고만 찍고 **버린다**. print 는 남아서 학습은 안 죽고
                    # wandb 패널만 조용히 빈다. 생략하면 "현재 step" 에 붙는다.
                    wandb.log(log, commit=False)
            except Exception:
                pass
        except Exception as e:      # 콜백이 학습을 죽이면 안 된다
            print(f"[dim] 로깅 실패(무시): {e}", flush=True)


def _make_compute_loss(num_flow_samples: int, logger: "_DimLogger | None"):
    def compute_loss(self, rng, observation, actions, *, train: bool = False):
        preprocess_rng, flow_rng = jax.random.split(rng, 2)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_shape = actions.shape[:-2]

        # --- 프리픽스는 딱 한 번만 돈다 (연산의 99.5%) ---
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions
        )

        def one_sample(sample_rng):
            noise_rng, time_rng = jax.random.split(sample_rng, 2)
            noise = jax.random.normal(noise_rng, actions.shape)
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
            time_expanded = time[..., None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, time
            )
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            pre = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([pre, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return jnp.square(v_t - u_t)                       # [b, ah, ad]

        if num_flow_samples == 1:
            sq = one_sample(flow_rng)
        else:
            rngs = jax.random.split(flow_rng, num_flow_samples)
            with at.disable_typechecking():
                sq = jnp.mean(jax.vmap(one_sample)(rngs), axis=0)   # [N,b,ah,ad] -> [b,ah,ad]

        if logger is not None:
            # 값만 호스트로 흘린다. 학습 계산에는 영향이 없다.
            jax.debug.callback(logger, jnp.mean(sq, axis=(0, 1)), jnp.mean(sq, axis=(0, 2)))

        return jnp.mean(sq, axis=-1)                           # [b, ah] — openpi 규약 그대로

    return compute_loss


def apply(num_flow_samples: int = 1, log_per_dim: bool = False, log_interval: int = 100) -> None:
    """`Pi0.compute_loss` 를 갈아 끼운다. 한 번만 적용된다."""
    global _applied
    if _applied:
        return
    if num_flow_samples == 1 and not log_per_dim:
        return                                                  # 바꿀 게 없으면 손대지 않는다
    logger = _DimLogger(log_interval) if log_per_dim else None
    _pi0.Pi0.compute_loss = _make_compute_loss(num_flow_samples, logger)
    _applied = True
    print(f"[pi0_patch] num_flow_samples={num_flow_samples}, 차원별 로깅={log_per_dim}", flush=True)
