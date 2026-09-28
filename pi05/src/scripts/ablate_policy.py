"""정책이 카메라와 지시문을 **실제로 보고 있는지** 절제(ablation)로 판정한다.

    python scripts/ablate_policy.py [샘플수]

## 왜 필요한가

학습 손실이 낮아도, 오프라인 오차가 '가만히 있기'와 비슷하면 두 가지 가능성이 있다.

  (A) 모델이 관측을 무시하고 평균 동작만 낸다  -> 심각. 다시 학습해야 한다
  (B) 모델은 관측을 쓰는데, 예측이 확률적이라 한 번 뽑은 표본의 오차가 클 뿐이다
      -> 추론 방식(표본 여러 개 평균 등)으로 개선 가능

손실 비교로는 둘을 못 가른다. 확실한 방법은 **관측을 망가뜨려 보는 것**이다.
카메라를 엉뚱한 것으로 바꿔도 오차가 그대로면 모델은 카메라를 안 보고 있다.

세 가지를 잰다.
  1. 정상        — 있는 그대로
  2. 카메라 교란 — 다른 샘플의 이미지 3장으로 교체 (state/지시문은 그대로)
  3. 지시문 교란 — 다른 샘플의 지시문으로 교체 (이미지/state 는 그대로)
  4. 표본 평균   — 같은 관측으로 K 번 뽑아 평균 (확률성 영향 측정)
"""
import os, pathlib, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / "data/lerobot"))
os.environ["HF_HUB_OFFLINE"] = "1"
# 8GB 카드는 가중치 6.81GB 를 올리는 것만으로도 한계라, 반복 추론 중에 죽는다.
# 이 검사는 GPU 를 고집할 이유가 없으므로 기본을 CPU 로 둔다(10.5초/회).
if (sys.argv[3] if len(sys.argv) > 3 else "cpu") == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # 위 주석이 말하는 "둘 다" 의 나머지 절반. 이게 빠져 있어서 BFC 할당기가
    # 그대로 쓰였고, 2026-08-29 에 정확히 위에 적힌 그 오류로 죽었다.
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
sys.path.insert(0, str(ROOT / "configs"))

import lerobot_patch, task_subset, cooking_config, r1pro_policy, policy_loader

# 어느 설정/체크포인트를 볼지는 환경변수로 바꾼다. 기본은 2026-08-27 에 학습한
# 1차 체크포인트(지시문 466종). 2차 학습 결과를 보려면:
#   COOKING_CONFIG=pi05_cooking_top20 \
#   COOKING_CKPT=checkpoints/pi05_cooking_top20/run2/<step> python scripts/<이 파일>
CONFIG_NAME = os.environ.get("COOKING_CONFIG", "pi05_cooking")
CKPT = ROOT / os.environ.get("COOKING_CKPT", "checkpoints/pi05_cooking/run1/9999")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
# 표본 평균에 쓸 반복 횟수. 0 이면 건너뛴다(카메라/지시문 판정만 급할 때).
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
DELTA = r1pro_policy.DELTA_DIMS


def main():
    lerobot_patch.apply(); task_subset.apply(); cooking_config.register()
    import jax
    from openpi.training import config as _config, data_loader as _dl
    from openpi.policies import policy_config as _pc

    cfg = _config.get_config(CONFIG_NAME)
    policy, dc = policy_loader.load_policy(cfg, CKPT)
    ds = _dl.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    # 이 데이터셋은 변환 전이라 state 가 원본 61차원이다. 대응표를 그에 맞춰 고른다.
    A2S = r1pro_policy.action_to_state_for(len(ds[0]["state"]))
    print(f"{jax.devices()}  샘플 {N}개")

    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(ds), N)
    other = rng.integers(0, len(ds), N)          # 교란에 쓸 다른 샘플

    def obs_of(it):
        return {"observation/image": np.asarray(it["image"]),
                "observation/left_wrist_image": np.asarray(it["left_wrist_image"]),
                "observation/right_wrist_image": np.asarray(it["right_wrist_image"]),
                "observation/state": np.asarray(it["state"], dtype=np.float32),
                "prompt": it["prompt"]}

    def delta(pred, base):
        return np.asarray(pred, dtype=np.float64)[:, DELTA] - base

    res = {k: [] for k in ("정상", "카메라 교란", "지시문 교란", f"표본{K}개 평균", "가만히")}
    t0 = time.time()
    for n, (i, j) in enumerate(zip(idx, other), 1):
        it, ot = ds[int(i)], ds[int(j)]
        st = np.asarray(it["state"], dtype=np.float64)
        base = np.array([st[A2S[d]] for d in DELTA])
        d_true = np.asarray(it["actions"], dtype=np.float64)[:, DELTA] - base

        o = obs_of(it)
        res["정상"].append(np.mean((delta(policy.infer(dict(o))["actions"], base) - d_true) ** 2))

        oc = dict(o)                                  # 카메라만 교체
        for k in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image"):
            oc[k] = np.asarray(ot[k.split("/")[1]])
        res["카메라 교란"].append(np.mean((delta(policy.infer(dict(oc))["actions"], base) - d_true) ** 2))

        op = dict(o); op["prompt"] = ot["prompt"]     # 지시문만 교체
        res["지시문 교란"].append(np.mean((delta(policy.infer(dict(op))["actions"], base) - d_true) ** 2))

        if K > 0:
            avg = np.mean([delta(policy.infer(dict(o))["actions"], base) for _ in range(K)], axis=0)
            res[f"표본{K}개 평균"].append(np.mean((avg - d_true) ** 2))

        res["가만히"].append(np.mean(d_true ** 2))
        if n % 25 == 0:
            print(f"  {n}/{N}  ({time.time()-t0:.0f}초)", flush=True)

    still = np.array(res["가만히"]).sum()
    print(f"\n{'조건':<16}{'오차':>10}{'가만히 대비':>12}{'승률':>8}")
    for k in ("정상", f"표본{K}개 평균", "카메라 교란", "지시문 교란"):
        if not res.get(k):
            continue
        v = np.array(res[k]); s = np.array(res["가만히"])
        print(f"{k:<16}{v.mean():>10.5f}{v.sum()/still:>12.3f}{(v<s).mean()*100:>7.0f}%")
    print(f"{'가만히':<16}{np.array(res['가만히']).mean():>10.5f}{1.0:>12.3f}")

    norm = np.array(res["정상"]).sum()
    cam = np.array(res["카메라 교란"]).sum() / norm
    pro = np.array(res["지시문 교란"]).sum() / norm
    print(f"\n카메라를 망가뜨리면 오차가 {cam:.2f}배")
    print(f"지시문을 망가뜨리면 오차가 {pro:.2f}배")
    print("\n판정:")
    print("  카메라:", "✅ 실제로 보고 있다" if cam > 1.15 else "❌ 거의 안 보고 있다")
    print("  지시문:", "✅ 실제로 보고 있다" if pro > 1.15 else "❌ 거의 안 보고 있다")


if __name__ == "__main__":
    main()
