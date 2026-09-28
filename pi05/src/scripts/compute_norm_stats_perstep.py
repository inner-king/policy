"""스텝별(청크 위치별) 정규화 통계 — v3 학습용. 로컬에서 무료로 돈다.

    HF_LEROBOT_HOME=$PWD/data/lerobot \
    third_party/openpi/.venv/bin/python scripts/compute_norm_stats_perstep.py \
        --config-name pi05_cooking_top20_v3 --max-frames 400000 \
        --batch-size 256 --num-workers 6

**저장소에 커밋된 통계는 위 명령 그대로(40만 프레임 = 청크 399,872개)** 만든
것이다. 통계 파일 자체에는 표본 수가 안 남으므로 여기에 적어 둔다. 셀 하나마다
40만 표본이 들어가므로 1% 꼬리에 4,000개다 — 100만까지 갈 필요가 없었다.

## 왜 필요한가

델타 액션은 `action[t] - state(지금)` 이라 **청크 뒤로 갈수록 커진다.** 그런데
지금 통계는 청크를 통합해 하나로 낸다(openpi 의 `RunningStats.update` 가
`batch.reshape(-1, last)` 로 청크 위치를 반드시 뭉갠다 — `normalize.py:35`).
그 결과 팔 관절의 정규화 후 표준편차가 t=0 에서 0.061, t=31 에서 0.434 로
**7.1배** 차이가 나고, 분산으로는 **19배**다.

손실은 정규화된 값의 제곱오차라 앞쪽 청크가 학습 신호를 거의 못 받는다.
그런데 receding horizon 은 32개 중 **앞 16개만 실행한다**(`rollout_b1k.py:49`).
실행하는 쪽을 덜 가르치고 있었다. 근거와 수치는 docs/HANDOFF.md 15.2.

## 왜 손실 가중치가 아니라 정규화인가

한때 `w[t] = (통합std/std_t)²` 손실 가중치를 검토했으나 실측으로 폐기했다.
위치와 무관한 5개 차원(베이스 3 + 그리퍼 2)이 정규화 후 분산의 58.9% 를
차지해 비율을 희석하는 탓에 실제 가중치가 0.68~1.53 — 항등변환에 가깝다.

정규화라야 하는 진짜 이유는 **모델 입력까지 고쳐진다**는 점이다.
`x_t = t·ε + (1−t)·â` 인데(`pi0.py:199`) 지금은 앞쪽 청크의 â 가 표준편차
0.06 이라 입력이 사실상 순수 잡음이다. 손실 가중치로는 이걸 못 고친다.
또 위치마다 표준편차를 1 로 맞추면 잡음 바닥이 모든 위치에서 같게 유지되어
기울기 잡음이 커지지 않는다. 근거는 docs/HANDOFF.md 15.4.

## openpi 를 고치지 않는다

우승팀은 `NormStats` 에 `per_timestamp_q01/q99` 필드를 새로 달았지만, 우리
openpi(커밋 215abfb)는 2차원 통계를 그대로 받는다 — 실측으로 확인했다:

    정규화   `transforms.py:141-145`  stats.q01[..., :x.shape[-1]] 가 (32,23) 을
                                     (32,23) 액션에 그대로 브로드캐스트
    역정규화 `transforms.py:176-179`  잘라낸 x[..., :23] 이 (32,23) 이고 통계도
                                     (32,23) 이라 **모양이 정확히 일치**한다.
                                     맞는 이유가 "분기를 타서"가 아니라는 점이
                                     중요하다 — 변환 순서가 바뀌어 x 가 23-D 로
                                     들어와도(그때는 분기를 안 탄다) 역시 맞는다
    직렬화   numpydantic NDArray 가 (32,23) 을 왕복 보존

따라서 **통계 파일만 2차원으로 만들면 된다.**

## 어떤 차원을 스텝별로 하나

청크 위치에 따라 커지는 것은 **델타로 바꾸는 차원뿐**이다
(`r1pro_policy.DELTA_DIMS` = 토르소 4 + 양팔 14 = 18개). 베이스 속도 3개와
그리퍼 2개는 원래 델타 성격이라 위치와 무관하다. 이들에 스텝별 통계를 쓰면
효과는 없고 표본 잡음만 들어오므로 **통합값을 그대로 쓴다.**

⚠️ **액션 6번(토르소 4번째 관절)은 델타 차원이지만 제외한다.** q99−q01 이
0.00182 rad(0.1도)라 사실상 고정된 축인데(HANDOFF 12.6), 지금도 정규화 후
표준편차 0.388 로 학습 신호의 4.1% 를 먹고 있다. 스텝별 정규화를 하면 모든
위치에서 1.0 으로 부풀어 **순수 잡음이 차원 하나 몫을 계속 차지한다.**
통합값으로 두면 최소한 지금보다 나빠지지는 않는다.
(4.1% 를 아예 없애려면 손실 마스크가 필요한데, 이번에 바꾸는 것이 이미
다섯이라 넣지 않는다.)
"""
import argparse
import pathlib
import sys

import numpy as np
import tqdm

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))

import lerobot_patch
lerobot_patch.apply()

import task_subset
task_subset.apply()

import cooking_config
import r1pro_policy

from openpi.shared import normalize as _normalize
from openpi.training import config as _config

# 스텝별로 통계를 낼 차원 = 델타 차원에서 잡음 축(6번)을 뺀 것.
PER_STEP_DIMS = tuple(d for d in r1pro_policy.DELTA_DIMS if d != 6)
POOLED_DIMS = tuple(d for d in range(r1pro_policy.ACTION_DIM) if d not in PER_STEP_DIMS)


class KeepNumeric:
    """문자열(프롬프트)을 떨어뜨린다. JAX 로 못 넘기고 통계에도 필요 없다.

    openpi 의 `compute_norm_stats.py` 에 같은 일을 하는 `RemoveStrings` 가 있지만
    **가져다 쓰면 안 된다.** 그 파일은 스크립트라 정상 import 경로가 없어서
    동적 로드를 해야 하는데, openpi 의 DataLoader 는 **spawn** 워커라 변환 객체를
    피클해 보낸다. 자식 프로세스에는 그 동적 모듈이 없어서 이렇게 죽는다:

        ModuleNotFoundError: No module named '_openpi_norm_stats'

    그래서 같은 일을 하는 것을 여기(정상 import 되는 모듈)에 둔다.
    """

    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items()
                if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _check_openpi_intact() -> None:
    """openpi 원본이 우리 래퍼로 덮여 있지 않은지 본다 (실제로 그런 적이 있다)."""
    path = ROOT / "third_party/openpi/scripts/compute_norm_stats.py"
    if path.exists() and "cooking-vla" in path.read_text():
        sys.exit(
            f"[cooking-vla] openpi 원본이 우리 래퍼로 덮여 있다: {path}\n"
            "  `git -C third_party/openpi checkout -- scripts/compute_norm_stats.py` 로 되돌릴 것.\n"
            "  (그대로 두면 scripts/compute_norm_stats.py 가 자기 자신을 무한 재귀 호출한다.)"
        )


def build_loader(dc, cfg, batch_size: int, num_workers: int, max_frames: int):
    """학습과 **같은** 경로로 데이터를 뽑는다 (repack + data_transforms 까지)."""
    from openpi.training import data_loader as _dl

    ds = _dl.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    ds = _dl.TransformedDataset(
        ds, [*dc.repack_transforms.inputs, *dc.data_transforms.inputs, KeepNumeric()]
    )
    if max_frames is not None and max_frames < len(ds):
        num_batches, shuffle = max_frames // batch_size, True
    else:
        num_batches, shuffle = len(ds) // batch_size, False
    loader = _dl.TorchDataLoader(
        ds, local_batch_size=batch_size, num_workers=num_workers,
        shuffle=shuffle, num_batches=num_batches,
    )
    return loader, num_batches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--max-frames", type=int, default=400_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    a = ap.parse_args()

    cooking_config.register()
    cfg = _config.get_config(a.config_name)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    H = cfg.model.action_horizon
    _check_openpi_intact()

    loader, num_batches = build_loader(
        data_config, cfg, a.batch_size, a.num_workers, a.max_frames
    )

    # 청크 위치마다 하나씩. openpi 의 검증된 히스토그램 분위수를 그대로 재사용한다.
    per_step = [_normalize.RunningStats() for _ in range(H)]
    pooled = _normalize.RunningStats()      # 청크를 뭉갠 통계 (지금까지 쓰던 것과 같다)
    state_stats = _normalize.RunningStats()
    seen = 0

    for batch in tqdm.tqdm(loader, total=num_batches, desc="스텝별 통계"):
        act = np.asarray(batch["actions"], dtype=np.float32)      # (b, H, 23)
        assert act.ndim == 3 and act.shape[1] == H, act.shape
        state_stats.update(np.asarray(batch["state"], dtype=np.float32))
        pooled.update(act)                                        # 내부에서 (-1, 23) 로 편다
        for t in range(H):
            per_step[t].update(act[:, t, :])
        seen += act.shape[0]

    ps = _normalize.NormStats(**{k: np.stack([getattr(s.get_statistics(), k) for s in per_step])
                                 for k in ("mean", "std", "q01", "q99")})   # 각각 (H, 23)
    pl = pooled.get_statistics()

    # 위치와 무관한 차원 + 잡음 축은 통합값으로 되돌린다 (위 주석 참조).
    for k in ("mean", "std", "q01", "q99"):
        arr = getattr(ps, k)
        arr[:, list(POOLED_DIMS)] = getattr(pl, k)[list(POOLED_DIMS)]

    out = pathlib.Path(cooking_config.ASSETS_DIR) / data_config.asset_id
    _normalize.save(out, {"state": state_stats.get_statistics(), "actions": ps})

    # --- 보고 ---
    w = ps.q99 - ps.q01
    print(f"\n표본 {seen:,} 청크 / horizon {H} / 저장 {out}/norm_stats.json")
    print(f"스텝별 차원 {len(PER_STEP_DIMS)}개 {PER_STEP_DIMS}")
    print(f"통합값 차원 {len(POOLED_DIMS)}개 {POOLED_DIMS}  (베이스 3 + 잡음축 6 + 그리퍼 2)")
    print(f"\n{'청크 위치':>10}{'팔 폭(q99-q01) 평균':>22}{'t=31 대비':>12}")
    arm = [d for d in PER_STEP_DIMS if d >= 7]
    for t in [0, 1, 4, 8, 15, 23, 31]:
        print(f"{t:>10}{w[t, arm].mean():>22.5f}{w[t, arm].mean()/w[31, arm].mean():>12.3f}")
    print(f"\n통합 통계로 쟀을 때 팔 폭 {(pl.q99 - pl.q01)[arm].mean():.5f} (모든 위치 동일)")
    print("→ 스텝별로 하면 앞쪽 청크의 목표가 그만큼 확대되어 학습 신호가 균등해진다")
    bad = [(t, d) for t in range(H) for d in range(23) if w[t, d] <= 1e-6]
    print(f"\n폭이 0 인 (위치,차원) 칸: {len(bad)}개" + (f" 🔴 {bad[:5]}" if bad else " (없음)"))


if __name__ == "__main__":
    main()
