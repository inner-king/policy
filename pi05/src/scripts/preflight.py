"""학습 직전 점검 — CU 를 태우기 전에 **Colab 과 같은 경로**로 배치를 뽑아 본다.

    HF_LEROBOT_HOME=$PWD/data/lerobot HF_HUB_OFFLINE=1 \
    third_party/openpi/.venv/bin/python scripts/preflight.py pi05_cooking_top20_v3

설정을 읽어 보는 것만으로는 부족하다. 2차 학습 직전 점검에서 **DataLoader
경로에서만 터지는 버그 두 개**를 잡았다(HANDOFF 12.11) — 단품 검사(`ds[int(i)]`)
는 통과하는데 spawn 워커를 거치면 죽는 종류였다. 그래서 실제 로더를 쓴다.

여기서 보는 것 중 **조용히 틀리는** 종류를 특히 조심할 것:

- 정규화 통계를 못 찾으면 openpi 는 경고만 찍고 **정규화 없이** 학습한다
  ("Norm stats not found ... skipping"). 에러가 아니라서 다 태우고 나서야 안다.
- 부분집합 필터가 안 걸리면 20종이 아니라 466종 전체로 학습한다.
- freeze_filter 가 어긋나면 엉뚱한 것을 학습하거나 아무것도 학습하지 않는다.
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))

import lerobot_patch
lerobot_patch.apply()

import task_subset
task_subset.apply()

import cooking_config
import r1pro_policy


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "pi05_cooking_top20_v3"
    cooking_config.register()

    from openpi.training import config as _config
    from openpi.training import data_loader as _dl

    cfg = _config.get_config(name)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    ok = True

    print(f"{'=' * 70}\n설정: {name}\n{'=' * 70}")
    # ⚠️ 아래는 **설정 파일의 기본값**이다. 본 학습 셀은 CLI 로 덮어쓰므로
    #    (--num-train-steps / --lr-schedule.peak-lr / --seed / --batch-size)
    #    이 숫자를 "이번 학습에 쓸 값" 으로 읽으면 안 된다. 여기서 보는 목적은
    #    데이터 경로·정규화 통계·얼리는 범위가 성한지이지 하이퍼파라미터가 아니다.
    print("  [아래 5줄은 설정 기본값 — 본 학습 셀이 CLI 로 덮어쓴다]")
    print(f"  배치 {cfg.batch_size} / 스텝 {cfg.num_train_steps:,} / 워커 {cfg.num_workers}")
    print(f"  horizon {cfg.model.action_horizon} / max_token_len {cfg.model.max_token_len}")
    print(f"  ema_decay {cfg.ema_decay} / lr peak {cfg.lr_schedule.peak_lr} "
          f"decay_steps {cfg.lr_schedule.decay_steps:,}")
    print(f"  asset_id {dc.asset_id}")
    print(f"  weight_loader {getattr(cfg.weight_loader, 'params_path', '?')}")

    # --- 1. 정규화 통계 ---
    print(f"\n── 1. 정규화 통계 ──")
    if dc.norm_stats is None:
        print("  🔴 통계를 못 찾았다. 이대로 학습하면 **정규화 없이** 돈다(경고만 찍힌다).")
        print(f"     scripts/compute_norm_stats_perstep.py --config-name {name} 을 먼저 돌릴 것.")
        return 1
    for k, st in dc.norm_stats.items():
        print(f"  {k:8s} q01 {np.asarray(st.q01).shape}  q99 {np.asarray(st.q99).shape}")
    aq = np.asarray(dc.norm_stats["actions"].q01)
    is_v3 = cooking_config.is_v3(name)
    if is_v3 and aq.ndim != 2:
        print(f"  🔴 v3 인데 액션 통계가 {aq.ndim}차원이다. 스텝별 정규화가 안 걸린다.")
        ok = False
    elif is_v3:
        w = np.asarray(dc.norm_stats["actions"].q99) - aq
        arm = [d for d in r1pro_policy.DELTA_DIMS if d >= 7]
        # 이 값은 **데이터의 폭**이라 앞뒤가 달라야 정상이다 (델타가 뒤로 갈수록 커진다).
        # 정규화가 이걸 흡수하는지는 아래 2번의 '정규화 후 std' 로 판정한다.
        print(f"  위치별 데이터 폭(팔): t=0 {w[0, arm].mean():.5f} / t=31 {w[-1, arm].mean():.5f}"
              f"  (비 {w[0, arm].mean() / w[-1, arm].mean():.3f} — 데이터가 원래 이만큼 다르다)")

    # --- 2. 실제 DataLoader (spawn 워커 포함) ---
    print(f"\n── 2. 실제 학습 경로로 배치 뽑기 ──")
    loader = _dl.create_data_loader(cfg, num_batches=1, shuffle=True)
    obs, act = next(iter(loader))
    for k, v in obs.images.items():
        print(f"  image[{k}]  {tuple(v.shape)}  {v.dtype}")
    print(f"  state              {tuple(obs.state.shape)}  {obs.state.dtype}")
    print(f"  tokenized_prompt   {tuple(obs.tokenized_prompt.shape)}")
    used = np.asarray(obs.tokenized_prompt_mask).sum(-1)
    print(f"    실제 토큰 {used.min()}~{used.max()}  (설정 {cfg.model.max_token_len})")
    if used.max() >= cfg.model.max_token_len:
        print("  🔴 프롬프트가 잘리고 있다 — max_token_len 을 늘릴 것 (에러 없이 학습이 망가진다)")
        ok = False
    a = np.asarray(act, dtype=np.float64)
    print(f"  actions            {tuple(a.shape)}")
    print(f"    정규화 후 std {a.std():.3f} / |x|>1 {100*(np.abs(a) > 1).mean():.1f}% "
          f"/ 최대 {np.abs(a).max():.1f} / NaN {int(np.isnan(a).sum())}")
    if np.isnan(a).any():
        print("  🔴 NaN 이 있다"); ok = False

    # 스텝별 정규화가 실제로 먹었는지: 앞뒤 청크의 표준편차가 비슷해야 한다
    if is_v3 and a.shape[1] >= 16:
        arm = [d for d in r1pro_policy.DELTA_DIMS if d >= 7]
        f, b = a[:, :8, arm].std(), a[:, -8:, arm].std()
        print(f"    팔 차원 정규화 후 std: 청크앞8 {f:.3f} / 뒤8 {b:.3f}  (비 {f/b:.2f}"
              f" — 1 에 가까워야 앞뒤가 균등하게 배운다. 통합 정규화면 약 0.35)")
        if f / b < 0.5:
            print("  🔴 앞쪽 청크가 여전히 눌려 있다 — 스텝별 통계가 안 먹었다"); ok = False

    print(f"\n{'=' * 70}\n{'통과 — 학습을 시작해도 된다' if ok else '🔴 실패 — 위 항목을 고칠 것'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
