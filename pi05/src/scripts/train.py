"""openpi 학습 실행 래퍼 — cooking-vla 설정을 등록한 뒤 openpi의 train.py 로 넘긴다.

    uv run --project third_party/openpi python scripts/train.py pi05_cooking_smoke --exp-name=smoke1

openpi를 전혀 수정하지 않기 위한 얇은 층이다 (이유는 configs/cooking_config.py 참조).
"""
import pathlib, runpy, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))

import lerobot_patch
lerobot_patch.apply()          # 6.7만 에피소드에서 데이터셋 생성 321초/25GB -> 188초/10.8GB

import task_subset
task_subset.apply()            # asset_id 에 필터가 등록된 설정이면 인덱스를 걸러 준다

import cooking_config
cooking_config.register()

import pi0_patch


def _check_openpi_intact() -> None:
    """openpi 원본이 우리 래퍼로 덮여 있지 않은지 본다.

    실제로 `third_party/openpi/scripts/compute_norm_stats.py` 가 우리 래퍼 내용으로
    덮여 있던 적이 있다(HANDOFF 15.15). 우리 래퍼는 그 파일을 `runpy` 로 실행하므로
    **자기 자신을 무한 재귀 호출**한다. `third_party` 는 gitignore 대상이라 우리
    저장소의 git 이 이걸 못 잡는다. 여기는 CU 를 태우는 입구라 값어치가 크다.
    """
    for name in ("train.py", "compute_norm_stats.py"):
        f = ROOT / "third_party/openpi/scripts" / name
        if f.exists() and "cooking-vla" in f.read_text():
            sys.exit(
                f"\n[cooking-vla] openpi 원본이 우리 래퍼로 덮여 있다: {f}\n"
                f"  git -C third_party/openpi checkout -- scripts/{name}\n"
                "  로 되돌린 뒤 다시 실행할 것.\n")


_check_openpi_intact()


_DEFAULT_SEED = 42     # openpi TrainConfig.seed 의 기본값


def _apply_model_patches(cfg) -> None:
    """v3 설정이면 손실 계산에 두 가지를 붙인다 (openpi 는 안 건드린다).

    `_config.cli()` 안에서 부르는 이유는, 여기서만 **실제로 확정된 설정 객체**를
    손에 쥘 수 있기 때문이다. 모델은 이보다 나중에(`init_train_state`) 만들어지므로
    `Pi0.compute_loss` 를 갈아 끼우기에 시점이 충분히 이르다.

    스모크에도 똑같이 적용한다 — 손실 경로가 달라지면 재는 초/스텝이 본 학습과
    맞지 않아 스모크의 뜻이 없어진다.
    """
    if not cooking_config.is_v3(cfg.name):
        return
    pi0_patch.apply(
        num_flow_samples=cooking_config.NUM_FLOW_SAMPLES_V3,
        log_per_dim=True,
        log_interval=cfg.log_interval,
    )


def _guard_seed(cfg) -> None:
    """이어서 학습할 때 `--seed` 를 안 바꾸면 **같은 데이터만 다시 본다**.

    두 가지가 겹쳐서 생기는 문제다.

    1. 재개할 때 데이터 로더 상태를 복원하지 않는다. openpi 의
       `checkpoints.restore_state()` 첫 줄이 `del data_loader` 다.
    2. 셔플 시드가 설정값으로 고정돼 있다 (`TrainConfig.seed = 42`,
       `data_loader.py` 가 `generator.manual_seed(seed)`).

    그래서 **프로세스를 새로 띄울 때마다 데이터가 늘 같은 순서로 처음부터**
    시작한다. 한 에폭을 다 돌면 문제가 안 되는데, 우리는 상위 20종 194만
    프레임을 배치 32 로 도니까 1 에폭이 60,732 스텝이다. 12,500 스텝이면
    0.21 에폭, 즉 앞쪽 40만 샘플만 본다.

    이어서 학습하면서 시드를 그대로 두면 **그 40만 샘플을 또 보고**, 나머지
    154만 프레임은 영영 못 본다. 에러도 경고도 없다. 손실은 오히려 더 잘
    떨어져 보인다 — 이미 본 데이터니까. 그래서 여기서 막는다.

    시드를 바꾸면 순열이 다시 섞여 새 40만 샘플의 약 79% 가 처음 보는
    프레임이 된다. `config.seed` 는 데이터 셔플과 잡음 샘플링에만 쓰이고
    (`train.py` 의 `jax.random.key(config.seed)`), 둘 다 다음 학습에서는
    달라지기를 바라는 것이라 바꿔도 안전하다.
    """
    init = str(getattr(cfg.weight_loader, "params_path", "") or "")
    # 같은 설정의 **이전 run 에서 이어받을 때만** 문제가 된다. 사전학습
    # (pi05_base) 이나 다른 설정에서 넘어오는 것은 데이터를 처음 보는 것이라
    # 시드가 기본값이어도 맞다.
    # ⚠️ 2026-09-13: 이름만 비교하면 **v3 -> v4 처럼 설정 이름은 달라도 같은 데이터·
    #    같은 셔플 경로를 이어 가는 경우**에 이 방어벽이 조용히 꺼진다. v4 는
    #    `.../pi05_cooking_top20_v3/run7/...` 에서 이어받으므로 `cfg.name` 이 경로에
    #    없다. v3·v4 는 asset_id(데이터)가 같으므로 같은 계보로 본다.
    same_data = cfg.name in init or (
        cooking_config.is_v3(cfg.name)
        and any(f"pi05_cooking_top20_{v}/" in init for v in ("v3", "v4")))
    if not same_data:
        return
    if cfg.seed != _DEFAULT_SEED:
        return
    sys.exit(
        f"\n[cooking-vla] 같은 설정({cfg.name})의 이전 체크포인트에서 이어받는데\n"
        f"시드가 기본값({_DEFAULT_SEED}) 그대로다.\n\n"
        f"  이어받는 곳: {init}\n\n"
        "openpi 는 재개할 때 데이터 위치를 복원하지 않고, 셔플 시드는 고정이다.\n"
        "이대로 두면 **지난 학습이 본 것과 똑같은 샘플을 처음부터 다시** 본다.\n"
        "우리는 1 에폭의 20% 밖에 안 도니까, 나머지 80% 는 영영 안 쓰인다.\n"
        "에러도 경고도 안 나고 손실은 오히려 잘 떨어져 보인다(이미 본 데이터라서).\n\n"
        f"다음 인자를 추가할 것 (42 가 아닌 아무 값):\n"
        f"  --seed={_DEFAULT_SEED + 1}\n"
    )


def _guard_lr_schedule() -> None:
    """`--num-train-steps` 만 바꿨을 때 학습률이 안 식는 사고를 막는다.

    `lr_schedule.decay_steps` 는 설정을 만들 때 값이 박히고, tyro 는 그 둘을
    연결해 주지 않는다. 그래서 `--num-train-steps=11000` 만 주면 decay_steps 는
    30,000 으로 남고, 코사인 곡선의 앞 1/3 만 지나간 채 학습이 끝난다.
    실측: 8,000 스텝에서 끝낼 때 마지막 lr 이 5e-6 이 아니라 4.4e-5 였다.

    **에러가 나지 않는다.** 손실 곡선도 그럴듯하게 내려간다. 다만 마지막
    체크포인트가 학습률이 높은 상태에서 저장돼 성능이 눈에 띄게 나빠진다.
    그래서 여기서 아예 실행을 막고 붙여야 할 인자를 알려 준다.
    """
    from openpi.training import config as _config

    _orig_cli = _config.cli

    def cli():
        cfg = _orig_cli()
        # 스모크는 초/스텝만 재는 용도다. `--num-train-steps=60` 처럼 짧게
        # 잘라 재는 것이 정상 사용법이라 여기서 막으면 방해만 된다.
        _apply_model_patches(cfg)
        if "smoke" in cfg.name:
            return cfg
        _guard_seed(cfg)
        sched = cfg.lr_schedule
        decay = getattr(sched, "decay_steps", None)
        if decay is not None and decay != cfg.num_train_steps:
            warmup = min(1_000, max(1, cfg.num_train_steps // 10))
            sys.exit(
                "\n[cooking-vla] 학습률 스케줄이 학습 길이와 어긋난다.\n"
                f"  --num-train-steps = {cfg.num_train_steps:,}\n"
                f"  lr_schedule.decay_steps = {decay:,}   <- 이 값이 따라오지 않았다\n\n"
                "이대로 두면 학습이 끝나는 시점의 학습률이 최종값(5e-6)이 아니라\n"
                "거의 최고점(5e-5)이라, 가중치가 요동치는 상태로 저장된다.\n"
                "에러 없이 조용히 성능만 나빠지는 종류의 문제라 여기서 막는다.\n\n"
                "다음 두 인자를 추가할 것:\n"
                f"  --lr-schedule.decay-steps={cfg.num_train_steps}\n"
                f"  --lr-schedule.warmup-steps={warmup}\n"
            )
        return cfg

    _config.cli = cli


_guard_lr_schedule()

runpy.run_path(str(ROOT / "third_party/openpi/scripts/train.py"), run_name="__main__")
