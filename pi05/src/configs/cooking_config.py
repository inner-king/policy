"""cooking-vla 학습 설정 — openpi의 `_CONFIGS` 에 런타임으로 등록한다.

`third_party/openpi` 는 이 저장소에서 gitignore 대상(다시 clone하는 대상)이라
openpi의 `config.py` 를 직접 고치면 재현이 안 된다. 그래서 설정을 여기에 두고
`scripts/train.py` / `scripts/compute_norm_stats.py` 래퍼가 openpi를 실행하기
전에 `register()` 를 호출한다. openpi의 `cli()` 는 `_CONFIGS_DICT` 를 호출
시점에 읽으므로(`config.py:978-979`) 이 방식이면 openpi 쪽 수정이 전혀 없다.

    uv run --project third_party/openpi python scripts/compute_norm_stats.py pi05_cooking
    uv run --project third_party/openpi python scripts/train.py pi05_cooking --exp-name=run1
"""

import dataclasses
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import r1pro_policy
import task_subset

import flax.nnx as nnx

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders
from openpi.shared import nnx_utils as _nnx_utils

# LeRobotDataset 은 root 인자 없이 `HF_LEROBOT_HOME/{repo_id}` 를 본다
# (`data_loader.py:140-146`). 그래서 HF repo 이름과 로컬 디렉터리 이름을 맞춰 두면
# 로컬과 Colab이 같은 설정으로 돌아간다:
#
#   export HF_LEROBOT_HOME=~/cooking-vla/data/lerobot
#   ~/cooking-vla/data/lerobot/jiu121/behavior-cooking/{data,videos,meta}
#
# ⚠️ 빌드 스크립트는 아직 옛 이름(cooking-vla/behavior-cooking-224)으로 쓰고 있다.
# 변환이 끝나면 디렉터리를 옮겨야 한다 (meta 어디에도 repo id가 박혀 있지 않아
# 이름만 바꾸면 된다 — 확인함):
#
#   mkdir -p data/lerobot/jiu121
#   mv data/lerobot/cooking-vla/behavior-cooking-224 data/lerobot/jiu121/behavior-cooking
REPO_ID = "jiu121/behavior-cooking"

# 정규화 통계의 위치. openpi 는 기본적으로 `./assets/{설정 이름}` 을 보는데
# (`config.py:538-540`), 그러면 pi05_cooking 과 pi05_cooking_smoke 가 서로 다른
# 폴더를 보게 되어 스모크에서 통계를 못 찾는다:
#   "Norm stats not found in .../assets/pi05_cooking_smoke/..., skipping."
# 통계가 없어도 조용히 넘어가기 때문에(정규화가 no-op 이 된다) 알아채기 어렵다.
# 그래서 설정 이름과 무관하게 한 곳을 보도록 절대 경로로 고정한다.
ASSETS_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "assets" / "pi05_cooking")

# max_token_len: pi05 기본값 200으로는 프롬프트가 100% 잘린다. state 61개가
# 대부분 3자리 정수라 state만으로 약 200토큰을 쓰기 때문이다. 실측 최대 260,
# 지시문 472종 전부에 대한 하드 상한 274. 여유를 둬서 280.
# 근거는 docs/HANDOFF.md 3장.
MAX_TOKEN_LEN = 280        # 구 설정(state 61-D). 실측 최대 274.

# state 를 25-D 로 줄인 뒤의 값. 466종 지시문 전부에 대해 실측한 최대가 130,
# 가장 긴 지시문 + 3자리 숫자만인 최악이 130 이라 150 으로 잡는다.
# 모델은 설정된 길이만큼 **항상** 계산하므로 이걸 줄이면 그대로 속도가 된다:
# 프리픽스가 이미지 768 + 텍스트 280 = 1,048 에서 918 토큰으로 12% 줄어든다.
MAX_TOKEN_LEN_V2 = 150

# 액션 청크 길이. 공식 baseline(pi05_b1k)은 32 를 쓴다. 50 보다 예측이 쉽고
# 연산도 준다. 서빙은 32 를 예측해 16 만 실행하는 receding horizon 이 공식 방식.
ACTION_HORIZON_V2 = 32

# 재학습 대상 지시문 20종 (에피소드 수 상위). 문자열을 그대로 박아 둔다 —
# 나중에 데이터가 바뀌어도 무엇으로 학습했는지가 코드에 남는다.
#
# 이 20종에는 **화면이 거의 같고 지시문만 다른 쌍**이 들어 있다. 예:
#   "pick up the brussels sprouts from the cutting board"
#   "pick up the half brussels sprouts from the cutting board"
# 이런 쌍이 있어야 모델이 언어를 볼 이유가 생긴다. 5종이나 1종으로 줄이면
# 카메라만 봐도 다 구분되어 언어 조건화를 배울 압력이 사라진다.
TOP20 = (
    "open the door of the fridge",
    "close the door of the fridge",
    "put the half brussels sprouts on the baking sheet",
    "pick up the carving knife from the countertop",
    "pick up the half brussels sprouts from the cutting board",
    "put the bratwurst in the hinged jar",
    "open the door of the oven",
    "pick up the brussels sprouts from the cutting board",
    "chop the brussels sprouts with the carving knife",
    "chop the half beet with the parer",
    "chop the half bell pepper with the parer",
    "put the carving knife back on the countertop to the right of the cutting board",
    "pick up the cutting board from the countertop",
    "open the door of the microwave",
    "close the door of the microwave",
    "pick up the carving knife from the cutting board",
    "close the door of the oven",
    "turn on the oven",
    "pick up the other bratwurst from the cutting board",
    "turn on the microwave",
)

# 9차 학습에서 더하는 12종. 고른 기준은 네 가지이고, 전부 `scripts/pick_top32.py`
# 가 데이터로 검사한다 (근거는 docs/HANDOFF.md 40.3).
#
#   1. **자동 판정기가 붙는 형태여야 한다.** `rollout_b1k.py` 의 정규식 5종
#      (문 열기·닫기 / 켜기 / 집기 / 놓기 / 자르기) 중 하나에 걸려야 한다.
#      안 걸리면 롤아웃이 돌아도 성패를 못 재고 영상 판독만 남는다.
#   2. 🔴 **미학습 기준선 9종을 건드리면 안 된다.** `scripts/rollout_unseen.sh`
#      가 일반화를 재는 데 쓰는 문장들이다. 여기에 하나라도 넣으면 그 측정이
#      영영 사라진다 — 특히 `open/close the door of the electric refrigerator`
#      는 "배운 fridge 와 거의 같은 물체" 자리라 유혹이 크지만 넣으면 안 된다.
#   3. **프레임 10만 이상.** 상한(CAP_FRAMES_V3)이 10만이라 그 아래면 다른
#      지시문보다 적게 노출된다.
#   4. **언어를 읽어야 풀리는 짝을 만든다.** 화면만 봐도 구분되면 언어 조건화를
#      배울 압력이 생기지 않는다(TOP20 머리말과 같은 이유). 그래서 기존 20종과
#      짝을 이루도록 골랐다 — 괄호 안이 그 짝이다.
NEW12 = (
    # 출처만 다른 같은 물체 — "from the ..." 를 읽어야 어디로 갈지 정해진다
    "pick up the cutting board from the bar",               # (TOP20: from the countertop)
    "pick up the cutting board from the drop in sink",      # (위와 같은 물체, 세 번째 출처)
    "pick up the carving knife from the bar",               # (TOP20: countertop / cutting board)
    # 사슬 완성 — 열기·닫기·켜기는 배웠는데 "넣기"만 빠져 있었다
    "put the hotdog in the microwave",                      # (TOP20: open/close/turn on the microwave)
    "put the tray in the oven",                             # (TOP20: open/close/turn on the oven)
    # 자르기의 앞 단계 — 재료를 도마에 올려야 자를 수 있다
    "put the beet on the cutting board",                    # (TOP20: chop the half beet with the parer)
    "put the bell pepper on the cutting board",             # (TOP20: chop the half bell pepper with the parer)
    # 냉장고 **안**에서 집기 — 문 여닫기는 배웠지만 안에서 꺼내 본 적이 없다
    "pick up the plate from the fridge",
    "pick up the brisket from the fridge",                  # (위와 같은 형태, 다른 물체)
    # 되돌려 놓기 — 칼만 배웠고 도마는 못 배웠다
    "put the cutting board back on the countertop",         # (TOP20: put the carving knife back on the countertop)
    # 새 도구 — parer 로 자르는 건 2종 배웠는데 parer 를 집는 건 없었다
    "pick up the parer from the bar",
    # 'door of' 가 아니라 'lid of' — 같은 동사, 다른 부위
    "open the lid of the hinged jar",                       # (TOP20: put the bratwurst in the hinged jar)
)

TOP32 = TOP20 + NEW12

# 부분집합은 액션 분포가 달라 정규화 통계를 따로 내야 한다. asset_id 를 나눠
# 통계 경로와 필터를 한 몸으로 묶는다 (configs/task_subset.py 설명 참조).
ASSET_ID_V1 = REPO_ID
ASSET_ID_V2 = REPO_ID + "-top20"
task_subset.register(ASSET_ID_V2, TOP20, mode="exact")

# LoRA. PyTorch 백엔드는 LoRA/FSDP/EMA를 지원하지 않으므로 JAX 백엔드 전용이다.
LORA = {"paligemma_variant": "gemma_2b_lora", "action_expert_variant": "gemma_300m_lora"}

# ---------------------------------------------------------------------------
# v3 (3차 학습). 근거는 docs/HANDOFF.md 15장.
# ---------------------------------------------------------------------------

# 지시문 하나가 가져갈 수 있는 최대 프레임 수. 상위 20종 안에서도 편차가 26.8배라
# (최다 25.70% vs 최소 0.96%) 가장 못 배운 지시문이 학습의 1% 도 못 받는다.
# 상한 100,000 이면 194만 프레임이 남고 최대/최소가 1.44배가 된다.
# 완전 균등(최소값 69,606 으로 전부 잘라내기)은 데이터의 81% 를 버리므로 안 쓴다.
CAP_FRAMES_V3 = 100_000

# 다표본 flow matching 의 표본 수. 프리픽스(연산의 99.5%)를 한 번만 돌리고
# 서픽스만 N 번 돌리므로, N=8 이면 액션 전문가의 기울기 표본이 8배가 된다.
#
# 계산 비용은 **+5% 안팎**이다(계산값). 액션 전문가가 전체 연산의 0.5% 이고
# v3 에서는 그 역전파까지 살아 있으므로, 8번 돌리면 대략 그만큼 늘어난다.
# SigLIP 동결로 버는 -5% 와 거의 상쇄되어 v2 와 비슷한 초/스텝이 나올 것으로
# 보지만, vmap 커널 오버헤드(32토큰짜리 작은 연산 8회)가 더 얹힌다.
# **스모크에서 3.0~3.1초가 나와도 정상이다. 반드시 재측정할 것.**
# 우승팀은 15 를 쓴다.
# 구현은 `configs/pi0_patch.py`, 검증은 `scripts/check_pi0_patch.py`.
NUM_FLOW_SAMPLES_V3 = 8


def is_v3(name: str) -> bool:
    """v3 의 **손실 경로·정규화**를 쓰는 설정인가 (스모크도 같은 경로여야 초/스텝이 맞다).

    ⚠️ 2026-09-13: **v4 도 참이다.** v4 는 v3 와 `freeze_filter` 하나만 다르고
    다표본 flow matching(N=8)·스텝별 정규화는 그대로 쓴다. 이 함수가 이름 접두사
    `pi05_cooking_top20_v3` 만 보던 때에는 v4 가 False 로 나와서
      - `scripts/train.py` 가 **다표본 패치를 조용히 건너뛰고**
      - `scripts/preflight.py` 가 (32,23) 스텝별 통계 검사를 건너뛰었을 것이다.
    에러가 안 나고, 바꾸는 변수가 둘로 늘며, 스모크 초/스텝도 틀리게 나온다.
    """
    return name.startswith(("pi05_cooking_top20_v3", "pi05_cooking_top20_v4",
                            "pi05_cooking_top32_v5"))

ASSET_ID_V3 = REPO_ID + "-top20-v3"

# 3차 학습은 2차 체크포인트에서 이어받는다. Colab 에서는 Drive 경로를 넣어 준다.
#   export COOKING_INIT_PARAMS=/content/drive/MyDrive/cooking-vla/checkpoints/pi05_cooking_top20/run2/12999/params
INIT_PARAMS_V3 = os.environ.get(
    "COOKING_INIT_PARAMS",
    str(pathlib.Path(__file__).resolve().parent.parent
        / "checkpoints/pi05_cooking_top20/run2/12999/params"),
)


def freeze_filter_v3():
    """Action expert 본체를 **풀고**, 대신 SigLIP 시각 인코더를 **얼린다.**

    ## 왜 이걸 바꾸나

    2차 체크포인트에서 파라미터를 직접 세어 보면 배분이 거꾸로 되어 있다.

        SigLIP 시각 인코더    414.8M   100% 학습 중      <- 우승팀은 이걸 얼린다
        Gemma 2B (언어)      2508.5M   얼림 + LoRA 27.9M
        Action expert         427.9M   얼림 + LoRA 22.1M  <- 관절 명령을 만드는 부분

    사전학습이 이미 잘 해 놓은 것(보기)을 414M 통째로 다시 배우면서, 사전학습에
    없는 것(이 로봇의 관절을 움직이기)은 4.9% 만 건드린다. 2025 우승팀 설정은
    `freeze_vision_backbone=True` 이고 action expert 는 전체 학습이다
    (`docs/winner-config-2025.md`).

    ## 왜 공짜인가

    action expert 는 액션 토큰 32개만 처리하고 Gemma 2B 는 프리픽스 918개를
    처리한다. 연산량은 파라미터 x 토큰이라 action expert 는 전체의 0.5% 뿐이다.
    얼리든 풀든 시간이 안 변한다. 오히려 SigLIP 을 얼려서 5% 빨라진다.
    학습 파라미터 467M -> 480M, 상태+기울기 12.4 -> 12.6 GiB (계산값).

    ## 왜 변형 이름은 그대로 두나

    `action_expert_variant` 를 `gemma_300m`(LoRA 제거)으로 바꾸면 파라미터 트리가
    달라져서, `_merge_params` 가 참조 트리에 없는 키를 **조용히 버린다**
    (`weight_loaders.py:88-95`). action expert LoRA 22.1M 이 에러 없이 증발한다.
    `gemma_300m_lora` 를 그대로 두고 이 필터만 바꾸면 트리가 안 변해서
    2차 체크포인트가 무손실로 이어진다.

    ⚠️ `freeze_filter` 는 `tyro.conf.Suppress` 라(`config.py:493`) CLI 로 못 바꾼다.
    반드시 여기서 지정해야 한다.
    """
    llm = _nnx_utils.PathRegex(".*llm.*")
    expert1 = _nnx_utils.PathRegex(".*llm.*_1.*")     # action expert (전문가 1)
    lora = _nnx_utils.PathRegex(".*lora.*")
    siglip = _nnx_utils.PathRegex(r"PaliGemma/img/.*")
    return nnx.Any(
        nnx.All(llm, nnx.Not(expert1), nnx.Not(lora)),  # Gemma 2B 본체만 얼린다
        siglip,                                          # 시각 인코더도 얼린다
    )


def freeze_filter_v4():
    """**SigLIP 만** 얼린다 — Gemma 2B 언어부까지 전부 학습한다 (2026-09-13).

    ## 왜 바꾸나 — 우승팀과 우리의 차이가 여기 하나뿐이다

        항목            우승팀          우리 v3
        SigLIP          동결            동결          <- 같다
        Action expert   전체 학습       전체 학습     <- 같다
        Gemma 2B        **전체 학습**   동결 + LoRA   <- **여기만 다르다**

    `docs/winner-config-2025.md` 를 뜯어 확인했다. 우승팀은 `freeze_filter` 가
    `nnx.Nothing` 이라 LoRA 를 전혀 안 쓰고, 시각 인코더만 모델 쪽 플래그
    (`freeze_vision_backbone=True`)로 얼린다. v4 는 그 배분과 같아진다.

    ## 무엇을 기대하나

    21.4 의 문제 — **같은 물체에 대한 두 동작을 못 고른다** ("문 열어라" 를 주면
    전자레인지를 켠다. 문장을 정반대로 바꾼 효과 1.08배가 같은 문장 재실행의
    잡음 1.00배와 구분되지 않는다) — 의 직접 처방이다.

    문장을 읽는 쪽이 본체의 1.1%(27.9M)만 열려 있으면, 모델은 문장을 해석하는
    것보다 **화면과 관절값만 보고 답을 맞히는 쪽이 손실을 싸게 내린다.**
    `docs/related-work-2026.md` 4장이 이것을 "가장 큰 차이이자 가장 큰 위험" 으로
    적어 두었다 — 비교한 연구 전부가 모델 전체를 학습하는데 우리만 얼렸고,
    **이 방식으로 한 곳이 없어서 기댈 외부 결과가 없다.**

    ## 대가 (`scripts/check_freeze_filter.py` 의 계산값 — 스모크로 반드시 실측할 것)

        학습 파라미터   480.1M -> 2,988.6M  (SigLIP 414.8M 만 빼고 전부)
        상태+기울기     12.6 -> 45.3 GiB   (얼린 것 bf16 2B / 학습분 fp32 4B + 기울기 4B + Adam 8B)
        EMA 사본        7.2 -> 11.9 GiB    (학습분을 fp32 로 복사하므로 **같이 는다**)
        활성값          약 11 GiB          (v3 배치 32 기준 추정. v3 도 LoRA 때문에 Gemma 2B
                                            역전파를 이미 하므로 크게 안 늘 것으로 본다)
        합계            계산 약 68 GiB -> **실측 최대 73.8~73.9 GiB** (09-14 스모크)
                                       A100 40GB 에는 **안 들어간다**
                                       A100 80GB 는 MEM_FRACTION 0.97(한도 77.2 GiB)로 들어간다
                                       G4(0.9, 한도 85.6 GiB)는 여유
        초/스텝         2.74 -> 계산 3.92 -> **실측 A100 80GB 3.5 / G4 3.3** (HANDOFF 39.11)
        체크포인트 1개  18.0 -> 46.1 GiB  (EMA 11.9 + raw params 11.9 + 옵티마이저 22.3)

    ⚠️ **체크포인트가 2.6배라 Drive 여유와 업로드 시간을 먼저 확인해야 한다.**
    한 번 올리는 데 약 68분(11.5 MB/s)이다. 3.92초/스텝이면 저장 간격이 최소 약
    2,100스텝이어야 큐가 다음 저장 전에 비운다 (HANDOFF 23장).

    ## 🔴 이 방향의 위험 — 반드시 같이 잴 것

    학습 데이터는 **지시문 20종 / 194만 프레임 / 시뮬레이터 한 곳**뿐이다.
    언어부 2.5B 를 여기에 전부 맞추면 그 20개 문장 구분에 특화되어, **처음 보는
    문장·물체를 다루는 능력이 떨어질 수 있다.** LoRA 는 업데이트를 좁은 공간에
    가둬 사전학습 능력을 비교적 보존하는데, 본체를 풀면 그 안전장치가 없어진다.

    그래서 **학습 전에 `scripts/rollout_unseen.sh` 로 기준선을 반드시 잴 것.**
    안 재고 학습하면 좋아졌는지 나빠졌는지 영영 알 수 없다.

    SigLIP 을 계속 얼리는 것이 이 위험을 줄이는 가장 확실한 조치다 —
    시뮬레이터 렌더링에 눈을 맞추지 않는 쪽이 실제 환경 전이에 유리하다.
    (다만 렌더링 도메인 갭은 이 프로젝트가 한 번도 측정한 적이 없다, 14.11-2.)

    ## 트리는 안 바뀐다 — run7 을 무손실로 이어받는다

    변형 이름(`gemma_2b_lora` / `gemma_300m_lora`)을 그대로 두고 **필터만** 바꾸므로
    파라미터 트리가 v3 와 같다. `_merge_params` 가 키를 조용히 버리는 사고
    (`weight_loaders.py:88-95`)가 일어나지 않는다. LoRA 어댑터 50.0M 도 그대로
    남아 함께 학습된다 — 본체가 열렸으니 의미는 거의 없지만, 없애려고 변형
    이름을 바꾸면 트리가 깨진다.
    """
    return _nnx_utils.PathRegex(r"PaliGemma/img/.*")   # SigLIP 만 얼린다


task_subset.register(ASSET_ID_V3, TOP20, mode="exact", cap=CAP_FRAMES_V3)


# ---------------------------------------------------------------------------
# v5 (9차 학습). 근거는 docs/HANDOFF.md 40장.
# ---------------------------------------------------------------------------
#
# **v3 와 바꾸는 큰 변수는 하나다 — 지시문 20종 -> 32종.**
# freeze_filter 는 v3 그대로(SigLIP 동결 + Gemma 2B 본체 동결 + LoRA), horizon·
# 토큰·델타 액션·다표본 N=8·EMA·배치·학습률 전부 그대로다.
#
# ## 왜 v4(Gemma 2B 본체까지 학습)로 가지 않나
#
# 8차(run8)가 v4 였는데 **로스는 절반인데 정책은 더 나빴다.**
#
#     최종 로스            run7 ~0.029   ->  run8 0.0146
#     영상 판독 12판        ✅9 △2 ❌1    ->  ✅7 △1 ❌4
#     표현 바꾸기 3판       3/3          ->  3/3        (차이 없음)
#     처음 보는 대상 7판    1            ->  0
#     몸통이 데모 범위 밖    34판 중 5     ->  34판 중 8
#
# 언어 이해가 좋아진 흔적은 없고(표현 바꾸기 동일), 몸통 자세만 흐트러졌다.
# 게다가 v3 의 언어부 LoRA 를 0 으로 만드는 절제 실험에서, v3 도 언어부를 이미
# 제대로 학습하고 있었음이 확인됐다 (40.2):
#
#     원본                 오차 0.02425 / 지시문 효과 +277.9%
#     언어부 LoRA 0        오차 0.08060 (3.3배) / 지시문 효과 +3.7%   <- 무너진다
#     액션부 LoRA 0        오차 0.02626 (1.08배) / 지시문 효과 +256.9% <- 거의 그대로
#
# 언어 쪽 학습이 전부 LoRA 27.9M 안에만 쌓이므로(본체가 얼려 있으니) 그것을 빼면
# 무너지고, 액션 전문가는 본체가 이미 다 학습돼서 LoRA 를 빼도 그대로다.
# 즉 v3 의 언어부는 "덤"이 아니라 **작동하는 학습 경로**다.
ASSET_ID_V5 = REPO_ID + "-top32-v5"

# 상한은 v3 와 **같은 100,000 을 쓴다.** 신규 12종은 전부 121,565 프레임 이상이라
# 똑같이 잘리고, 최대/최소 비율이 1.44배로 v3 와 같게 유지된다
# (최소는 TOP20 의 `chop the half bell pepper with the parer` 69,606).
#
# ⚠️ **기존 20종의 상한을 낮춰 신규 12종에 가중치를 주고 싶은 유혹이 있다.**
# 신규 12종은 누적 노출이 0 에서 시작하는데 기존 20종은 이미 25만 회를 넘겼기
# 때문이다. 그래도 하지 않는다 — 바꾸는 변수가 둘이 되어, 신규가 안 되는 것이
# "어려워서"인지 "기존을 덜 봐서 무너진 것"인지 갈라낼 수 없게 된다.
# 이번 차수를 평가한 뒤에 따로 결정한다.
CAP_FRAMES_V5 = CAP_FRAMES_V3

task_subset.register(ASSET_ID_V5, TOP32, mode="exact", cap=CAP_FRAMES_V5)

# 9차는 **7차 체크포인트**에서 이어받는다 (8차가 아니다 — 위 표 참조).
# 파라미터 트리가 v3 와 같으므로 무손실로 복원된다.
#   export COOKING_INIT_PARAMS_V5=/content/drive/MyDrive/cooking-vla/checkpoints/pi05_cooking_top20_v3/run7/25999/params
INIT_PARAMS_V5 = os.environ.get(
    "COOKING_INIT_PARAMS_V5",
    str(pathlib.Path(__file__).resolve().parent.parent
        / "checkpoints/pi05_cooking_top20_v3/run7/25999/params"),
)

# 스텝 수. 배치 32 이므로 지시문당 노출 = 스텝 수와 같다 (32*스텝/32종).
#
#     스텝      지시문당 노출   run7(26,000스텝 20종 = 41,600회) 대비   시간(3.0초/스텝)
#     20,000    20,000        48.1%                                 16.7시간
#     22,000    22,000        52.9%                                 18.3시간   <- 고른 값
#     26,000    26,000        62.5%                                 21.7시간
#     32,000    32,000        76.9%                                 26.7시간
#     41,600    41,600       100.0%                                 34.7시간
#
# **22,000 을 고른다** — 세션을 **22시간**으로 잡고 거기서 역산한 값이다
# (2026-09-24 변경, HANDOFF 40.9).
#
#     22시간 - 준비 1.5시간 - 데이터셋·컴파일 0.15시간 - 여유 0.83시간 = 19.5시간
#     19.5시간 / 3.0초 = 약 23,400  ->  저장 간격 2,000 의 배수로 내려 22,000
#
# ## 왜 공칭 24시간이 아니라 22시간인가
#
# 8차가 24시간 벽에 부딪혀 **마지막 체크포인트를 잃었다**(39.12). 공칭값을 그대로
# 쓰면 계획 자체에 여유가 0 이라, 카드가 조금만 느리거나 세션이 일찍 끊기면 바로
# 마지막 저장이 날아간다. 그리고 여유 0.83시간은 임의의 값이 아니라
# **마지막 체크포인트를 Drive 에 다 올리는 시간(18.1 GiB / 11.5 MB/s = 27분)**
# + 검증 8분 + 벽 여유 15분이다. 노트북 계산 셀이 체크포인트 크기에서 자동으로 낸다.
#
# ⚠️ 이 값은 **자리표시자**다. 노트북 계산 셀이 그날의 경과 시간·CU 잔액·스모크
# 실측 초/스텝으로 다시 내고, 본 학습 셀은 그 값을 CLI 로 덮어쓴다.
#
# 노출이 run7 의 52.9% 라 신규 12종은 이번 차수에 덜 익는다 — 그건 평가로 확인하고
# 필요하면 10차에서 이어 돌린다.
NUM_STEPS_V5 = 22_000


@dataclasses.dataclass(frozen=True)
class CookingDataConfig(_config.DataConfigFactory):
    """세그먼트 단위 BEHAVIOR 요리 데이터셋."""

    # 절대 관절 목표값을 델타로 바꿀지. 근거는 docs/phase3-action-space.md.
    delta_actions: bool = True
    # False 면 state 를 원본 61-D 그대로 쓴다 (2026-08-27 체크포인트 분석 전용).
    select_state: bool = True

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        # build_dataset.py 가 쓴 컬럼명 -> 정책 입력 키
        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "observation/image": "image",
                    "observation/left_wrist_image": "left_wrist_image",
                    "observation/right_wrist_image": "right_wrist_image",
                    "observation/state": "state",
                    "actions": "actions",
                    "prompt": "prompt",
                })
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[r1pro_policy.R1ProInputs(model_type=model_config.model_type,
                                             select_state=self.select_state)],
            outputs=[r1pro_policy.R1ProOutputs()],
        )
        if self.delta_actions:
            # 대응표는 state 를 줄였는지에 따라 달라진다. 어긋나면 에러 없이
            # 엉뚱한 관절값을 빼게 되므로 여기서 한 곳에서 고른다.
            a2s = tuple(sorted(
                (r1pro_policy.ACTION_TO_STATE if self.select_state
                 else r1pro_policy.ACTION_TO_STATE_RAW).items()))
            # 정규화 앞에 놓인다 — 통계가 델타 분포로 잡히도록.
            data_transforms = data_transforms.push(
                inputs=[r1pro_policy.R1ProDeltaActions(action_to_state=a2s)],
                outputs=[r1pro_policy.R1ProAbsoluteActions(action_to_state=a2s)],
            )

        return dataclasses.replace(
            self.create_base_config(pathlib.Path(ASSETS_DIR), model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory()(model_config),
        )


def _model_config(max_token_len=MAX_TOKEN_LEN, action_horizon=None, **kw):
    # discrete_state_input 은 지정하지 않는다 — pi05=True 의 기본값(True)이 살아야
    # state 가 32-D 연속 입력이 아니라 이산 언어 토큰으로 들어간다.
    # 템플릿 pi05_libero 는 이걸 False 로 명시 오버라이드하므로 베끼면 안 된다.
    # (state 를 25-D 로 줄인 뒤에는 False 도 기술적으로 가능해졌지만,
    #  공식 baseline 도 True 이고 한 번에 바꾸는 것을 늘리지 않기 위해 유지한다.)
    extra = {} if action_horizon is None else {"action_horizon": action_horizon}
    return pi0_config.Pi0Config(pi05=True, max_token_len=max_token_len, **extra, **kw)


def _base(name: str, num_train_steps: int, batch_size: int = 256, *,
          v2: bool = False, v3: bool = False, v4: bool = False,
          v5: bool = False) -> _config.TrainConfig:
    """v2=True 면 2차 설정 (state 25-D / horizon 32 / 상위 20종 / 토큰 150).

    v3=True 면 2차 설정에 더해 3차 학습의 다섯 가지가 붙는다 (HANDOFF 15.10):
      1. `freeze_filter` 교체 — action expert 본체 해제 + SigLIP 동결
      2. 스텝별 정규화 통계 (asset_id 가 갈라진다. 통계 파일 자체가 (32,23))
      3. 다표본 flow matching — 모델이 아니라 `configs/pi0_patch.py` 가 담당
      4. EMA 0.99 (openpi TrainConfig 기본값. LoRA 때문에 껐던 것을 되돌린다)
      5. 지시문 상한 100,000 프레임 — `task_subset.register(..., cap=)`
    그리고 사전학습 대신 **2차 체크포인트에서 이어받는다.**

    v4=True 면 v3 와 **모든 것이 같고 `freeze_filter` 하나만 다르다** — Gemma 2B
    언어부까지 전부 학습한다(`freeze_filter_v4`). 데이터·정규화·EMA·상한은 그대로라
    v3 체크포인트를 무손실로 이어받는다. 근거와 대가는 `freeze_filter_v4` 의
    설명을 볼 것. **초/스텝이 1.25배(A100 80GB 실측), 체크포인트가 2.6배가 되므로 스모크 실측 전에
    본 학습을 시작하지 말 것.**
    """
    shape = v2 or v3 or v4 or v5
    mk = dict(max_token_len=MAX_TOKEN_LEN_V2, action_horizon=ACTION_HORIZON_V2) if shape else {}
    model = _model_config(**mk, **LORA)
    asset_id = (ASSET_ID_V5 if v5 else ASSET_ID_V3 if (v3 or v4)
                else ASSET_ID_V2 if v2 else ASSET_ID_V1)
    return _config.TrainConfig(
        name=name,
        model=model,
        data=CookingDataConfig(
            repo_id=REPO_ID,
            assets=_config.AssetsConfig(asset_id=asset_id),
            select_state=shape,
            base_config=_config.DataConfig(prompt_from_task=True),
        ),
        batch_size=batch_size,
        # 기본 2로는 데이터가 병목이다. 워커 4개로 배치당 0.84초(306 샘플/초)를
        # 실측했고 로컬 16코어에서 선형으로 늘었다. Colab vCPU 수에 맞춰 조정할 것
        # (`--num-workers` 로 덮어쓸 수 있다).
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            # optax 는 내부에서 (decay_steps - warmup_steps) 를 코사인 구간으로 쓴다.
            # warmup 을 1,000 으로 고정하면 500 스텝짜리 스모크에서 -500 이 되어
            # "requires positive decay_steps, got decay_steps=-500" 으로 죽는다.
            # 전체 스텝의 1/10 을 예열로 쓰되 1,000 을 넘기지 않는다.
            warmup_steps=min(1_000, max(1, num_train_steps // 10)),
            peak_lr=5e-5, decay_steps=num_train_steps, decay_lr=5e-6
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            # v3 는 사전학습이 아니라 **2차 체크포인트**에서 이어받는다.
            # 파라미터 트리가 v2 와 같으므로 무손실로 복원된다 (HANDOFF 15.11).
            # ⚠️ `--resume` 은 못 쓴다 — 옵티마이저 상태 모양이 달라지므로
            #    params 만 이어받고 Adam 모멘텀과 학습률 위치는 초기화된다.
            INIT_PARAMS_V5 if v5 else
            INIT_PARAMS_V3 if (v3 or v4) else "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=num_train_steps,
        # LoRA 만 학습할 때는 EMA 를 껐다 (openpi 의 LoRA 예시와 동일).
        # v3 는 action expert 본체 428M 을 학습하므로 openpi 기본값 0.99 로 되돌린다.
        # 짧은 학습의 잡음을 줄인다. 비용은 파라미터 사본 하나 +7.2 GiB, 시간은 거의 0.
        # v4 도 켠다. 다만 EMA 사본은 학습분을 fp32 로 들고 있어 7.2 -> 11.9 GiB 로 는다.
        ema_decay=0.99 if (v3 or v4 or v5) else None,
        freeze_filter=(freeze_filter_v4() if v4 else
                       freeze_filter_v3() if (v3 or v5) else model.get_freeze_filter()),
    )


CONFIGS = [
    # --- v1: 2026-08-27 에 돌린 설정. 지금은 그때 만든 체크포인트를 분석할 때만 쓴다.
    #     state 61-D / horizon 50 / 지시문 466종. 결과는 docs/HANDOFF.md 8~9장.
    _base("pi05_cooking", 30_000),
    dataclasses.replace(_base("pi05_cooking_smoke", 500), save_interval=250, log_interval=10),

    # --- v2: 재학습 설정. 바뀐 것 네 가지 (근거는 docs/HANDOFF.md 10~11장)
    #     state 61 -> 25 / max_token_len 280 -> 150 / horizon 50 -> 32 / 지시문 466 -> 20종
    _base("pi05_cooking_top20", 11_000, v2=True),
    dataclasses.replace(_base("pi05_cooking_top20_smoke", 500, v2=True),
                        save_interval=250, log_interval=10),

    # --- v3: 3차 학습. 바뀐 것 다섯 가지 (근거는 docs/HANDOFF.md 15장)
    #     freeze_filter 교체 / 스텝별 정규화 / 다표본 flow matching /
    #     EMA 0.99 / 지시문 상한 100,000. 2차 체크포인트에서 이어받는다.
    #
    # ⚠️ 배치 기본값을 32 로 둔다. v1/v2 는 256 이라 `--batch-size=32` 를 매번
    #    붙여야 했는데, 빠뜨리면 배치 256 으로 돌아 OOM 이 나거나 스텝이 8배
    #    무거워진다. 기본값을 실제 쓰는 값으로 맞춰 그 사고를 없앤다.
    _base("pi05_cooking_top20_v3", 14_000, 32, v3=True),
    dataclasses.replace(_base("pi05_cooking_top20_v3_smoke", 500, 32, v3=True),
                        save_interval=250, log_interval=10),

    # --- v4: 8차 후보. v3 와 **freeze_filter 하나만** 다르다 — Gemma 2B 언어부까지
    #     전부 학습한다. 그러면 파라미터 배분이 2025 우승팀과 같아진다
    #     (SigLIP 동결 / action expert 전체 / 언어부 전체). 근거는 `freeze_filter_v4`.
    #
    # ⚠️ **v3 보다 무겁다.** 학습 480M -> 2,989M, 상태+기울기 12.6 -> 45.3 GiB,
    #    EMA 7.2 -> 11.9 GiB, 체크포인트 18.0 -> 46.1 GiB.
    #    09-14 실측: VRAM 최대 73.9 GiB, 초/스텝 A100 80GB 3.5 · G4 3.3.
    #    **A100 40GB 에는 안 들어간다.** A100 80GB 는 MEM_FRACTION 0.97 이 필요하다.
    #    G4(96GB)가 맞다. A100 80GB 는 한도 72 GiB 라 빠듯하다.
    #
    # ⚠️ 이 설정으로 돌리기 전에 `scripts/rollout_unseen.sh` 로 **미학습 일반화
    #    기준선을 먼저 잴 것.** 언어부를 20종에 맞추면 처음 보는 문장·물체를
    #    다루는 능력이 떨어질 수 있는데, 지금 그 값을 한 번도 잰 적이 없다.
    _base("pi05_cooking_top20_v4", 14_000, 32, v4=True),
    dataclasses.replace(_base("pi05_cooking_top20_v4_smoke", 500, 32, v4=True),
                        save_interval=250, log_interval=10),

    # --- v5: 9차 학습. **v3 와 지시문 수 하나만 다르다** — 20종 -> 32종.
    #     freeze_filter 는 v3 그대로다(SigLIP 동결 + Gemma 2B 본체 동결 + LoRA).
    #     8차(v4)는 로스가 절반인데 영상 판독이 더 나빠서 계보를 닫고,
    #     **7차 체크포인트(run7/25999)에서 이어받는다.** 근거는 위 v5 블록과
    #     docs/HANDOFF.md 40장.
    #
    # ⚠️ **먼저 정규화 통계를 내야 한다.** asset_id 가 `-top32-v5` 로 갈라졌으므로
    #    v3 통계를 물려받지 않는다. 안 내고 돌리면 "Norm stats not found" 를
    #    **경고만 하고 정규화 없이** 학습한다 (ASSETS_DIR 주석 참조).
    #
    #        uv run --project third_party/openpi python \
    #          scripts/compute_norm_stats_perstep.py pi05_cooking_top32_v5
    #
    # ⚠️ 무게는 v3 와 같다(학습 480.1M). A100 40GB 로 충분하다.
    _base("pi05_cooking_top32_v5", NUM_STEPS_V5, 32, v5=True),
    dataclasses.replace(_base("pi05_cooking_top32_v5_smoke", 500, 32, v5=True),
                        save_interval=250, log_interval=10),
]


def register() -> None:
    """openpi의 설정 목록에 위 설정들을 끼워 넣는다."""
    for c in CONFIGS:
        if c.name in _config._CONFIGS_DICT:
            continue
        _config._CONFIGS.append(c)
        _config._CONFIGS_DICT[c.name] = c
