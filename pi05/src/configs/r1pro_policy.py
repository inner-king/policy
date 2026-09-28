"""R1 Pro(BEHAVIOR-1K) 데이터를 pi0.5 입력 형식으로 옮기는 변환들.

차원별 의미는 원본 메타데이터에 없어서 실측으로 알아냈다. 근거와 수치는
`docs/phase3-action-space.md` 에 있고, 여기서는 그 결론만 코드로 옮긴다.

  action 23-D = 베이스 3 + 토르소 4 + 왼팔 7 + 왼그리퍼 1 + 오른팔 7 + 오른그리퍼 1
  state  원본 61-D 중 **25개만** 골라 쓴다 (아래 STATE_SELECT).

## 왜 61개를 다 쓰지 않는가

pi0.5 는 state 를 연속값이 아니라 **언어 토큰**으로 넣는다. 프롬프트가
`"Task: {지시문}, State: {정수 나열};\nAction: "` 형태라(`tokenizer.py:26-29`)
**state 차원 수가 그대로 토큰 수가 된다.**

61차원을 다 넣었을 때 실측:

    지시문      약  15 토큰 (  6%)
    state 61개  약 200 토큰 ( 77%)

지시문이 프롬프트의 6% 로 밀렸고, 실제로 학습된 정책은 **지시문을 무시했다**
(절제 실험: 지시문을 바꿔도 오차 0.92배. 근거는 docs/HANDOFF.md 9장).

대회 공식 baseline(`wensi-ai/openpi` 의 `configs/robots/b1k.py`)은 25개만 쓴다.
그 선택을 그대로 따르면 state 가 약 80토큰이 되어 지시문 비중이 15% 로 오른다.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# 원본 61차원에서 골라 쓸 인덱스와 그 순서. 공식 baseline 과 같은 구성이고,
# 우리 정규화 통계로도 검증했다 — 24,25 와 49,50 이 모두 [0, 0.05] 범위로
# 그리퍼 손가락 위치(m)와 일치한다.
STATE_SELECT = (
    0, 1, 2,                              # 베이스 x, y, yaw (속도)
    53, 54, 55, 56,                       # 토르소 4관절
    3, 4, 5, 6, 7, 8, 9,                  # 왼팔 7관절
    24, 25,                               # 왼 그리퍼 손가락 2개
    28, 29, 30, 31, 32, 33, 34,           # 오른팔 7관절
    49, 50,                               # 오른 그리퍼 손가락 2개
)
RAW_STATE_DIM = 61                        # 데이터셋이 내주는 원본 차원
STATE_DIM = len(STATE_SELECT)             # 25

# 액션 차원 -> **축소된** state 안에서의 대응 위치.
#
# ⚠️ 원본 61차원 기준이 아니라 STATE_SELECT 로 재배열된 뒤의 인덱스다.
# STATE_SELECT 를 손대면 이 표도 반드시 같이 고쳐야 한다. 어긋나도 에러가
# 나지 않고 엉뚱한 관절값을 빼게 되어(델타가 망가진다) 알아채기 어렵다.
#
# 위 순서를 액션 순서에 맞춰 잡아 둔 덕분에 대응이 거의 항등이다:
#   액션  0~13 -> state  0~13,   액션 15~21 -> state 16~22 (왼 그리퍼 2칸 뒤)
# 그리퍼 명령(14, 22)은 대응 관절이 없어 빠져 있다.
ACTION_TO_STATE = {
    0: 0, 1: 1, 2: 2,                                        # 베이스
    3: 3, 4: 4, 5: 5, 6: 6,                                  # 토르소
    7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13,        # 왼팔
    15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22,  # 오른팔
}

# 원본 61차원을 그대로 쓰던 시절의 표. 2026-08-27 에 학습한 체크포인트
# (run1/9999)를 분석할 때만 쓴다. 새 학습에는 위 표를 쓸 것.
ACTION_TO_STATE_RAW = {
    0: 0, 1: 1, 2: 2,
    3: 53, 4: 54, 5: 55, 6: 56,
    7: 3, 8: 4, 9: 5, 10: 6, 11: 7, 12: 8, 13: 9,
    15: 28, 16: 29, 17: 30, 18: 31, 19: 32, 20: 33, 21: 34,
}

def action_to_state_for(state_dim: int) -> dict[int, int]:
    """state 차원을 보고 대응표를 고른다.

    어느 표를 써야 하는지는 **state 가 축소를 거쳤는지**에 달려 있다.
    `create_torch_dataset` 은 정책 입력 변환보다 앞 단계라 원본 61차원을
    그대로 내주고, 변환을 거친 뒤는 25차원이다. 진단 스크립트가 그것을 모른 채
    25-D 표를 쓰면 엉뚱한 관절값을 base 로 빼게 되는데, **에러가 나지 않는다.**
    2026-08-29 에 실제로 겪었다 — 절제 실험의 '가만히' 기준선이 0.0086 대신
    0.506 으로 60배 부풀어 나왔다.

    (절제 실험의 배율은 base 가 예측·정답 양쪽에서 상쇄되어 무사했지만,
     `eval_offline.py` 처럼 '가만히 대비'가 판정 기준인 검사는 통째로 망가진다.)
    """
    if state_dim == STATE_DIM:
        return ACTION_TO_STATE
    if state_dim == RAW_STATE_DIM:
        return ACTION_TO_STATE_RAW
    raise ValueError(
        f"state 차원이 {state_dim} 이다 — {RAW_STATE_DIM}(원본) 도 "
        f"{STATE_DIM}(축소) 도 아니라 어느 대응표를 써야 할지 알 수 없다")


# 델타로 바꿀 차원: 토르소와 양팔만.
# 베이스는 위치가 아니라 속도 명령이라 이미 델타 성격이고(델타화해도 범위가
# 1.0~1.2배밖에 안 줄고 의미가 가속도가 된다), 그리퍼는 대응 state가 없는
# [-1, 1] 개폐 명령이다. 근거표는 docs/phase3-action-space.md 참조.
DELTA_DIMS = tuple(list(range(3, 14)) + list(range(15, 22)))

ACTION_DIM = 23


def make_r1pro_example() -> dict:
    """정책 서버 테스트용 무작위 입력."""
    img = lambda: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    return {
        "observation/state": np.random.rand(RAW_STATE_DIM),
        "observation/image": img(),
        "observation/left_wrist_image": img(),
        "observation/right_wrist_image": img(),
        "prompt": "chop the onion with the carving knife",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _select_state(state) -> np.ndarray:
    """원본 61-D state 에서 학습에 쓰는 25-D 만 골라 낸다.

    이미 25-D 로 줄여서 들어온 경우도 그대로 통과시킨다(정책 서버가 미리
    줄여 보내는 구성을 허용). 그 외 차원은 조용히 넘기지 않고 즉시 실패시킨다
    — 차원이 어긋난 채로 학습이 돌면 에러 없이 결과만 망가진다.
    """
    arr = np.asarray(state, dtype=np.float32)
    if arr.shape[-1] == STATE_DIM:
        return arr
    if arr.shape[-1] == RAW_STATE_DIM:
        return arr[..., list(STATE_SELECT)]
    raise ValueError(
        f"state 차원이 {arr.shape[-1]} 이다. {RAW_STATE_DIM}(원본) 또는 "
        f"{STATE_DIM}(축소본) 이어야 한다."
    )


@dataclasses.dataclass(frozen=True)
class R1ProInputs(transforms.DataTransformFn):
    """데이터셋/추론 입력을 모델이 기대하는 키로 옮긴다.

    카메라 3대가 전부 실물이라 libero와 달리 zero-padding 이나 mask=False 가 없다.

    state 는 원본 61-D 를 받아 **STATE_SELECT 로 25-D 로 줄여서** 내보낸다.
    pi0.5 에서 state 는 연속 입력이 아니라 `TokenizePrompt` 의 이산 토큰으로
    들어가므로 차원 수가 곧 토큰 수다(위 모듈 설명 참조).
    뒤따르는 `PadStatesAndActions(32)` 가 25 -> 32 로 0 을 채운다.
    """

    model_type: _model.ModelType
    # False 로 두면 원본 61-D 를 그대로 흘려보낸다 (구 체크포인트 분석 전용).
    select_state: bool = True

    def __call__(self, data: dict) -> dict:
        raw = data["observation/state"]
        inputs = {
            "state": _select_state(raw) if self.select_state else np.asarray(raw, np.float32),
            "image": {
                "base_0_rgb": _parse_image(data["observation/image"]),
                "left_wrist_0_rgb": _parse_image(data["observation/left_wrist_image"]),
                "right_wrist_0_rgb": _parse_image(data["observation/right_wrist_image"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class R1ProOutputs(transforms.DataTransformFn):
    """모델 출력(32-D 패딩)에서 실제 액션 23-D만 되돌린다."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}


@dataclasses.dataclass(frozen=True)
class R1ProDeltaActions(transforms.DataTransformFn):
    """절대 관절 목표값을 청크 첫 state 기준 델타로 바꾼다.

    openpi의 `transforms.DeltaActions` 를 쓸 수 없는 이유는 그쪽이
    `actions[..., :n] -= state[..., :n]` 이라 **액션 차원 i가 state 차원 i에
    대응한다고 가정**하기 때문이다(`transforms.py:219`). R1 Pro는 대응이 흩어져
    있어서(토르소 53–56, 왼팔 3–9, 오른팔 28–34) 인덱스 맵을 명시로 받는다.

    이 변환은 정규화 **전**에 놓여야 한다. `compute_norm_stats.py` 가
    `data_transforms` 를 적용한 뒤 통계를 내므로(스크립트 35–40행)
    q01/q99가 델타 분포로 잡히고, 따로 손댈 것이 없다.
    """

    dims: tuple[int, ...] = DELTA_DIMS
    # 축소 state(25-D) 기준 표가 기본. 구 체크포인트는 ACTION_TO_STATE_RAW 를 준다.
    action_to_state: tuple[tuple[int, int], ...] = tuple(sorted(ACTION_TO_STATE.items()))

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        # 로더는 torch.Tensor 를 내주므로 numpy로 맞춘다 (추론 경로는 이미 numpy다).
        actions = np.asarray(data["actions"], dtype=np.float32).copy()   # (..., horizon, 23)
        state = np.asarray(data["state"], dtype=np.float32)
        m = dict(self.action_to_state)
        for j in self.dims:
            actions[..., j] -= state[..., m[j]][..., None]
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class R1ProAbsoluteActions(transforms.DataTransformFn):
    """추론 시 델타 예측을 다시 절대 목표값으로 되돌린다 (위의 역변환)."""

    dims: tuple[int, ...] = DELTA_DIMS
    action_to_state: tuple[tuple[int, int], ...] = tuple(sorted(ACTION_TO_STATE.items()))

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        actions = np.asarray(data["actions"], dtype=np.float32).copy()
        state = np.asarray(data["state"], dtype=np.float32)
        m = dict(self.action_to_state)
        for j in self.dims:
            actions[..., j] += state[..., m[j]][..., None]
        data["actions"] = actions
        return data
