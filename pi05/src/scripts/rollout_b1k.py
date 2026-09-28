"""데모로 원하는 스킬 지점까지 몰고 간 뒤, 거기서부터 정책에게 로봇을 맡긴다.

    conda run -n behavior python scripts/rollout_b1k.py \
        --task slicing_vegetables --demo outputs/demo/slicing_vegetables_ep8652.npz \
        --skill 1 --write-video

**conda `behavior` 환경에서 돌린다.** 정책은 별도 프로세스로 미리 띄워 둘 것:

    third_party/openpi/.venv/bin/python scripts/serve_policy_b1k.py --device cpu

## 왜 데모로 몰고 가나

정책은 **이동을 못 한다.** 2차 학습은 지시문 상위 20종만 썼고 `move to ...` 는
거기 없다. 그런데 학습된 스킬 10개는 전부 태스크 시작점이 아니라 중간에 있다
(`open the door of the fridge` 는 618프레임짜리 이동 뒤에 온다).

그래서 그 지점까지는 **사람이 한 녹화 액션을 그대로 흘려 넣고**, 도착한 다음
정책에 핸들을 넘긴다. 시뮬레이터 컨트롤러가 절대 관절값을 받고
(`use_delta_commands: false`) 데모 액션도 절대값이라 그대로 통한다.

## 실행 방식

정책 구간에서는 공식 baseline 과 같게 **32스텝을 예측해 앞의 16만 실행**하고
다시 예측한다(receding horizon).

## ⚠️ 알고 볼 것

- **인스턴스가 데모와 다를 수 있다.** 데모가 어느 장면 인스턴스에서 녹화됐는지
  주석에 없다. 냉장고 같은 **가구는 장면에 고정**이라 이동 재생은 같은 자리에
  도착하겠지만, 재료 위치는 인스턴스마다 다르다. 그래서 재료를 집어 옮긴 뒤의
  상태에 의존하는 스킬(썰기 등)은 재생이 어긋날 수 있다. **영상으로 확인할 것.**
- 여기서 나오는 성공/실패는 **그 스킬 하나**에 대한 것이지 태스크 전체가 아니다.
"""
import re
import argparse, json, os, sys, time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="slicing_vegetables")
    p.add_argument("--demo", required=True, help="export_demo_actions.py 가 만든 npz")
    p.add_argument("--skill", type=int, required=True, help="정책에 넘길 스킬 번호")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--instance", type=int, default=0)
    p.add_argument("--mode", default="train", choices=("train", "public_test", "hidden_test"))
    p.add_argument("--exec-steps", type=int, default=16, help="32 중 실행할 스텝 수")
    p.add_argument("--extra-steps", type=int, default=0,
                   help="스킬 길이를 넘겨 더 돌려 볼 스텝 수")
    p.add_argument("--null-policy", action="store_true",
                   help="🔬 대조군: 정책에 묻지 않고 **인계 자세를 그대로 굳힌다** "
                        "(바퀴는 0). 성공 판정은 그대로 돌린다.\n"
                        "왜 필요한가 — 재생이 로봇을 목표 바로 앞에 세워 주므로, "
                        "성공이 정책 덕분인지 재생·물리 잔여운동 덕분인지 구분되지 "
                        "않는다. 이 모드로 돌려서 그래도 성공하면 그 무대의 성공은 "
                        "정책의 성과가 아니다.")
    p.add_argument("--stop-on-success", action="store_true",
                   help=("스킬을 실제로 해내면 조금 더 돌린 뒤 멈춘다. "
                         "하네스의 `task.success` 는 태스크 전체(예: 통조림 만들기) "
                         "기준이라 스킬 하나로는 절대 True 가 안 되므로, 지시문에서 "
                         "목표 물체와 조건을 뽑아 시뮬레이터 상태로 직접 판정한다."))
    p.add_argument("--success-hold", type=int, default=15,
                   help="성공 조건이 이만큼 연속으로 유지돼야 인정한다 (물리 튐 방지)")
    # ── "크게 열리면 멈춘다" ─────────────────────────────────────────────
    # `--stop-on-success` 는 판정기(관절 범위의 5%)를 쓰므로 **살짝만 열려도**
    # 멈춘다 (32.2). 반대로 열림 정도만 보고 싶은 실험 — 예를 들어 "왼팔을
    # 쓰는가" 만 확인하는 짧은 판 — 에서는 문턱을 직접 주는 편이 낫다.
    p.add_argument("--preset-open", type=float, default=None,
                   help=(
                       "인계 시점에 목표 물체의 문을 이 비율(0~1)만큼 **직접 열어 "
                       "놓는다.** 36.9 의 '닭과 달걀' 가설을 가르는 장치다 — 문을 "
                       "미리 열어 주면 정책이 왼팔을 부르는지 본다.\n"
                       "⚠️ 열어 놓으면 `open the door` 판정은 **시작부터 참**이 되어 "
                       "꺼진다(문턱이 5%%). 이 실험은 성공 판정이 아니라 **팔을 쓰는지**"
                       "를 보는 것이므로 그래도 된다."))
    p.add_argument("--stop-open-frac", type=float, default=None,
                   help=("목표 물체의 열린 비율이 이 값(0~1)을 넘으면 "
                         "--stop-open-extra 만큼 더 돌리고 멈춘다. "
                         "판정기의 5%% 문턱과 달리 **크게 열렸을 때만** 멈춘다."))
    p.add_argument("--stop-open-extra", type=int, default=150,
                   help="--stop-open-frac 을 넘은 뒤 더 돌릴 스텝 (기본 150 = 5초)")
    # 🔑 36.14: 냉장고 실패 판은 왼팔 전완 비틀기(명령 11번)가 데모 범위
    # (-0.194~+0.176)를 8~13배 벗어나 한 방향으로 계속 돌다가 굳는다. 그것이
    # 원인인지 증상인지는 **묶어 보면** 안다. 판을 더 돌리는 것보다 세다.
    p.add_argument("--clamp", action="append", default=None, metavar="차원:하한:상한",
                   help=(
                       "정책이 낸 명령의 한 차원을 범위 안으로 잘라서 보낸다. "
                       "여러 번 줄 수 있다. 예: `--clamp 11:-0.25:0.25` 는 왼팔 "
                       "전완 비틀기를 데모 범위로 묶는다.\n"
                       "23차원 자리: 바퀴 0~2 / 몸통 3~6 / 왼팔 7~13 / 왼그리퍼 14 "
                       "/ 오른팔 15~21 / 오른그리퍼 22.\n"
                       "⚠️ **정책 구간에만** 걸린다. npz 의 actions 에는 자른 뒤 "
                       "값이 남고, 정책이 원래 얼마나 더 가려 했는지는 로그의 "
                       "'명령 묶기' 줄에 남는다."))
    # 🔑 37장: 냉장고 갈림길에서 정책이 오른팔을 다시 뻗느냐 왼팔을 뻗느냐는 몸을
    # 얼마나 돌았느냐로 갈린다(학습 데이터 문턱 -56.5°, 롤아웃도 같은 문턱). 그것이
    # 원인인지는 **회전만 바꿔 보면** 안다.
    p.add_argument("--yaw-assist", default=None, metavar="시작°:목표°:rad/s",
                   help=(
                       "정책 구간에서 누적 회전이 시작값을 넘으면, 목표값에 닿을 때까지 "
                       "회전 명령(23차원의 2번)을 이 속도로 대신 준다. 예: "
                       "`--yaw-assist -30:-65:-0.25`. 셋 다 음수여야 한다(오른쪽으로 돈다).\n"
                       "5초 안에 목표에 못 닿으면 스스로 꺼진다. 로그의 '[회전 보조]' 줄에 "
                       "켜고 끈 시각과 누적 회전이 남는다."))
    p.add_argument("--success-extra", type=int, default=450,
                   help="성공을 인정한 뒤 더 돌릴 스텝 수 (30fps 기준 450=15초)")
    p.add_argument("--replay-frames", type=int, default=None,
                   help=(
                       "데모를 몇 프레임 재생한 뒤 정책에 넘길지. 생략하면 그 스킬의 "
                       "시작 지점까지 전부 재생한다. 0 이면 재생 없이 처음부터 정책이 운전한다.\n"
                       "⚠️ 평가 하네스의 초기 로봇 위치는 데모의 시작 위치와 다르다 "
                       "(실측: slicing_vegetables 는 이미 냉장고 앞에서 시작한다). "
                       "게다가 베이스는 속도 명령이라 재생하면 오차가 누적된다. "
                       "이미 목표 지점에 서 있다면 0 을 주는 편이 정확하다."))
    p.add_argument("--output-dir", default="outputs/rollout")
    p.add_argument("--write-video", action="store_true", default=True)
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--place-near", default=None,
                   help=(
                       "이 이름이 들어간 물체 앞으로 로봇을 옮겨 놓고 시작한다 (예: fridge).\n"
                       "⚠️ 정책은 이동을 못 한다 — 2차 학습은 지시문 상위 20종만 썼고 "
                       "`move to ...` 는 거기 없다. 그런데 학습된 스킬은 전부 어딘가로 "
                       "이동한 뒤에 일어난다. 데모의 이동 액션을 재생하는 방법은 "
                       "실패했다(하네스의 초기 위치가 데모와 다르고, 베이스는 속도 명령이라 "
                       "적분 오차가 쌓인다). 그래서 좌표로 직접 옮긴다."))
    p.add_argument("--place-dist", type=float, default=0.8,
                   help="물체로부터 몇 m 앞에 세울지")
    p.add_argument("--place-angle", type=float, default=None,
                   help=(
                       "물체에서 본 **어느 쪽**에 세울지 (도, 0=동 90=북).\n"
                       "생략하면 '로봇의 현재 위치 방향' 이라는 어림을 쓰는데, 그건 "
                       "물체의 어느 면인지를 모른다. 실측: 냉장고 기준 리셋 방향은 "
                       "126.8도(북서 모서리)이고 데모가 선 곳은 184.1도(정서쪽)다."))
    p.add_argument("--config", default=None,
                   help=(
                       "정책 서버에 띄운 학습 설정 이름. **설명 파일의 '학습된 지시문' "
                       "표시에만 쓴다** (측정에는 영향 없음). top32 설정이면 TOP20+NEW12, "
                       "아니면 TOP20 을 학습 목록으로 본다. 2026-09-26 전에는 이 인자가 "
                       "없어서 run9 의 신규 12종이 전부 '학습 안 된 지시문' 으로 찍혔다."))
    p.add_argument("--prompt", default=None,
                   help=(
                       "지시문을 직접 지정한다. 생략하면 --skill 이 가리키는 스킬의 "
                       "지시문을 쓴다. **같은 자리·같은 화면에 문장만 바꿔 넣어** "
                       "정책이 말을 듣는지 눈으로 보는 실험에 쓴다."))
    p.add_argument("--judge-as", default=None,
                   help=(
                       "성공 판정·목표 물체 추적에만 쓸 지시문. 정책에는 --prompt 를 "
                       "주고 판정은 이 문장으로 한다. **바꿔 쓴 문장 시험**용이다 — "
                       "'shut the fridge' 는 판정기 정규식에 안 걸리므로 "
                       "'close the door of the fridge' 로 재야 한다. --chain 과 같이 못 쓴다."))
    # ── 이어서 지시하기 (28.6 가) ────────────────────────────────────────
    # 지금까지는 한 판에 문장 하나만 줬다. 실제 요리는 "문 열고 → 닫고 → 켜라"
    # 처럼 이어지는데, 그 전환을 재려면 한 에피소드 안에서 문장을 바꿔 줘야 한다.
    p.add_argument("--chain", default=None,
                   help=(
                       "지시문 여러 개를 '|' 로 이어 준다. 한 구간의 성공 판정이 "
                       "참이 되면(또는 예산을 다 쓰면) 다음 문장으로 넘어간다.\n"
                       "예: 'open the door of the oven|close the door of the "
                       "oven|turn on the oven'\n"
                       "⚠️ 앞 구간이 실패하면 뒤 구간의 조건이 **이미 참**인 채로 "
                       "시작할 수 있다 (문을 못 열었으면 '닫아라' 는 이미 달성). "
                       "그런 구간은 `무효(이미 참)` 로 찍고 성공에서 뺀다."))
    p.add_argument("--chain-steps", default=None,
                   help=("구간별 스텝 예산을 '|' 로 잇는다. 하나만 주면 전부 같게 쓴다. "
                         "생략하면 각 구간에 `스킬 길이 + --extra-steps` 를 준다."))
    p.add_argument("--chain-settle", type=int, default=150,
                   # ⚠️ argparse 는 도움말을 `help % params` 로 한 번 훑는다.
                   #    `%` 를 그냥 쓰면 `--help` 가 ValueError 로 터진다.
                   help=("구간 성공 뒤 몇 스텝 더 돌리고 다음 문장으로 넘기는가 "
                         "(기본 150 = 5초). 판정기는 문이 5%% 만 열려도 참이므로, "
                         "0 으로 두면 '살짝 열자마자 다음 지시' 가 된다."))
    p.add_argument("--chain-no-early", action="store_true",
                   help="성공해도 구간 예산을 끝까지 쓴다 (성공 뒤 행동을 보고 싶을 때)")
    p.add_argument("--save-actions", action="store_true",
                   help="정책이 낸 관절 명령을 npz 로 남긴다 (실행 간 수치 비교용)")
    p.add_argument("--replay-start", type=int, default=0,
                   help=(
                       "재생을 이 프레임부터 시작한다 (그 전은 건너뛴다).\n"
                       "팔 명령은 **절대 각도**라 앞부분을 건너뛰어도 마지막 자세는 "
                       "같아진다. 뒤쪽 100프레임만 재생하면 **이미 목표 지점에 선 "
                       "채로 팔이 제자리에서 올라간다** — 팔을 든 채 순간이동해서 "
                       "가구에 박히는 문제(2026-09-02 실측: 왼팔 최대 1.8 rad 틀어짐)를 "
                       "피할 수 있다."))
    p.add_argument("--place-after-replay", action="store_true",
                   help=(
                       "`--place-near` 순간이동을 **팔 재생이 끝난 뒤에** 한다.\n"
                       "실측(2026-09-02): 냉장고 0.8m 앞으로 먼저 옮긴 뒤 팔을 재생하면 "
                       "**오른쪽 팔꿈치가 냉장고에 막혀** 목표(-1.61 rad)에 못 가고 "
                       "+0.33 rad 로 반대편에 멈춘다. 데모는 멀리서 걸어오며 팔을 "
                       "들었으므로 그때는 걸릴 것이 없었다. 순서를 맞추면 해결된다."))
    p.add_argument("--replay-zero-base", action="store_true",
                   help=(
                       "데모 재생 구간에서 **베이스 명령만 0 으로 눌러** 팔·몸통만 "
                       "재생한다.\n"
                       "14.5 가 밝힌 대로 재생 실패의 원인은 '재생' 자체가 아니라 "
                       "**바퀴에 재생을 적용한 것**이었다. 팔·몸통·그리퍼는 절대 "
                       "위치 명령이라 어디서 시작하든 그 각도가 되므로 정확히 "
                       "재생된다. 베이스만 속도라 적분 오차가 쌓인다.\n"
                       "`--place-near` 로 자리를 잡고 이 옵션으로 팔 자세만 데모와 "
                       "맞추면, 정책이 **학습 데이터가 시작하는 바로 그 자세**에서 "
                       "넘겨받는다."))
    p.add_argument("--zero-base", action="store_true",
                   help=(
                       "정책이 낸 베이스 명령(액션 0~2)을 0 으로 눌러 로봇을 제자리에 "
                       "묶는다.\n"
                       "베이스만 **속도** 명령이고 나머지는 절대 위치라, 작은 잔차가 "
                       "위치 오차로 **적분**된다. 오프라인 진단에서 본 '가만히 있어야 할 때 "
                       "미세하게 떠는' 성질이 팔에서는 무해하지만 베이스에서는 몇 미터가 "
                       "된다. 조작 스킬만 보고 싶을 때 쓴다."))
    p.add_argument("--full-scene", action="store_true",
                   help=(
                       "장면 전체를 로딩한다. 기본(공식 eval 과 동일)은 태스크 관련 방만 "
                       "올리는 partial 모드라 kitchen/dining_room/living_room 외에는 벽까지 "
                       "없어서 뚫려 보인다 — 그건 정상이지만 눈으로 보기에는 불편하다. "
                       "느려지므로 확인용으로만 쓸 것."))
    return p.parse_args()


A = parse_args()
ZERO_BASE = A.zero_base
NULL_POLICY = A.null_policy
REPLAY_ZERO_BASE = A.replay_zero_base
REPLAY_START = A.replay_start

# cooking_config 는 openpi 의존성이 많아 behavior 환경에서 import 할 수 없다.
# 지시문 목록만 소스에서 문자열로 뽑아 온다(학습됨 표시에만 쓴다).
# ⚠️ 예전에는 TOP20 만 읽어서, 32종으로 배운 run9 의 신규 12종 설명 파일이
# 전부 "학습 안 된 지시문" 으로 찍혔다 (HANDOFF 40.18). 이제 --config 를 본다.
def _load_list(name):
    import ast, pathlib as _pl
    try:
        src = (_pl.Path(__file__).resolve().parent.parent /
               "configs/cooking_config.py").read_text()
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") == name for t in node.targets):
                return set(ast.literal_eval(node.value))
    except Exception:
        pass
    return set()


_TOP = _load_list("TOP20")
if A.config and "top32" in A.config:
    _TOP |= _load_list("NEW12")
print(f"[학습 지시문 목록] 설정 {A.config or '(안 줌 — TOP20 으로 본다)'} → {len(_TOP)}종 "
      "(설명 파일의 '학습된 지시문' 표시에만 쓴다)", flush=True)

import torch as th
from omegaconf import OmegaConf
from omnigibson.macros import gm

gm.HEADLESS = A.headless

from omnigibson.eval.evaluator import Evaluator, resolve_instance_ids
from omnigibson.eval.utils.eval_utils import seed_everything, DEFAULT_EVAL_SEED
from omnigibson.eval.utils.network_utils import WebsocketClientPolicy
import cv2


class ReplayThenPolicy:
    """앞부분은 데모 액션을 재생하고, 넘긴 시점부터 정책에게 묻는다.

    `Evaluator.step()` 이 `policy.forward(obs)` 만 부르므로, 이 객체를
    `evaluator.policy` 자리에 끼워 넣으면 재생·정책·영상녹화·채점이 한 루프에서
    돈다. 하네스를 고칠 필요가 없다.
    """

    def __init__(self, demo_actions, handover_at, instruction, client, cams, exec_steps,
                 demo_states=None, ref_frame=None):
        self.demo = demo_actions            # (N, 23) float32, 절대 관절값
        self.demo_states = demo_states      # (N, 61) 원본 관절 상태 (인계 자세 진단용)
        # ⚠️ 기준은 **스킬이 시작하는 프레임**이다. handover 를 쓰면
        # --replay-frames 0 일 때 데모 0번(리셋 자세)과 비교하게 되어 무의미하다.
        self.ref_frame = handover_at if ref_frame is None else ref_frame
        self.handover = handover_at
        self.instruction = instruction
        self.client = client
        self.cams = cams                    # {"image","left_wrist_image","right_wrist_image"} -> obs 키
        self.exec = exec_steps
        self.t = 0
        self.chunk, self.k = None, 0
        self.n_infer, self.lat = 0, []
        self.base_cmds = []          # 정책이 낸 베이스 명령을 모아 사후 분석한다
        self.all_acts = []           # 전체 관절 명령 (실행 간 비교용)
        # 🔴 **실제 관절값**도 같이 모은다 (2026-09-09). 명령만 있으면
        # "팔을 크게 움직이라고 시켰다" 와 "팔이 실제로 크게 움직였다" 를
        # 가를 수 없다. 문에 막혀 있으면 명령은 커도 움직임은 0 이다.
        self.all_states = []         # 전체 관절 실제값 (61차원 proprio)
        self._printed_keys = False
        self._reported = False
        # 🔑 **명령 묶기** (2026-09-10, 36.14). 특정 관절 명령을 범위 안으로
        # 잘라서 보낸다. 정책이 데모가 안 가 본 자세로 새는 것이 실패의 원인인지
        # **직접 쳐 보기** 위한 것이다 — 판을 더 돌려 상관을 다지는 것보다 세다.
        # 정책 구간에만 적용된다(재생 구간은 이 경로를 안 지난다).
        self.clamps = []             # [(차원, 하한, 상한)]
        self.clamp_hits = {}         # 차원 -> [잘린 스텝 수, 가장 많이 넘어간 양]
        # 🔑 **회전 보조** (2026-09-11, 37장). 정책이 몸을 돌리기 시작해 누적 회전이
        # 시작값을 넘으면, 목표값에 닿을 때까지 회전 명령(23차원의 2번)을 대신 준다.
        # 냉장고 갈림길에서 "얼마나 돌았나" 가 팔 선택을 가르는지 직접 치기 위한 것이다.
        self.yaw_assist = None       # (시작 rad, 목표 rad, 회전 명령 rad/s) — 셋 다 음수
        self.yaw_int = 0.0           # 정책 구간에서 실제 회전속도(proprio 2번)를 적분한 값
        self.assist_phase = "wait"   # wait -> on -> done
        self.assist_steps = 0
        self.assist_log = []         # [(정책 구간 초, 사건, 누적 회전 °)]

    @staticmethod
    def _rgb(x):
        """시뮬레이터 RGB(1080x1080, RGBA 일 수 있음) -> 224x224x3 uint8."""
        a = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
        if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
            a = np.transpose(a, (1, 2, 0))          # 채널 우선이면 뒤로 보낸다
        a = a[..., :3]
        if a.dtype != np.uint8:
            a = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8) if a.max() <= 1.0 \
                else a.astype(np.uint8)
        return cv2.resize(a, (224, 224), interpolation=cv2.INTER_AREA)

    def _report_handover(self, obs):
        """인계 순간의 관절 자세가 **학습 데이터가 시작하는 자세와 같은지** 찍는다.

        정책은 자기 세계좌표를 보지 못한다(`STATE_SELECT` 의 베이스 3개는 위치가
        아니라 **속도**다). 그래서 "정확한 위치에 놓여야 한다" 같은 제약은 없다.
        대신 **카메라에 보이는 그림**과 **관절 자세**만이 학습 분포와 맞아야 한다.
        여기서 뒤쪽 절반을 확인한다.
        """
        if self.demo_states is None:
            return
        st = obs.get(self.cams["proprio"])
        if st is None:
            return
        st = np.asarray(st.numpy() if hasattr(st, "numpy") else st, dtype=np.float64).reshape(-1)
        ref = np.asarray(self.demo_states[self.ref_frame], dtype=np.float64)
        groups = (("토르소", list(range(53, 57))), ("왼팔", list(range(3, 10))),
                  ("오른팔", list(range(28, 35))), ("그리퍼", [24, 25, 49, 50]))
        print(f"\n[인계 자세] 지금 로봇 vs 데모 프레임 {self.ref_frame}"
              f"(스킬 시작) 의 자세 (라디안)")
        for nm, ix in groups:
            d = np.abs(st[ix] - ref[ix])
            print(f"   {nm:5s} 평균차 {d.mean():.3f}  최대차 {d.max():.3f}")
        for nm, ix in (("오른팔", list(range(28, 35))), ("왼팔", list(range(3, 10)))):
            print(f"   {nm} 지금 {np.round(st[ix], 2)}")
            print(f"   {nm} 데모 {np.round(ref[ix], 2)}")
        print(flush=True)

    def forward(self, obs, *args, **kwargs):
        if not self._printed_keys:
            print("\n[obs 키]", *sorted(obs.keys()), sep="\n  ")
            self._printed_keys = True

        if self.t < self.handover:                  # ── 데모 재생 구간
            a = self.demo[max(self.t, REPLAY_START)].copy()
            if REPLAY_ZERO_BASE:
                a[:3] = 0.0        # 바퀴만 빼고 재생 (팔·몸통은 절대 위치라 정확하다)
            self.t += 1
            if self.t == self.handover:
                self._hold = a.copy()          # 대조군이 굳힐 자세
                if not self._reported:
                    self._reported = True
                    self._report_handover(obs)
            return th.tensor(a, dtype=th.float32)

        if not self._reported:                          # 인계 자세는 한 번만 찍는다
            self._reported = True
            self._report_handover(obs)

        try:                                        # 실제 관절값을 매 스텝 남긴다
            _st = obs.get(self.cams["proprio"])
            if _st is not None:
                _st = _st.numpy() if hasattr(_st, "numpy") else np.asarray(_st)
                self.all_states.append(
                    np.asarray(_st, dtype=np.float32).reshape(-1).copy())
        except Exception:
            pass

        if NULL_POLICY:                             # ── 🔬 대조군: 아무것도 안 한다
            a = getattr(self, "_hold", None)
            a = np.zeros(23, dtype=np.float32) if a is None else a.copy()
            a[:3] = 0.0                             # 바퀴 정지
            self.base_cmds.append(a[:3].copy())
            self.all_acts.append(a.copy())
            self.t += 1
            if self.t % 300 == 0:
                print(f"  [대조군] 자세 유지 {self.t - self.handover}스텝", flush=True)
            return th.tensor(a, dtype=th.float32)

        if self.chunk is None or self.k >= self.exec:   # ── 정책 구간: 재예측
            state = obs[self.cams["proprio"]]
            state = state.numpy() if hasattr(state, "numpy") else np.asarray(state)
            payload = {
                "image": self._rgb(obs[self.cams["head"]]),
                "left_wrist_image": self._rgb(obs[self.cams["left_wrist"]]),
                "right_wrist_image": self._rgb(obs[self.cams["right_wrist"]]),
                "state": np.asarray(state, dtype=np.float32).reshape(-1),
                "prompt": self.instruction,
            }
            t0 = time.time()
            self.chunk = np.asarray(self.client.infer(payload)["actions"], dtype=np.float32)
            self.lat.append(time.time() - t0)
            self.n_infer += 1
            self.k = 0
            print(f"  추론 {self.n_infer:3d}회  {self.lat[-1]:5.1f}초  "
                  f"(스텝 {self.t})", flush=True)

        a = self.chunk[self.k].copy()
        # 묶기는 **기록보다 먼저** 한다 — npz 의 actions 가 "실제로 보낸 것" 이라야
        # 나중에 묶임이 걸렸는지를 확인할 수 있다. 정책이 원래 뭘 원했는지는
        # clamp_hits 의 "넘어간 양" 으로 남긴다.
        for d, lo, hi in self.clamps:
            raw = float(a[d])
            if raw < lo or raw > hi:
                over = max(lo - raw, raw - hi)
                h = self.clamp_hits.setdefault(d, [0, 0.0])
                h[0] += 1
                h[1] = max(h[1], over)
                a[d] = min(max(raw, lo), hi)
        if self.yaw_assist is not None:
            st = obs[self.cams["proprio"]]
            st = st.numpy() if hasattr(st, "numpy") else np.asarray(st)
            # 실제 회전속도를 적분한다. 명령 a[2] 와 실제는 1:1 로 맞는다(데모 상관 0.955).
            self.yaw_int += float(np.asarray(st, dtype=np.float64).reshape(-1)[2]) / 30.0
            y0, y1, rate = self.yaw_assist
            now = (self.t - self.handover) / 30.0
            if self.assist_phase == "wait" and self.yaw_int <= y0:
                self.assist_phase = "on"
                self.assist_log.append((now, "시작", float(np.degrees(self.yaw_int))))
                print(f"  [회전 보조] 켬 @정책 {now:.1f}초  누적 회전 "
                      f"{np.degrees(self.yaw_int):+.1f}°", flush=True)
            if self.assist_phase == "on":
                self.assist_steps += 1
                if self.yaw_int <= y1 or self.assist_steps > 150:
                    why = "목표 도달" if self.yaw_int <= y1 else "5초 넘도록 못 닿음(막힘?)"
                    self.assist_phase = "done"
                    self.assist_log.append((now, why, float(np.degrees(self.yaw_int))))
                    print(f"  [회전 보조] 끔 @정책 {now:.1f}초  누적 회전 "
                          f"{np.degrees(self.yaw_int):+.1f}° — {why}", flush=True)
                else:
                    a[2] = rate
        self.base_cmds.append(a[:3].copy())
        self.all_acts.append(a.copy())
        if ZERO_BASE:
            a[:3] = 0.0
        self.k += 1
        self.t += 1
        return th.tensor(a, dtype=th.float32)

    def reset(self):
        pass


class ChunkClient:
    """청크를 통째로 받는 얇은 클라이언트.

    BEHAVIOR 의 `WebsocketClientPolicy.act()` 는 응답에서 `"action"` 하나만
    꺼내지만, 우리 서버는 32스텝짜리 `"actions"` 를 준다. receding horizon 을
    호출자가 제어해야 하므로 응답 전체가 필요해서 얇게 감싼다.
    """

    def __init__(self, host, port):
        self._c = WebsocketClientPolicy(host=host, port=port, allow_reconnect=True)

    def infer(self, obs):
        if self._c._ws is None:
            self._c._ws, self._c._server_metadata = self._c._wait_for_server()
        # ⚠️ `unpack_data` 는 msgpack 의 object_hook 이지 디코더가 아니다.
        #    바이트를 그대로 넘기면 "byte indices must be integers" 로 죽는다.
        #    BEHAVIOR 의 클라이언트가 쓰는 `unpackb`(훅이 이미 물려 있다)를 쓴다.
        from omnigibson.eval.utils.network_utils import unpackb
        self._c._ws.send(self._c._packer.pack(obs))
        resp = self._c._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"정책 서버 오류:\n{resp}")
        return unpackb(resp)


def describe_scene(ev):
    """장면에 무엇이 올라와 있고 로봇이 어디에 있는지 찍어 둔다.

    영상만 봐서는 "에셋이 덜 로딩된 것인지, 로봇이 딴 데를 보고 있는 것인지"를
    가를 수 없다. 물체 수와 좌표를 먼저 남겨 두면 그 둘을 구분할 수 있다.
    """
    try:
        objs = list(ev.env.scene.objects)
        rp = ev.robot.get_position_orientation()[0]
        rp = rp.numpy() if hasattr(rp, "numpy") else np.asarray(rp)
        print(f"\n[장면] 물체 {len(objs)}개 / 로봇 위치 "
              f"({rp[0]:.2f}, {rp[1]:.2f}, {rp[2]:.2f})")
        keys = ("fridge", "refrigerator", "cutting_board", "parer", "beet",
                "bell_pepper", "countertop")
        for o in objs:
            n = o.name.lower()
            if any(k in n for k in keys):
                p = o.get_position_orientation()[0]
                p = p.numpy() if hasattr(p, "numpy") else np.asarray(p)
                d = float(np.linalg.norm(p[:2] - rp[:2]))
                print(f"   {o.name:<44} ({p[0]:7.2f},{p[1]:7.2f},{p[2]:6.2f})  "
                      f"로봇과 {d:5.2f}m")
        print()
    except Exception as e:
        print(f"[장면] 조사 실패: {type(e).__name__}: {e}\n")



def make_success_check(ev, instruction):
    """지시문에서 목표 물체와 성공 조건을 뽑아 판정 함수를 만든다.

    하네스의 `task.success` 는 **태스크 전체**의 목표(BDDL)를 본다. 우리는 스킬
    하나만 돌리므로 그것으로는 영원히 False 다. 그래서 스킬 수준의 판정을 따로
    만든다 — 문을 열라고 했으면 그 문이 열렸는지, 집으라고 했으면 그 물체를
    쥐었는지를 시뮬레이터 상태로 직접 본다.

    ⚠️ **인계 시점에 이미 조건이 참이면 판정을 끈다.** 예를 들어 도마를 집는
    스킬은 이미 오른손에 칼을 쥔 채 시작하므로, 물체를 지정하지 않고 "뭔가
    쥐었나" 로 보면 첫 스텝에 성공으로 끝나 버린다.

    돌려주는 것: (판정함수, 설명) 또는 (None, 못 만든 이유)
    """
    import re
    from omnigibson.object_states import Open, ToggledOn, OnTop, Inside
    from omnigibson.controllers.controller_base import IsGraspingState

    ins = instruction.lower().strip()

    def find(noun, half=False):
        """이름에 명사가 든 물체를 찾는다. 'carving knife' -> carving_knife_*

        ⚠️ `half=True` 면 **반쪽 물체를 먼저** 찾는다. 2026-09-07 이전에는
        지시문에서 `half` 를 떼고 찾아서, `pick up the half brussels sprouts`
        가 **통째 방울양배추를 쥐어도 성공**으로 나왔다 (HANDOFF 25.2).
        """
        base = noun.replace(" ", "_")
        cands = ([f"half_{base}", f"half__{base}"] if half else []) + [base, noun.split()[-1]]
        for cand in cands:
            hits = [o for o in ev.env.scene.objects if cand in o.name.lower()]
            if hits:
                return hits[0]
        return None

    m = re.match(r"(open|close) the (?:door|lid) of the (.+)", ins)
    if m:
        want = m.group(1) == "open"
        obj = find(m.group(2))
        if obj is None:
            return None, f"물체를 못 찾음: {m.group(2)!r}"
        if Open not in obj.states:
            return None, f"{obj.name} 에 Open 상태가 없다"
        return (lambda: bool(obj.states[Open].get_value()) == want,
                f"{obj.name} 의 Open == {want}")

    m = re.match(r"turn on the (.+)", ins)
    if m:
        obj = find(m.group(1))
        if obj is None:
            return None, f"물체를 못 찾음: {m.group(1)!r}"
        if ToggledOn not in obj.states:
            return None, f"{obj.name} 에 ToggledOn 상태가 없다"
        return (lambda: bool(obj.states[ToggledOn].get_value()),
                f"{obj.name} 의 ToggledOn == True")

    m = re.match(r"pick up the (.+?)(?: from the .+)?$", ins)
    if m:
        raw = m.group(1).strip()
        is_half = raw.startswith("half ")
        noun = re.sub(r"^(other|half) ", "", raw).strip()
        obj = find(noun, half=is_half)
        if obj is None:
            return None, f"물체를 못 찾음: {noun!r}"
        def grasped():
            for arm in ev.robot.arm_names:
                if ev.robot.is_grasping(arm=arm, candidate_obj=obj) == IsGraspingState.TRUE:
                    return True
            return False
        return grasped, f"{obj.name} 를 쥐었는가" + ("  (반쪽 물체를 지목했다)" if is_half else "")

    # ── put the X (on|in) the Y  ────────────────────────────────────────
    # 25.2 의 `half_put`(바닥에 떨어뜨림)처럼 "들긴 드는데 놓는 자리가 틀린" 것을
    # 재려면 놓인 자리를 봐야 한다. OnTop/Inside 로 직접 본다.
    m = re.match(r"put the (.+?) (on|in|into|onto|back on) the (.+?)$", ins)
    if m:
        raw_a, prep, raw_b = m.group(1).strip(), m.group(2), m.group(3).strip()
        raw_b = re.split(r"\s+(?:to the|next to|on the)\s", raw_b)[0].strip()
        a_half = raw_a.startswith("half ")
        na = re.sub(r"^(other|half) ", "", raw_a).strip()
        oa, ob = find(na, half=a_half), find(raw_b)
        if oa is None or ob is None:
            return None, f"물체를 못 찾음: {na!r} 또는 {raw_b!r}"
        st = Inside if prep in ("in", "into") else OnTop
        if st not in oa.states:
            return None, f"{oa.name} 에 {st.__name__} 상태가 없다"
        return (lambda: bool(oa.states[st].get_value(ob)),
                f"{oa.name} 가 {ob.name} 의 {'안에' if st is Inside else '위에'} 있는가")

    # ── chop the X with the Y  ──────────────────────────────────────────
    # OmniGibson 에는 Sliced 상태가 없다. 자르면 **원래 물체가 장면에서 사라지고**
    # 반쪽들이 새로 생긴다. 그래서 "그 물체가 없어졌는가" 로 본다.
    m = re.match(r"chop the (.+?) with the (.+?)$", ins)
    if m:
        raw_a = m.group(1).strip()
        a_half = raw_a.startswith("half ")
        na = re.sub(r"^(other|half) ", "", raw_a).strip()
        oa = find(na, half=a_half)
        if oa is None:
            return None, f"물체를 못 찾음: {na!r}"
        target = oa.name
        return (lambda: all(o.name != target for o in ev.env.scene.objects),
                f"{target} 가 잘려 장면에서 사라졌는가")

    return None, f"이 지시문 형태는 아직 판정기가 없다: {ins!r}"


def open_fraction(obj):
    """문이 **얼마나** 열렸는지 0~1 로 돌려준다. 못 재면 None.

    ## 왜 필요한가 (2026-09-09, HANDOFF 32.2)

    `Open` 은 불리언인데 그 문턱이 **관절 범위의 5%** 다
    (`JOINT_THRESHOLD_BY_TYPE = 0.05`). 90도짜리 문이면 **4.5도만 움직여도
    "열림"** 이다. 사용자가 영상에서 "아주 살짝만 열고 만다" 를 봤는데
    판정기는 그것을 성공이라고 불렀다.

    5% 와 90% 를 같은 성공으로 적으면 **더 돌려서 나아졌는지를 영원히 못 잰다.**
    그래서 비율을 직접 읽는다. 관절이 여럿이면 가장 많이 열린 것을 쓴다
    (`_get_value` 도 `any(joint_openness)` 로 하나만 열려도 열림으로 본다).
    """
    from omnigibson.object_states import Open
    from omnigibson.object_states.open_state import _compute_joint_threshold
    if Open not in obj.states:
        return None
    try:
        _both, joints, dirs = obj.states[Open].relevant_joints_info
        if not joints:
            return None
        fr = []
        for j, d in zip(joints, dirs):
            _thr, open_end, closed_end = _compute_joint_threshold(j, d)
            rng = float(open_end - closed_end)
            if abs(rng) < 1e-9:
                continue
            pos = float(j.get_state()[0])
            fr.append((pos - float(closed_end)) / rng)
        return max(fr) if fr else None
    except Exception:
        return None


def set_open_fraction(obj, frac):
    """목표 물체의 문(뚜껑)을 **지정한 비율만큼 직접 열어 놓는다.**

    ## 왜 필요한가 (2026-09-10, 36.9)

    표준 인계로 냉장고를 시키면 정책이 **왼팔을 아예 들지 않는다** — 6판 전부
    왼팔꿈치(관절 10)가 -0.14 를 못 넘는다. 데모는 25초에 -0.75 로 꺾고 왼팔로
    문을 밀어젖힌다. 정책은 문을 9초에 14% 까지 열었다가 손을 놓고, 그 뒤
    문은 도로 닫힌다.

    그래서 **닭과 달걀** 가설이 나왔다 — 문을 더 열려면 왼팔이 필요한데,
    왼팔을 부르는 방아쇠가 "문이 크게 열린 화면" 이라면 영영 도달할 수 없다.

    **이 함수가 그 가설을 가른다.** 문을 미리 30% 열어 놓고 시작해서
      왼팔이 나온다  -> 방아쇠는 **문 열림 정도**다 (처방: 정밀도)
      안 나온다      -> 방아쇠는 다른 것(자세·시간)이다 (처방: 구조)

    ⚠️ 관절을 즉시 세팅하는 것이라 접촉 물리를 거치지 않는다. 인계 시점에
    로봇 팔이 문 근처에 있으면 문이 팔을 뚫고 열릴 수 있다. 표준 인계는 팔이
    내려가 있어 안전하지만, 다른 인계 지점에서는 영상으로 확인할 것.

    돌려주는 것: [(관절이름, 맞춘 값), ...] 또는 None (못 하면)
    """
    from omnigibson.object_states import Open
    from omnigibson.object_states.open_state import _compute_joint_threshold
    if Open not in obj.states:
        return None
    try:
        _both, joints, dirs = obj.states[Open].relevant_joints_info
    except Exception:
        return None
    if not joints:
        return None
    done = []
    # 🔴 **첫 관절 하나만** 연다. 냉장고처럼 문이 여럿인 물체에서 전부 열면
    #    데모에 없는 무대가 된다. `open_fraction` 은 가장 많이 열린 것을 쓰므로
    #    하나만 열어도 비율은 원하는 값이 나온다.
    for j, d in zip(joints[:1], dirs[:1]):
        _thr, open_end, closed_end = _compute_joint_threshold(j, d)
        rng = float(open_end - closed_end)
        if abs(rng) < 1e-9:
            continue
        target = float(closed_end) + float(frac) * rng
        j.set_pos(target)                      # drive=False -> 즉시 반영
        done.append((getattr(j, "joint_name", str(j)), round(target, 4)))
    return done


def find_target_obj(ev, instruction):
    """지시문이 가리키는 물체를 장면에서 찾는다. 못 찾으면 None."""
    ins = instruction.lower().strip()
    noun = None
    for pat in (r"(?:open|close) the (?:door|lid) of the (.+)",
                r"turn on the (.+)",
                r"pick up the (.+?)(?: from the .+)?$",
                r"put the (.+?) (?:on|in|into|onto|back on) the .+?$",
                r"chop the (.+?) with the .+?$"):
        m = re.match(pat, ins)
        if m:
            noun = re.sub(r"^(other|half) ", "", m.group(1).strip())
            break
    if noun is None:
        return None
    base = noun.replace(" ", "_")
    return next((o for o in ev.env.scene.objects
                 if base in o.name.lower() or noun.split()[-1] in o.name.lower()), None)


def report_target_states(ev, instruction, label="최종 상태"):
    """지시문이 가리킨 물체의 **상태를 전부** 찍는다 — 판정에 쓴 것 하나만이 아니라.

    ## 왜 필요한가 (2026-09-08, HANDOFF 21.4)

    21.4 에서 정책에 `open the door of the microwave` 를 줬더니 **문을 여는 대신
    전원을 켰다.** 그런데 판정기는 `Open` 만 보므로 로그에는 그냥 "실패" 로
    남는다. 무엇을 **대신** 했는지는 기록에 없었고, 사람이 영상에서 표시등이
    빨강->초록으로 바뀌는 것을 보고서야 알아냈다.

    "못 했다" 와 "다른 걸 했다" 는 완전히 다른 진단이다. 앞은 실력 문제고 뒤는
    **문장을 못 고르는 문제**다. 그 둘을 영상 색깔로 가르게 두면 안 된다.

    그래서 목표 물체가 가진 상태를 전부 찍는다. `open the door of X` 를 주고
    `ToggledOn=True` 가 찍히면 21.4 가 그대로 재현된 것이다.
    """
    import re
    from omnigibson.object_states import Open, ToggledOn
    from omnigibson.controllers.controller_base import IsGraspingState

    obj = find_target_obj(ev, instruction)
    if obj is None:
        return {}

    vals, bits = {}, []
    for st, nm in ((Open, "Open"), (ToggledOn, "ToggledOn")):
        if st in obj.states:
            try:
                v = bool(obj.states[st].get_value())
                vals[nm] = v
                bits.append(f"{nm}={v}")
                if nm == "Open":
                    fr = open_fraction(obj)
                    if fr is not None:
                        vals["열린비율"] = round(fr, 3)
                        bits.append(f"열린비율={fr:.1%}")
            except Exception:
                bits.append(f"{nm}=?")
    try:
        held = [a for a in ev.robot.arm_names
                if ev.robot.is_grasping(arm=a, candidate_obj=obj) == IsGraspingState.TRUE]
        vals["쥔손"] = ",".join(held) if held else "없음"
        bits.append("쥔손=" + vals["쥔손"])
    except Exception:
        pass
    if bits:
        print(f"[목표 물체 {label}] {obj.name}: {'  '.join(bits)}", flush=True)
    return vals


def check_handover_state(ev, demo, handover, nouns):
    """🔴 인계 시점에 **물체 상태가 데모대로 재현됐는지** 본다.

    ## 왜 필요한가 (2026-09-07, HANDOFF 26장)

    지금까지 무대가 정상인지 보는 유일한 장치가 **관절 각도 비교**였다.
    그런데 그것이 초록불을 켜면서 물체 상태는 어긋나 있는 경우가 확인됐다:

    - `oven_close` — 재생에 '오븐 문 열기' 가 들어 있는데 문이 안 열려서,
      **이미 닫힌 문을 닫으라고** 시켰다.
    - `board_pick` — 관절이 소수점 셋째 자리까지 일치(0.002 라디안)했는데
      **오른손에 칼이 없었다.** 사용자가 영상을 보고서야 드러났다.

    로봇의 관절은 절대 위치라 정확히 재현되지만, **물체는 접촉 물리에 의존해서
    어긋난다.** 이 함수는 그 어긋남을 실행 중에 잡아 경고로 남긴다.

    돌려주는 것: 사람이 읽을 경고 문자열 목록 (없으면 빈 목록)
    """
    from omnigibson.controllers.controller_base import IsGraspingState
    warn, held = [], {}
    if handover <= 0:
        return warn, held
    a = np.asarray(demo[handover - 1], dtype=np.float64)
    cmd = {"left": a[14], "right": a[22]}       # -1 쥠 / +1 열림
    cands = []
    for n in nouns:
        o = next((o for o in ev.env.scene.objects if n in o.name.lower()), None)
        if o is not None:
            cands.append(o)
    for arm in ev.robot.arm_names:
        key = "left" if "left" in arm else "right"
        want_closed = cmd.get(key, 1.0) < 0
        try:
            state = ev.robot.is_grasping(arm=arm)
            grasping = (state == IsGraspingState.TRUE)
        except Exception as e:
            warn.append(f"{key} 손 파지 상태를 못 읽었다: {type(e).__name__}")
            continue
        what = None
        if grasping:
            for o in cands:
                try:
                    if ev.robot.is_grasping(arm=arm, candidate_obj=o) == IsGraspingState.TRUE:
                        what = o.name; break
                except Exception:
                    pass
            held[key] = what or "(무엇인지 특정 못 함)"
        if want_closed and not grasping:
            warn.append(
                f"🔴 {key} 손: 데모는 이 시점에 **무언가 쥔 채**였는데 "
                f"실제로는 **아무것도 안 쥐고 있다** — 재생이 물체를 놓쳤다. "
                f"이 실행의 결과는 모델의 성과로 읽으면 안 된다.")
        if (not want_closed) and grasping:
            warn.append(
                f"⚠️ {key} 손: 데모는 이 시점에 손을 폈는데 실제로는 "
                f"{held.get(key)} 를 쥐고 있다 — 재생이 어긋났다.")
    return warn, held


def expected_script(demo, bounds, skill, fps, held=None, prompt=None):
    """**모델이 인계받은 뒤 무엇을 해야 하는지**를 시간 순서로 적는다.

    ## 왜 필요한가 (2026-09-07)

    사용자가 앞으로 영상을 직접 꼼꼼히 보고 판정하기로 했다. 그런데 영상만
    봐서는 **무엇이 정답인지**를 알 수 없다 — 예를 들어 `chop the brussels
    sprouts` 에서 왼손에 든 것을 썰어야 하는지 도마에 있는 다른 것을 썰어야
    하는지는 데모를 봐야 안다. 그래서 그 정답을 폴더마다 적어 둔다.

    데모 재생 구간은 적지 않는다. **인계 시점(0.0초)부터**만 쓴다.
    시각은 사람이 그 스킬을 한 속도이므로 **모델이 그대로 맞출 필요는 없다** —
    순서와 대상이 맞는지를 보는 기준이다.
    """
    LG, RG = 14, 22
    by = {x["skill_idx"]: x for x in bounds}
    cur = by.get(skill)
    if cur is None:
        return ["(데모에서 이 스킬을 못 찾아 기대 동작을 적지 못했다)"]
    order = sorted(by)
    pos = order.index(skill)
    prev = by[order[pos - 1]] if pos > 0 else None
    nxt = by[order[pos + 1]] if pos + 1 < len(order) else None
    t0, n = cur["t0"], cur["n"]
    seg = np.asarray(demo[t0:t0 + n], dtype=np.float64)
    ins = prompt or cur["instruction"]

    L = ["## 🎯 모델이 해야 하는 일 (인계 시점 = 0.0초 기준)", "",
         f"**지시문**: `{ins}`", ""]

    # ── 시작 상태 ────────────────────────────────────────────────────────
    L += ["### 시작할 때 손 상태", ""]
    for nm, ch, key in (("왼손", LG, "left"), ("오른손", RG, "right")):
        closed = seg[0, ch] < 0 if len(seg) else False
        real = (held or {}).get(key)
        if closed:
            L.append(f"- **{nm}**: 무언가 **쥔 채로** 시작해야 한다"
                     + (f" — 실제로 쥔 것: `{real}`" if real else
                        " — ⚠️ 실제로 무엇을 쥐었는지 확인 못 함"))
        else:
            L.append(f"- **{nm}**: **편 채로** 시작"
                     + (f" — ⚠️ 그런데 실제로는 `{real}` 를 쥐고 있었다" if real else ""))
    L.append("")

    # ── 문맥: 직전에 무엇을 했나, 그래서 대상이 무엇인가 ──────────────────
    if prev is not None:
        L += ["### 직전까지 사람이 해 준 것 (재생 구간의 마지막)", "",
              f"- `{prev['instruction']}`", ""]
        pm = re.match(r"pick up the (.+?)(?: from the .+)?$", prev["instruction"].lower())
        cm = re.match(r"(chop|put) the (.+?) (?:with|on|in|into|onto|back on) the", ins.lower())
        if pm and cm and pm.group(1).strip() == cm.group(2).strip():
            L += [f"> 🔴 **대상은 바로 그것이다.** 직전에 집은 `{pm.group(1).strip()}` 을",
                  f"> 지금 {'썰어야' if cm.group(1)=='chop' else '놓아야'} 한다 — "
                  f"도마에 있는 **다른** 것이 아니다.", ""]

    # ── 사람이 한 순서 ───────────────────────────────────────────────────
    L += ["### 사람이 한 순서 (데모, 인계 시점부터의 초)", ""]
    ev_lines = []
    for nm, ch in (("왼손", LG), ("오른손", RG)):
        c = seg[:, ch] < 0
        for i in range(1, len(c)):
            if c[i] != c[i - 1]:
                ev_lines.append((i / fps, f"**{nm}**을 {'쥔다' if c[i] else '편다'}"))
    if not ev_lines:
        L.append(f"- 이 구간에서 사람은 **그리퍼를 한 번도 바꾸지 않는다** "
                 f"(양손 상태를 유지한 채 {n / fps:.1f}초 동안 팔만 움직인다).")
    else:
        for t, what in sorted(ev_lines):
            L.append(f"- **{t:.1f}초** — {what}")
    L += ["", f"- 사람은 이 스킬을 **{n / fps:.1f}초**에 끝낸다. "
              f"모델이 더 오래 걸리는 것 자체는 실패가 아니다 — **순서와 대상**을 볼 것.", ""]

    if nxt is not None:
        L += ["### 이 다음에 와야 할 것 (이번 판정에는 안 들어간다)", "",
              f"- `{nxt['instruction']}`",
              "  이 동작까지 이어서 하면 잘한 것이지만, 이번 성공 판정 대상은 아니다.", ""]

    # ── 사람이 영상에서 확인할 것 ─────────────────────────────────────────
    L += ["### 영상에서 확인해 주실 것", "",
          "- [ ] **대상이 맞는가** — 지시문이 가리키는 바로 그 물체를 다뤘는가",
          "- [ ] **손이 맞는가** — 위 '시작할 때 손 상태' 와 '사람이 한 순서' 대로인가",
          "- [ ] **쥐었는가** — 그리퍼만 오므린 것이 아니라 물체가 실제로 딸려 오는가",
          "- [ ] **놓는 자리가 맞는가** — 놓기 동작이면, 목표 위에 놓였는가 (바닥에 떨어뜨리지 않았는가)",
          "- [ ] **끝나고 손을 폈는가** — 다음 동작을 하려면 펴야 한다",
          "- [ ] **끊긴 것은 아닌가** — 영상이 끝날 때 아직 무언가 하는 중이었는가", ""]
    return L


def write_guide(out_dir, video, task, inst, demo_path, skill, instruction, trained,
                handover, n_steps, fps, zero_base, success, note="",
                expected=None, warnings=None, judge=None, skill_ok=None,
                segments=None, notes=None):
    """이 폴더의 영상이 **어디까지 사람이고 어디부터 정책인지** 적어 둔다.

    영상은 맨 처음부터 녹화되므로, 재생 구간이 있으면 앞부분이 사람 데모다.
    이걸 안 적어 두면 재생 구간의 행동을 정책의 성과로 오독하게 된다 —
    2026-09-03 에 실제로 그런 혼동이 두 번 있었고, 그래서 자동으로 남긴다.
    영상 1프레임 = 시뮬레이터 1스텝이라 초 = 프레임 / fps 로 환산된다.
    """
    total = handover + n_steps
    hs, ts = handover / fps, total / fps
    lines = [
        f"# {os.path.basename(out_dir)} — 이 영상 읽는 법",
        "",
    ]
    # 🔴 무대가 어긋난 실행은 **맨 위**에 딱지를 박는다 (27.3 ①). 자세한 내용은
    # 아래 별도 절에 있지만, 사용자는 결과 폴더를 여러 개 훑으며 영상을 보므로
    # 21번째 줄까지 내려가야 보이면 놓친다.
    if warnings:
        lines += [
            "> 🔴 **이 실행은 무대가 어긋났다 — 결과를 모델의 성과·실패로 읽으면 안 된다.**",
            "> 아래 '무대가 어긋났다' 절을 먼저 볼 것.",
            "",
        ]
    elif notes:
        # 무대는 정상인데 읽을 때 조심할 것(모델이 한 부수 변화 등). 예전에는 이것도
        # 위의 🔴 딱지를 받아서, 놓기에 성공한 판까지 "무대가 어긋났다" 로 읽혔다 (40.16).
        lines += [
            "> ⚠️ **무대는 정상이다. 다만 결과를 읽을 때 주의할 점이 있다** — 아래 '주의' 절.",
            "",
        ]
    # 🔬 대조군은 정책에 **아예 묻지 않은** 실행이다. 이 딱지가 없으면 아래
    # "모델이 해야 하는 일" 을 읽고 모델이 운전한 영상으로 오독하게 된다.
    if NULL_POLICY:
        lines += [
            "> 🔬 **이 실행은 대조군이다 — 정책에 아예 묻지 않았다.**",
            "> 인계 자세를 그대로 굳힌 채(바퀴 0) 같은 무대를 돌리고 성공 판정만",
            "> 똑같이 걸었다. **여기서도 성공하면 그 무대의 성공은 정책의 성과가",
            "> 아니다.** 아래 '모델이 해야 하는 일' 은 같은 무대의 정책 실행을",
            "> 읽을 때 쓰라고 남긴 것이지, 이 영상 속에서 일어난 일이 아니다.",
            "",
        ]
    lines += [
        f"- **태스크**: `{task}` / **장면 인스턴스**: `{inst}`",
        f"- **데모 파일**: `{demo_path}` 의 스킬 {skill}",
        f"- **정책에 준 지시문**: `{instruction}`  ({'학습된 지시문' if trained else '학습 안 된 지시문'})",
        f"- **영상**: `{os.path.basename(video) if video else '없음'}`  ({fps}fps, 총 {total:,}프레임 = {ts:.1f}초)",
        "",
        "## 어디까지가 사람이고 어디부터가 모델인가",
        "",
        "| 구간 | 프레임 | 시간 | 무엇 |",
        "|---|---|---|---|",
    ]
    if handover > 0:
        lines.append(f"| 데모 재생 | 0 ~ {handover-1:,} | 0.0초 ~ **{hs:.1f}초** | "
                     f"사람이 녹화한 액션을 그대로 재생 (모델 아님) |")
        lines.append(
            f"| **정책** | {handover:,} ~ {total-1:,} | **{hs:.1f}초 ~ {ts:.1f}초** | "
            + (f"**🔬 대조군 — 정책에 안 물었다. 자세를 굳힌 채 {n_steps:,}스텝** |"
               if NULL_POLICY else
               f"**학습한 모델이 운전 ({n_steps:,}스텝)** |"))
    else:
        lines.append(f"| **정책** | 0 ~ {total-1:,} | **0.0초 ~ {ts:.1f}초** | "
                     f"**처음부터 끝까지 학습한 모델 (재생 없음)** |")
    # 🔗 연쇄 실행은 **한 영상 안에서 지시문이 바뀐다.** 언제 무엇을 시켰는지
    #    적어 두지 않으면 영상을 볼 수가 없다 — 27.3 ⑤ 와 같은 이유다.
    if segments and len(segments) > 1:
        lines += [
            "",
            "## 🔗 이 영상은 **지시문이 도중에 바뀐다** (이어서 지시하기)",
            "",
            "정책 구간 안에서 아래 순서로 문장을 갈아 줬다. 한 구간의 성공 판정이",
            "참이 되면(또는 예산을 다 쓰면) 다음 문장으로 넘어간다.",
            "",
            "| # | 지시문 | 영상 시간 | 결과 | 이 구간에서 문이 얼마나 | 판정 기준 |",
            "|---|---|---|---|---|---|",
        ]
        for r in segments:
            mark = ("🔴 **무효 — 시작부터 조건이 참이었다**" if r["pre_true"]
                    else "✅ 해냈다" if r["ok"] else "❌ 못 했다")
            if r["ok"] and r["ok_sec"] is not None:
                mark += f" ({r['ok_sec']}초 만에)"
            _s0 = r["t0"].get("열린비율")
            opn = ("—" if r["open_max"] is None else
                   f"시작 {'?' if _s0 is None else format(_s0, '.0%')} → "
                   f"최대 {r['open_max']:.0%} → 끝 {r['open_end']:.0%}")
            lines.append(
                f"| {r['seg']} | `{r['instruction']}` | "
                f"**{r['from']/fps:.1f}초 ~ {r['to']/fps:.1f}초** | {mark} | "
                f"{opn} | {r['judge']} |")
        if any(r["pre_true"] for r in segments):
            lines += [
                "",
                "> 🔴 **`무효` 구간을 성공으로 세면 안 된다.** 앞 구간이 실패해서 뒤 구간의",
                "> 과제가 이미 달성된 상태였을 수 있다 — 문을 못 열었으면 `닫아라` 는",
                "> 시작부터 참이다. 그래도 그 구간을 계속 돌리는 이유는, **이미 다 된 일을",
                "> 시켰을 때 정책이 무엇을 하는지**(가만히 있는지, 엉뚱한 것을 건드리는지)가",
                "> 그 자체로 볼 값어치가 있기 때문이다. 영상에서 그 부분을 봐 달라.",
            ]
        lines += [
            "",
            "> ⚠️ **성공 판정은 `한 번이라도 됐나` 이지 `끝에 그렇나` 가 아니다.** 문은",
            "> 관절 가동범위의 **5%만 넘으면** `열림`이다 (32.2). 위 표의 `해냈다` 는",
            "> 그 순간 조건이 찼다는 뜻이고, 얼마나 열렸는지는 로그의 `열림 최대/끝` 을 볼 것.",
        ]
    lines += [
        "",
        f"- **바퀴 명령**: {'0 으로 눌렀다 — 정책은 이동하지 못한다' if zero_base else '그대로 실행했다 — 이동도 정책이 한 것이다'}",
        f"- **태스크 성공**: {success}",
    ]
    if note:
        lines += ["", f"> ⚠️ {note}"]
    if judge is not None:
        lines += ["", f"- **스킬 성공 판정**: {judge}",
                  f"- **판정 결과**: {'✅ 해냈다' if skill_ok else '❌ 못 했다 (또는 판정기가 없다)'}"]
    if warnings:
        lines += ["", "## 🔴 이 실행은 무대가 어긋났다 — 결과를 그대로 읽으면 안 된다", ""]
        lines += [f"- {w}" for w in warnings]
        lines += ["", "재생은 로봇 관절을 정확히 재현하지만 **물체 상태는 접촉 물리에",
                  "의존해서 어긋날 수 있다**. 위 경고가 있으면 그 실패는 모델 탓이 아니다.",
                  "(HANDOFF 26장)"]
    if notes:
        lines += ["", "## ⚠️ 주의 — 무대 문제는 아니다 (모델이 한 일 또는 판정의 한계)", ""]
        lines += [f"- {w}" for w in notes]
    if expected:
        lines += [""] + expected
    lines += [
        "",
        "## 같이 있는 npz 파일",
        "",
        "`actions_*.npz` 는 **정책 구간만** 담은 기록이다. 재생 구간은 들어 있지 않다.\n"
        "  - `actions` (스텝, 23): 정책이 **낸 명령**\n"
        "  - `states` (스텝, 61): 로봇의 **실제 관절값** — 명령대로 움직였는지 보려면 이쪽을 본다\n"
        "    (2026-09-09 부터 저장. 그 전 파일에는 `actions` 만 있다)",
        "23차원 구성: 바퀴 0~2 / 몸통 3~6 / 왼팔 7~13 / **왼그리퍼 14** / 오른팔 15~21 / **오른그리퍼 22**.",
        "그리퍼는 +1 이 열림, -1 이 쥠이다. 바퀴 값은 `--zero-base` 를 적용하기 **전** 값이라,",
        "눌러서 실행되지 않은 명령도 그대로 남아 있다.",
        "",
    ]
    path = os.path.join(out_dir, "README.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[구간 설명] {path}", flush=True)


def place_robot_near(ev, needle: str, dist: float, angle_deg: float | None = None):
    """로봇을 `needle` 이 이름에 든 물체 앞으로 옮기고 그쪽을 바라보게 한다.

    서 있을 자리는 **물체에서 로봇의 현재 위치 쪽으로** dist 만큼 떨어진 지점으로
    잡는다. 로봇이 지금 서 있는 곳은 적어도 걸어 다닐 수 있는 자리이므로,
    그 방향은 벽 속이 아닐 가능성이 높다. (navmesh 를 쓰지 않는 어림이다.)
    """
    import omnigibson as og
    import omnigibson.utils.transform_utils as T

    def xy(v):
        v = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
        return np.asarray(v, dtype=np.float64)

    hits = [o for o in ev.env.scene.objects if needle.lower() in o.name.lower()]
    if not hits:
        raise SystemExit(f"'{needle}' 이 이름에 든 물체가 장면에 없다")
    obj = hits[0]
    target = xy(obj.get_position_orientation()[0])
    rpos = xy(ev.robot.get_position_orientation()[0])

    d = rpos[:2] - target[:2]
    n = np.linalg.norm(d)
    if angle_deg is not None:
        # ⚠️ 기본 어림(로봇의 현재 위치 방향)은 **물체의 어느 면인지를 모른다.**
        # 실측(2026-09-02): 냉장고에서 리셋 위치 방향은 126.8도(북서 모서리)인데
        # 데모가 실제로 선 곳은 184.1도(정서쪽)였다. 57도 차이라 문 앞이 아니라
        # 모서리에 서게 된다. 이 옵션으로 방향을 직접 준다.
        a = np.radians(angle_deg)
        d = np.array([np.cos(a), np.sin(a)])
    else:
        d = d / n if n > 1e-6 else np.array([1.0, 0.0])
    stand = target[:2] + d * dist
    yaw = float(np.arctan2(target[1] - stand[1], target[0] - stand[0]))
    newpos = th.tensor([stand[0], stand[1], float(rpos[2])], dtype=th.float32)
    quat = T.euler2quat(th.tensor([0.0, 0.0, yaw], dtype=th.float32))

    z_before = float(rpos[2])
    ev.robot.set_position_orientation(position=newpos, orientation=quat)
    for _ in range(5):                       # 물리를 몇 스텝 돌려 자세를 안정시킨다
        og.sim.step()
    after = xy(ev.robot.get_position_orientation()[0])
    print(f"[배치] {obj.name} 앞으로 이동", flush=True)
    print(f"   전: ({rpos[0]:.2f}, {rpos[1]:.2f})  거리 {n:.2f}m")
    got = float(np.linalg.norm(after[:2] - target[:2]))
    dz = float(after[2]) - z_before
    print(f"   후: ({after[0]:.2f}, {after[1]:.2f})  거리 "
          f"{got:.2f}m  (yaw {np.degrees(yaw):.0f}도)  z {after[2]:.3f} "
          f"({dz:+.3f})", flush=True)
    # ⚠️ 요청한 거리에 못 서면 물체와 겹쳐 물리 엔진이 밀어낸 것이다. 그 과정에서
    #    문 같은 가동부가 함께 열려 **정책이 연 것처럼 보인다.** 2026-08-29 에
    #    0.6m 로 놓았다가 0.72m 로 밀려나면서 냉장고 문이 열렸고, 하마터면
    #    "정책이 문을 열었다"로 오독할 뻔했다.
    if abs(got - dist) > 0.05:
        print(f"   ⚠️ 요청 {dist:.2f}m 에 못 섰다 ({got:.2f}m). 물체와 겹쳐 밀려난 것이다 — "
              f"가동부(문 등)가 이미 열렸을 수 있으니 첫 프레임을 반드시 확인할 것",
              flush=True)
    # ⚠️ 위의 검사는 **수평 거리만** 본다. 18.8 에서 184도 배치는 로봇이 옆이
    #    아니라 **위로** 밀려서(z 0.01 -> 0.13m) 이 경고가 뜨지 않았고, 팔이
    #    조리대에 끼어 재생 명령을 못 따라갔는데도 실험이 그대로 진행됐다.
    #    높이는 별도로 봐야 한다. z 가 내려가면 바닥이 없어 떨어진 것이다
    #    (partial_scene_load 라 주방 밖에는 바닥이 없다).
    if abs(dz) > 0.05:
        which = "가구와 겹쳐 위로 밀려났다" if dz > 0 else "바닥이 없어 아래로 떨어졌다"
        raise SystemExit(
            f"   🔴 배치 실패: z 가 {dz:+.3f}m 움직였다 — {which}.\n"
            f"      이 자리에서는 팔이 끼어 명령을 따라가지 못하므로 실험이 무효다.\n"
            f"      `scripts/probe_stand_spots.py` 로 빈 방향을 먼저 찾을 것.")
    print(flush=True)
    # 옮긴 뒤의 관측으로 갱신해야 정책이 새 자리에서 본 화면을 받는다.
    # ⚠️ `_sync_lights_and_get_obs()` 를 인자 없이 부르면 안 된다 — 조명 동기화가
    #    꺼져 있으면 받은 인자(None)를 그대로 돌려줘서 뒤에서 터진다.
    obs, _ = ev.env.get_obs()
    ev.obs = ev._preprocess_obs(ev._sync_lights_and_get_obs(obs))


def main():
    d = np.load(A.demo, allow_pickle=True)
    demo, bounds = d["actions"], json.loads(str(d["boundaries"]))
    b = next((x for x in bounds if x["skill_idx"] == A.skill), None)
    if b is None:
        sys.exit(f"스킬 {A.skill} 이 데모에 없다. 있는 것: "
                 f"{[x['skill_idx'] for x in bounds]}")
    n_steps = b["n"] + A.extra_steps
    handover = b["t0"] if A.replay_frames is None else A.replay_frames
    if A.prompt:
        b = dict(b, instruction=A.prompt, trained=A.prompt in _TOP)
    if A.judge_as and A.chain:
        sys.exit("--judge-as 는 --chain 과 같이 쓸 수 없다 (구간마다 판정 문장이 달라야 한다)")
    if A.judge_as:
        print(f"  ⚖️ 판정은 정책 지시문이 아니라 {A.judge_as!r} 로 잰다", flush=True)

    # ── 이어서 지시하기: 구간 목록과 예산을 만든다 ────────────────────────
    # 연쇄가 아니면 "구간 하나짜리 연쇄" 로 취급한다. 코드 길을 하나로 두어
    # 예전 실행과 결과가 갈리지 않게 하기 위한 것이다.
    if A.chain:
        chain = [s.strip() for s in A.chain.split("|") if s.strip()]
        if A.chain_steps:
            budg = [int(x) for x in A.chain_steps.split("|")]
            if len(budg) == 1:
                budg = budg * len(chain)
            if len(budg) != len(chain):
                sys.exit(f"--chain 은 {len(chain)}개인데 --chain-steps 는 {len(budg)}개다")
        else:
            budg = [b["n"] + A.extra_steps] * len(chain)
        b = dict(b, instruction=chain[0], trained=chain[0] in _TOP)
    else:
        chain, budg = [b["instruction"]], [n_steps]
    n_steps = sum(budg)

    print(f"태스크 {A.task} / 인스턴스 {A.instance} ({A.mode})")
    print(f"데모 재생 {handover:,}프레임 -> 스킬 {A.skill} 의 지시문으로 정책에 인계"
          + ("  (재생 없음)" if handover == 0 else ""))
    if len(chain) > 1:
        print(f"  🔗 이어서 지시하기 — {len(chain)}구간"
              + ("" if A.chain_no_early else
                 f" (성공하면 {A.chain_settle}스텝 뒤 다음 문장으로)"))
        _acc = handover
        for _i, (_s, _n) in enumerate(zip(chain, budg), 1):
            print(f"    {_i}. {_s!r}  ({'학습됨' if _s in _TOP else '🔴 학습 안 됨'})"
                  f"  최대 {_n:,}스텝 = {_n/A.video_fps:.0f}초"
                  f"  [프레임 {_acc:,}~{_acc+_n-1:,}]")
            _acc += _n
    else:
        print(f"  지시문: {b['instruction']!r}  ({'학습됨' if b['trained'] else '학습 안 됨'})")
    print(f"  정책 구간: {n_steps:,}스텝 / 총 {handover+n_steps:,}스텝\n")

    seed_everything(DEFAULT_EVAL_SEED)
    inst = resolve_instance_ids(A.task, [A.instance], mode=A.mode)[0]
    cfg = OmegaConf.create({
        "env_wrapper": {"_target_": "omnigibson.eval.wrappers.DefaultWrapper"},
        "policy_name": "local",
        # 정책 객체는 아래에서 우리 것으로 갈아 끼운다.
        "model": {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None},
        "headless": A.headless, "partial_scene_load": not A.full_scene,
        "max_steps": handover + n_steps + 10,
        "write_video": A.write_video, "mode": A.mode,
        "seed": DEFAULT_EVAL_SEED,
        "task": {"name": A.task}, "robot": None,
    })

    os.makedirs(A.output_dir, exist_ok=True)
    # ⚠️ 파일명에 지시문을 넣지 않으면, 같은 태스크·스킬로 문장만 바꿔 여러 번
    #    돌릴 때 **서로 덮어써서 마지막 것만 남는다.** 2026-08-29 에 이동 실험
    #    영상 두 개를 이렇게 잃었다.
    _slug = "".join(c if c.isalnum() else "-" for c in b["instruction"])[:52].strip("-")
    if len(chain) > 1:
        # 연쇄는 첫 문장 이름만 붙이면 단발 실행과 구분이 안 된다.
        _slug = f"chain{len(chain)}_" + _slug[:44].strip("-")
    video = os.path.join(A.output_dir,
                         f"{A.task}_inst{inst}_{_slug}.mp4")

    with Evaluator(cfg) as ev:
        ev.reset()
        ev.load_task_instance(int(inst))
        ev.reset()
        cams = {
            "head": ev.robot_camera_names["head"] + "::rgb",
            "left_wrist": ev.robot_camera_names["left_wrist"] + "::rgb",
            "right_wrist": ev.robot_camera_names["right_wrist"] + "::rgb",
            "proprio": f"{ev.robot.name}::proprio",
        }
        print("카메라/관절 키:", json.dumps(cams, ensure_ascii=False, indent=2))
        describe_scene(ev)
        if A.place_near and not A.place_after_replay:
            place_robot_near(ev, A.place_near, A.place_dist, A.place_angle)

        pol = ReplayThenPolicy(demo, handover, b["instruction"],
                               ChunkClient(A.host, A.port), cams, A.exec_steps,
                               demo_states=d["states"] if "states" in d.files else None,
                               ref_frame=b["t0"])
        if A.clamp:
            for spec in A.clamp:
                try:
                    d, lo, hi = spec.split(":")
                    d, lo, hi = int(d), float(lo), float(hi)
                except ValueError:
                    sys.exit(f"❌ --clamp 형식이 틀렸다: {spec!r} (예: 11:-0.25:0.25)")
                if not 0 <= d < 23 or lo > hi:
                    sys.exit(f"❌ --clamp 값이 범위 밖이다: {spec!r}")
                pol.clamps.append((d, lo, hi))
            print("[명령 묶기] " + ", ".join(
                f"{d}번 → [{lo:+.3f}, {hi:+.3f}]" for d, lo, hi in pol.clamps), flush=True)
        if A.yaw_assist:
            try:
                y0, y1, rate = (float(v) for v in A.yaw_assist.split(":"))
            except ValueError:
                sys.exit(f"❌ --yaw-assist 형식이 틀렸다: {A.yaw_assist!r} (예: -30:-65:-0.25)")
            if not (y1 < y0 < 0 and rate < 0):
                sys.exit(f"❌ --yaw-assist 는 목표 < 시작 < 0, 속도 < 0 이어야 한다: "
                         f"{A.yaw_assist!r}")
            if ZERO_BASE:
                sys.exit("❌ --yaw-assist 는 바퀴 명령이 실행돼야 쓸 수 있다 (ZERO_BASE 가 켜져 있다)")
            pol.yaw_assist = (float(np.radians(y0)), float(np.radians(y1)), rate)
            print(f"[회전 보조] 누적 회전 {y0:+.0f}° 를 넘으면 {y1:+.0f}° 까지 "
                  f"{rate:+.2f} rad/s 로 이어 준다", flush=True)
        ev.policy = pol
        if A.write_video:
            ev.start_recording(video, rate=A.video_fps)

        # 🔴 인계 시점에 물체 상태가 데모대로 재현됐는지 본다 (26장)
        stage_warn, held_at_handover = [], {}
        # 무대 문제가 **아닌** 주의점(모델이 한 부수 변화 등)은 따로 모은다. 예전에는
        # stage_warn 에 섞여서 설명 파일 맨 위에 '무대가 어긋났다' 가 붙었다 (HANDOFF 40.16/40.18).
        note_warn = []
        tgt_at_handover = {}
        open_curve, open_obj = [], None   # 문이 얼마나 열렸나 (32.2)
        open_disagree = []                # 판정기와 열린 비율이 어긋난 순간들
        nouns = sorted({w for x in bounds
                        for w in re.findall(r"[a-z]+", x["instruction"].lower())
                        if len(w) > 3})

        # 스킬 수준 성공 판정을 준비한다 (인계 뒤부터만 본다)
        check, why = (None, "쓰지 않음")
        # 🔴 판정기는 **항상** 켠다. 예전에는 `--stop-on-success` 안에 묶여 있어서,
        # 조기 종료를 끄면 판정기까지 같이 꺼졌다 — 2026-09-09 에 문 열기를 끝까지
        # 재려고 조기 종료를 껐더니 **언제 열렸는지 기록이 통째로 사라졌다.**
        # 관찰(판정)과 개입(조기 종료)은 별개여야 한다.
        try:
            check, why = make_success_check(ev, A.judge_as or b["instruction"])
        except Exception as e:
            check, why = None, f"{type(e).__name__}: {e}"
        print(f"\n[성공 판정] {why}"
              + ("" if A.stop_on_success else "   (조기 종료는 껐다 — 재기만 한다)"),
              flush=True)
        hold, stop_at = 0, None
        first_ok = None                      # 처음 성공 조건이 찬 스텝

        # ── 이어서 지시하기: 구간 상태 (28.6 가) ─────────────────────────
        # 연쇄가 아니면 **구간 1개짜리 연쇄**로 돈다. 코드 길을 둘로 두면
        # 단발 실행의 결과가 조용히 갈릴 수 있어서다.
        seg = 0                              # 지금 구간 (0부터)
        seg_from = handover                  # 이 구간이 시작한 스텝
        seg_until = handover + budg[0]       # 예산이 끝나는 스텝
        seg_adv = None                       # 성공해서 넘어가기로 한 스텝
        seg_open = []                        # 이 구간의 (초, 열린비율)
        seg_t0 = {}                          # 이 구간 시작 시점의 목표 물체 상태
        seg_pre_true = False                 # 🔴 시작부터 조건이 참이었나
        seg_closed = False                   # 지금 구간을 이미 기록했나
        segrec = []                          # 구간별 기록 (요약표의 근거)

        def start_seg(idx, at):
            """구간 `idx` 를 연다 — 문장을 갈고, 판정기를 새로 만들고,
            시작 시점 상태를 찍는다."""
            nonlocal check, why, hold, first_ok, open_obj
            nonlocal seg_open, seg_t0, seg_pre_true, seg_adv
            ins = chain[idx]
            print(f"\n{'─'*68}\n🔗 [{idx+1}/{len(chain)}구간] {ins!r}"
                  f"   (인계 +{(at-handover)/A.video_fps:.1f}초부터, "
                  f"최대 {budg[idx]/A.video_fps:.0f}초)\n{'─'*68}", flush=True)
            pol.instruction = ins
            # 🔴 앞 문장으로 받아 둔 청크를 **버린다.** 안 버리면 문장을 바꾼
            #    뒤에도 최대 exec_steps(16스텝 = 0.5초) 동안 **이전 지시의
            #    행동**이 그대로 나가서, 전환 시점이 흐려진다.
            pol.chunk, pol.k = None, 0
            jt = A.judge_as or ins           # 판정·추적용 문장 (정책이 듣는 것은 ins)
            try:
                check, why = make_success_check(ev, jt)
            except Exception as e:
                check, why = None, f"{type(e).__name__}: {e}"
            hold, first_ok, seg_adv, seg_open = 0, None, None, []
            try:
                seg_t0 = report_target_states(ev, jt, label=f"{idx+1}구간 시작") or {}
                _o = find_target_obj(ev, jt)
                open_obj = _o if (_o is not None
                                  and open_fraction(_o) is not None) else None
            except Exception as e:
                seg_t0, open_obj = {}, None
                print(f"[{idx+1}구간 시작] 상태를 못 읽었다: "
                      f"{type(e).__name__}: {e}", flush=True)
            # 🔴 시작부터 조건이 참이면 이 구간은 **측정이 안 된다.** 앞 구간이
            #    실패해서 그런 경우가 많다 — 문을 못 열었으면 "닫아라" 는 이미
            #    달성된 과제다. 그대로 세면 연쇄가 2/3 성공처럼 보인다 (28.6 다).
            seg_pre_true = False
            if check is not None:
                try:
                    seg_pre_true = bool(check())
                except Exception:
                    seg_pre_true = False
            print(f"[{idx+1}구간 판정] {why}"
                  + ("   🔴 **시작부터 참이다 — 이 구간은 측정 못 한다**"
                     if seg_pre_true else ""), flush=True)

        def finish_seg(at):
            """구간을 닫고 기록한다 — 순간 성공과 끝 상태를 **따로** 남긴다."""
            ins = chain[seg]
            try:
                t1 = report_target_states(ev, ins, label=f"{seg+1}구간 끝") or {}
            except Exception:
                t1 = {}
            mx = max((f for _, f in seg_open), default=None)
            rec = {
                "seg": seg + 1, "instruction": ins, "judge": why,
                "from": seg_from, "to": at,
                "sec": round((at - seg_from + 1) / A.video_fps, 1),
                "pre_true": seg_pre_true,
                # 🔴 시작부터 참이던 구간은 성공으로 세지 않는다.
                "ok": (first_ok is not None) and not seg_pre_true,
                "ok_sec": (None if first_ok is None
                           else round((first_ok - seg_from) / A.video_fps, 1)),
                "open_max": mx,
                "open_end": (seg_open[-1][1] if seg_open else None),
                "t0": dict(seg_t0), "t1": dict(t1),
            }
            segrec.append(rec)
            if len(chain) > 1:
                mark = ("🔴 무효(시작부터 참)" if seg_pre_true
                        else "✅ 해냈다" if rec["ok"] else "❌ 못 했다")
                extra = ""
                if mx is not None:
                    extra = f"  열림 최대 {mx:.1%} / 끝 {rec['open_end']:.1%}"
                print(f"[{seg+1}구간 결과] {mark}"
                      + (f"  ({rec['ok_sec']}초 만에)" if rec["ok"] else "")
                      + f"  {rec['sec']}초 씀{extra}", flush=True)
            return rec

        t0 = time.time()
        for i in range(handover + n_steps):
            terminated, truncated = ev.step()
            if i == handover - 1:
                print(f"\n▶ 재생 끝 ({time.time()-t0:.0f}초). 여기서부터 정책이 운전한다\n")
                try:
                    stage_warn, held_at_handover = check_handover_state(
                        ev, demo, handover, nouns)
                except Exception as e:
                    stage_warn = [f"인계 물체 상태를 못 읽었다: {type(e).__name__}: {e}"]
                for k, v in held_at_handover.items():
                    print(f"[인계 물체] {k} 손이 쥔 것: {v}", flush=True)
                for w in stage_warn:
                    print(f"[무대 검사] {w}", flush=True)
                if not stage_warn:
                    print("[무대 검사] ✅ 손에 쥔 것이 데모와 맞는다 "
                          "(문·전원 등 물체 자체의 상태는 이 검사가 보지 않는다)",
                          flush=True)
                # 인계 시점의 상태도 찍는다. 이것이 없으면 끝 상태만 보고
                # "정책이 바꾼 것" 인지 "원래 그랬던 것" 인지 못 가른다.
                # 2026-09-08 에 `micro_close` 의 ToggledOn=True 가 정책 탓인지
                # 확인하려고 다른 실행 세 개를 뒤져야 했다.
                # 🚪 문을 미리 열어 놓는다. **상태를 찍기 전에** 해야 로그의
                #    `인계 시점` 이 실제 시작 상태를 보여 준다.
                if A.preset_open is not None:
                    try:
                        _po = find_target_obj(ev, b["instruction"])
                        _done = set_open_fraction(_po, A.preset_open) if _po else None
                        if _done:
                            print(f"[무대 세팅] {_po.name} 의 문을 "
                                  f"{A.preset_open:.0%} 로 직접 열었다 — {_done}",
                                  flush=True)
                        else:
                            print(f"🔴 [무대 세팅] 문을 못 열었다 "
                                  f"(목표 물체 {_po.name if _po else '없음'}) — "
                                  "이 판은 실험 조건을 못 만들었다", flush=True)
                            stage_warn.append(
                                "`--preset-open` 이 문을 열지 못했다. 이 판은 "
                                "비교 대상이 아니다.")
                    except Exception as e:
                        print(f"🔴 [무대 세팅] 실패 — {type(e).__name__}: {e}",
                              flush=True)
                        stage_warn.append(
                            f"`--preset-open` 이 터졌다 ({type(e).__name__}). "
                            "이 판은 비교 대상이 아니다.")
                try:
                    tgt_at_handover = report_target_states(
                        ev, b["instruction"], label="인계 시점")
                    _o = find_target_obj(ev, b["instruction"])
                    open_obj = _o if (_o is not None
                                      and open_fraction(_o) is not None) else None
                except Exception as e:
                    print(f"[목표 물체 인계 시점] 못 읽었다: {type(e).__name__}: {e}",
                          flush=True)
                # 1구간의 시작 상태 = 인계 시점 상태. 뒤 구간과 같은 형식으로
                # 남겨야 요약표에서 나란히 읽힌다.
                seg_t0 = dict(tgt_at_handover or {})
                if check is not None:
                    try:
                        seg_pre_true = bool(check())
                    except Exception:
                        seg_pre_true = False
                if seg_pre_true and len(chain) > 1:
                    print("🔴 [1구간] 시작부터 조건이 참이다 — 이 구간은 측정 못 한다",
                          flush=True)
                print(flush=True)
                if A.place_near and A.place_after_replay:
                    # 팔을 다 든 **뒤에** 옮긴다. 반대로 하면 팔이 가구에 막힌다.
                    place_robot_near(ev, A.place_near, A.place_dist, A.place_angle)
                    describe_scene(ev)
                    pol._reported = False      # 옮긴 뒤 자세를 다시 찍는다
            # ── 스킬을 실제로 해냈는지 본다 (구간 시작 이후, 5스텝마다)
            if (check is not None and i >= seg_from and stop_at is None
                    and seg_adv is None and i % 5 == 0):
                try:
                    hit = check()
                except Exception as e:
                    print(f"[성공 판정] 중단 — {type(e).__name__}: {e}", flush=True)
                    check = None
                    hit = False
                if hit:
                    hold += 5
                    if hold >= A.success_hold:
                        if i - seg_from < A.success_hold:
                            # 구간 시작부터 참이면 정책이 한 일이 아니다
                            print(f"[성공 판정] {seg+1}구간은 시작 시점에 이미 조건이 "
                                  "참이다 — 판정을 끈다", flush=True)
                            check = None
                            # 🔴 이것은 무대 결함이다 — 시킬 일이 이미 되어
                            # 있으니 정책이 할 수 있는 게 없다. 예전에는 판정만
                            # 조용히 끄고 넘어가서, `micro_close` 가 무대 검사
                            # 초록불을 켠 채 "실패" 로 기록됐다. 파지 검사는
                            # 문·전원 상태를 안 보므로 여기서 따로 잡는다.
                            stage_warn.append(
                                f"인계 시점에 성공 조건이 **이미 참**이었다 ({why}). "
                                "재생이 앞 단계의 물체 상태를 재현하지 못한 것이다 — "
                                "이 실행은 이 스킬을 측정하지 못했다.")
                        elif first_ok is None:
                            first_ok = i
                            _in_seg = (i - seg_from) / A.video_fps
                            if len(chain) > 1 and not A.chain_no_early:
                                # 연쇄에서는 "해냈으니 다음" 이 자연스러운 의미다.
                                # 다만 판정기는 문이 5%만 열려도 참이므로,
                                # 곧바로 넘기지 않고 --chain-settle 만큼 더 준다.
                                seg_adv = i + A.chain_settle
                                print(f"\n✅ [{seg+1}구간 성공] 스텝 {i} "
                                      f"({_in_seg:.1f}초 만에) — {A.chain_settle}스텝 "
                                      f"뒤 다음 문장으로 넘긴다", flush=True)
                            elif A.stop_on_success:
                                stop_at = i + A.success_extra
                                print(f"\n✅ [스킬 성공] 스텝 {i} "
                                      f"({_in_seg:.1f}초 만에) "
                                      f"— {A.success_extra}스텝 더 돌리고 멈춘다", flush=True)
                            else:
                                print(f"\n✅ [스킬 성공] 스텝 {i} "
                                      f"({_in_seg:.1f}초 만에) "
                                      f"— 조기 종료는 껐으므로 계속 돌린다", flush=True)
                else:
                    hold = 0
            # 🔴 문이 **얼마나** 열렸는지 1초마다 남긴다 (32.2). 불리언만 보면
            # 5% 와 90% 가 같은 '성공' 이라 "더 돌리면 더 여는가" 를 영원히 못 잰다.
            if open_obj is not None and i >= handover and (i - handover) % 30 == 0:
                _fr = open_fraction(open_obj)
                if _fr is not None:
                    open_curve.append(((i - handover) / A.video_fps, _fr))
                    seg_open.append(((i - seg_from) / A.video_fps, _fr))
                    # 🚪 "크게 열렸으면 멈춘다". 판정기(5%)와 달리 문턱을 직접 준다.
                    if (A.stop_open_frac is not None and stop_at is None
                            and _fr >= A.stop_open_frac):
                        stop_at = i + A.stop_open_extra
                        print(f"\n🚪 [문이 크게 열렸다] 스텝 {i} "
                              f"({(i-handover)/A.video_fps:.1f}초) — 열린 비율 "
                              f"{_fr:.1%} ≥ {A.stop_open_frac:.0%}, "
                              f"{A.stop_open_extra}스텝 더 돌리고 멈춘다", flush=True)
                    # 🔴 판정기(Open 불리언)를 **같은 순간에** 같이 읽는다.
                    # 2026-09-09 `fridge_open2` 1회차에서 11초에 열린 비율이
                    # 16.3% 였는데 판정기는 24초에야 참이 됐다. 임계값이 5% 이니
                    # 둘은 반드시 같아야 한다 — 어긋난다면 둘 중 하나가 틀렸고,
                    # 성공률 표는 판정기 위에 서 있으므로 먼저 가려야 한다.
                    try:
                        from omnigibson.object_states import Open as _Open
                        _bv = bool(open_obj.states[_Open].get_value())
                        if _bv != (_fr > 0.05):
                            open_disagree.append(
                                ((i - handover) / A.video_fps, _fr, _bv))
                    except Exception:
                        pass
            # ── 이어서 지시하기: 구간을 닫고 다음 문장으로 넘긴다
            #    (해냈으면 settle 뒤에, 못 했으면 예산을 다 쓴 뒤에)
            if i >= handover and ((seg_adv is not None and i >= seg_adv)
                                  or i >= seg_until - 1):
                finish_seg(i)
                seg_closed = True
                if seg + 1 < len(chain):
                    seg += 1
                    seg_from, seg_until = i + 1, i + 1 + budg[seg]
                    seg_closed = False
                    start_seg(seg, seg_from)
                else:
                    # 🔴 마지막 구간이 닫히면 **반드시 끊는다.** 안 끊으면 다음
                    #    스텝에도 `i >= seg_until - 1` 이 계속 참이라 같은 구간을
                    #    매 스텝 다시 기록한다 (앞 구간이 일찍 성공해 예산이
                    #    남았을 때 실제로 수백 줄이 쌓였다).
                    print(f"[연쇄 끝] 마지막 구간까지 마쳤다 @스텝 {i}"
                          + (" — 해냈다" if seg_adv is not None else ""), flush=True)
                    break
            if stop_at is not None and i >= stop_at:
                print(f"성공 뒤 여유 구간까지 마쳤다 @스텝 {i}", flush=True)
                break
            if terminated or truncated:
                print(f"에피소드 종료 (terminated={terminated}, truncated={truncated}) @스텝 {i}")
                break
        # 중간에 끊겼으면(에피소드 종료·조기 종료) 열려 있던 구간도 닫아 기록한다.
        if not seg_closed and i >= handover:
            finish_seg(i)
        if A.write_video:
            ev.stop_recording()
        if A.save_actions and pol.all_acts:
            slug = "".join(c if c.isalnum() else "-" for c in b["instruction"])[:56]
            if len(chain) > 1:
                slug = f"chain{len(chain)}_" + slug[:48]
            ap_ = os.path.join(A.output_dir, f"actions_{slug}.npz")
            _extra = {}
            if pol.all_states:
                _ns = min(len(pol.all_states), len(pol.all_acts))
                _extra["states"] = np.stack(pol.all_states[:_ns])
            if open_curve:
                # 🔴 문 열림 곡선을 통째로 남긴다 (초, 비율). 로그에는 12점으로
                #    솎아서 찍히는데(7초 간격), 2026-09-10 에 늦은 인계 두 판이
                #    갈린 시점을 짚으려다 **56~63초 사이 어디인지 못 좁혔다.**
                #    1초 간격 원본이 있으면 그런 일이 안 생긴다.
                _extra["open_curve"] = np.asarray(open_curve, dtype=np.float32)
            if len(chain) > 1:
                # 🔴 구간 경계를 같이 저장한다. 이게 없으면 나중에 궤적을
                #    분석할 때 **어느 줄이 어느 문장이었는지** 알 수 없고,
                #    "결과가 갈린 시점 이전 구간에서 재라"(34.6) 를 못 지킨다.
                #    행동 배열의 인덱스 = 스텝 - handover 다.
                _extra["segments"] = json.dumps(
                    [{"seg": r["seg"], "instruction": r["instruction"],
                      "a0": r["from"] - handover, "a1": r["to"] - handover,
                      "ok": r["ok"], "pre_true": r["pre_true"]} for r in segrec],
                    ensure_ascii=False)
            np.savez_compressed(ap_, actions=np.stack(pol.all_acts),
                                instruction=b["instruction"], **_extra)
            print(f"[행동 저장] {ap_}", flush=True)

        # 끝난 뒤에도 좌표를 찍는다 — 이동 실험은 "물체에 가까워졌나"가 판정 기준이라
        # 시작·종료 거리를 나란히 봐야 한다.
        describe_scene(ev)
        if pol.base_cmds:
            bc = np.stack(pol.base_cmds)
            print(f"\n[베이스 명령] {'0 으로 눌렀음' if ZERO_BASE else '그대로 실행'}"
                  f"  ({len(bc)}스텝)", flush=True)
            for i, nm in enumerate(("vx", "vy", "wz")):
                print(f"   {nm}: 평균 {bc[:,i].mean():+.4f}  절대평균 "
                      f"{np.abs(bc[:,i]).mean():.4f}  최대 {np.abs(bc[:,i]).max():.4f}")
        try:
            fp = ev.robot.get_position_orientation()[0]
            fp = fp.numpy() if hasattr(fp, "numpy") else np.asarray(fp)
            print(f"[최종 로봇 위치] ({fp[0]:.2f}, {fp[1]:.2f})", flush=True)
        except Exception:
            pass
        # ⚠️ flush 를 빼면 이 세 줄을 잃는다. OmniGibson 은 종료할 때 버퍼를
        #    비우지 않고 프로세스를 끊어서, 표준출력에 남아 있던 것이 사라진다.
        #    2026-09-03 에 롤아웃 두 건의 성공 여부를 이렇게 놓쳤다.
        print(f"\n총 {time.time()-t0:.0f}초 / 추론 {pol.n_infer}회 "
              f"(중앙값 {np.median(pol.lat) if pol.lat else 0:.1f}초)", flush=True)
        ok = bool(ev.env.task.success)
        skill_ok = first_ok is not None
        # ── 이어서 지시하기 요약 ─────────────────────────────────────────
        if len(chain) > 1:
            done = [r for r in segrec if r["ok"]]
            void = [r for r in segrec if r["pre_true"]]
            skill_ok = len(done) == len(chain)
            print(f"\n{'='*68}\n🔗 [연쇄 결과] {len(done)}/{len(chain)}구간 성공"
                  + (f"  (무효 {len(void)}구간 — 시작부터 조건이 참이었다)" if void else "")
                  + f"\n{'='*68}", flush=True)
            print("| # | 지시문 | 결과 | 걸린 시간 | 구간 길이 | 열림 시작→최대→끝 |")
            print("|---|---|---|---|---|---|")
            for r in segrec:
                mark = ("🔴 무효(시작부터 참)" if r["pre_true"]
                        else "✅ 해냈다" if r["ok"] else "❌ 못 했다")
                # 🔴 **시작 열림**을 반드시 같이 적는다. 앞 구간이 문을 5%만 열고
                #    넘겼다면 뒤의 `닫아라` 는 거의 아무것도 안 해도 참이 된다 —
                #    시작값 없이 결과만 보면 그 가짜 성공을 못 가른다.
                _s0 = r["t0"].get("열린비율")
                opn = ("—" if r["open_max"] is None else
                       f"{'?' if _s0 is None else format(_s0, '.1%')} → "
                       f"{r['open_max']:.1%} → {r['open_end']:.1%}")
                print(f"| {r['seg']} | `{r['instruction']}` | {mark} | "
                      f"{'—' if r['ok_sec'] is None else str(r['ok_sec'])+'초'} | "
                      f"{r['sec']}초 | {opn} |")
            # 문 계열 구간이 "거의 닫힌 문" 을 받고 시작했으면 짚어 준다.
            for r in segrec:
                _s0 = r["t0"].get("열린비율")
                if (r["ok"] and _s0 is not None and _s0 < 0.10
                        and re.match(r"close the (door|lid) of ",
                                     r["instruction"].lower())):
                    note_warn.append(
                        f"{r['seg']}구간(`{r['instruction']}`)은 **시작 열림이 "
                        f"{_s0:.1%}** 였다. 판정 문턱이 5% 이므로 이 성공은 거의 "
                        "아무것도 안 해도 나는 값이다 — 깨끗한 성공으로 세지 말 것.")
            # 구간마다 "시킨 것 말고 무엇이 바뀌었나" 를 따로 본다. 연쇄 전체를
            # 한 덩어리로 보면 앞 구간이 바꾼 것을 뒤 구간의 부작용으로 오독한다.
            for r in segrec:
                _a = ("Open" if re.match(r"(open|close) the (door|lid) of ",
                                         r["instruction"].lower())
                      else "ToggledOn" if r["instruction"].lower().startswith("turn on ")
                      else "쥔손" if r["instruction"].lower().startswith("pick up ")
                      else None)
                # 놓기(`put ...`)에서 목표 물체를 놓는 것은 과제 그 자체다 — 부수 변화가 아니다.
                _put = r["instruction"].lower().startswith("put ")
                ch = {k: (r["t0"][k], v) for k, v in r["t1"].items()
                      if k in r["t0"] and r["t0"][k] != v
                      and k != _a and not (_a == "Open" and k == "열린비율")
                      and not (_put and k == "쥔손" and v == "없음")}
                if ch:
                    txt = " / ".join(f"{k}: {x} -> {y}" for k, (x, y) in ch.items())
                    print(f"[{r['seg']}구간 부작용] 시키지 않은 상태가 바뀌었다 — {txt}",
                          flush=True)
                    note_warn.append(
                        f"{r['seg']}구간(`{r['instruction']}`)에서 **시키지 않은 "
                        f"상태까지 바뀌었다** — {txt}. 판정기는 지시문이 요구한 "
                        f"것({_a}) 하나만 본다 (21.4).")
            if void:
                note_warn.append(
                    "연쇄 중 **시작부터 조건이 참이던 구간**이 있다 — "
                    + ", ".join(f"{r['seg']}구간(`{r['instruction']}`)" for r in void)
                    + ". 앞 구간이 실패해서 뒤 구간의 과제가 이미 달성된 상태였을 "
                      "수 있다. 그 구간의 결과는 정책의 성과로 읽으면 안 된다 (28.6 다).")
        # 묶기가 실제로 걸렸는지 — 안 걸렸으면 이 판은 대조군과 다를 게 없다
        if pol.clamps:
            n_pol = max(1, len(pol.all_acts))
            for d, lo, hi in pol.clamps:
                cnt, over = pol.clamp_hits.get(d, [0, 0.0])
                print(f"\n[명령 묶기] {d}번 [{lo:+.3f}, {hi:+.3f}] — "
                      f"{cnt}/{n_pol}스텝 ({cnt / n_pol:.0%}) 에서 잘렸다, "
                      f"정책이 가장 많이 넘어가려 한 양 {over:+.3f}", flush=True)
            if all(pol.clamp_hits.get(d, [0])[0] == 0 for d, _, _ in pol.clamps):
                print("   ⚠️ **한 번도 안 잘렸다 — 이 판은 묶기가 없는 것과 같다.**",
                      flush=True)
        # 보조가 실제로 걸렸는지 — 안 걸렸으면 이 판은 대조군과 다를 게 없다
        if pol.yaw_assist is not None:
            if pol.assist_log:
                for t_, what, y_ in pol.assist_log:
                    print(f"\n[회전 보조] 정책 {t_:.1f}초 {what} — 누적 회전 {y_:+.1f}°", flush=True)
            else:
                print(f"\n[회전 보조] ⚠️ **안 걸렸다** — 정책이 스스로 "
                      f"{np.degrees(pol.yaw_assist[0]):+.0f}° 까지 안 돌았다. "
                      "이 판은 대조군과 같다.", flush=True)
            print(f"[회전 보조] 정책 구간 전체의 누적 회전 {np.degrees(pol.yaw_int):+.1f}°",
                  flush=True)
        if open_curve:
            fr0, frN = open_curve[0][1], open_curve[-1][1]
            peak_t, peak = max(open_curve, key=lambda x: x[1])
            print(f"\n[열린 비율] 인계 {fr0:.1%} -> 끝 {frN:.1%}"
                  f"  (최대 {peak:.1%} @ {peak_t:.1f}초)", flush=True)
            step = max(1, len(open_curve) // 12)
            print("   " + "  ".join(f"{t:.0f}s:{f:.0%}"
                                    for t, f in open_curve[::step]), flush=True)
            # 끝에서도 계속 열리는 중이면 더 길게 재야 한다
            if len(open_curve) >= 4 and frN >= peak - 1e-6 and frN - open_curve[-4][1] > 0.02:
                print("   ⚠️ **끝나는 순간에도 계속 열리는 중이었다 — 더 길게 다시 잴 것**",
                      flush=True)
        if open_disagree:
            print(f"   🔴 **판정기와 열린 비율이 어긋난 순간 {len(open_disagree)}회** "
                  "— 임계값 5% 대로라면 둘은 같아야 한다:", flush=True)
            for _t, _f, _b in open_disagree[:8]:
                print(f"      {_t:5.1f}초  열린비율={_f:6.1%}  판정기Open={_b}", flush=True)
            if len(open_disagree) > 8:
                print(f"      ... 외 {len(open_disagree)-8}회", flush=True)
        print("성공:", ok, f"(태스크 전체 기준) / 스킬 성공: {skill_ok}", flush=True)
        # 판정에 쓴 상태 하나만이 아니라 목표 물체의 상태를 전부 찍는다.
        # "못 했다" 와 "다른 걸 했다" 를 로그에서 가르기 위한 것이다 (21.4).
        try:
            tgt_at_end = report_target_states(ev, b["instruction"], label="최종 상태")
            # ⚠️ **부작용 검사** — "시킨 것을 했는가" 만이 아니라 "시키지 않은 것을
            # 안 했는가" 도 본다. 2026-09-09 에 `open the door of the oven` 이
            # 문을 열면서 **전원까지 켰다.** 판정기는 `Open` 만 보므로 그냥
            # 성공으로 기록됐다. 21.4 의 "같은 물체의 두 동작을 못 고른다" 가
            # 성공한 실행 안에도 들어 있는 것이라, 이걸 안 적으면 놓친다.
            # 지시문이 요구한 상태 하나를 짚는다. 그 밖의 변화가 '부작용' 이다.
            _ins = b["instruction"].lower().strip()
            asked = ("Open" if re.match(r"(open|close) the (door|lid) of ", _ins)
                     else "ToggledOn" if _ins.startswith("turn on ")
                     else "쥔손" if _ins.startswith("pick up ")
                     else None)
            changed = {k: (tgt_at_handover[k], v)
                       for k, v in (tgt_at_end or {}).items()
                       if k in tgt_at_handover and tgt_at_handover[k] != v}
            if changed:
                print("[상태 변화] " + " / ".join(
                    f"{k}: {a} -> {b_}" for k, (a, b_) in changed.items()), flush=True)
            # `열린비율` 은 `Open` 의 정도일 뿐 별개 상태가 아니다. 부작용에서 뺀다
            # (2026-09-09: 문을 100% 연 실행에 "시키지 않은 상태가 바뀌었다" 오탐).
            # 놓기(`put ...`)에서 목표 물체를 놓는 것(쥔손 → 없음)은 과제 그 자체라 뺀다.
            # 예전에는 hotdog·tray 를 제대로 놓은 판에 "시키지 않은 상태 변화" 가 붙었다
            # (HANDOFF 40.16). `[상태 변화]` 줄에는 그대로 남는다.
            side = {k: v for k, v in changed.items()
                    if k != asked and not (asked == "Open" and k == "열린비율")
                    and not (_ins.startswith("put ") and k == "쥔손" and v[1] == "없음")}
            # 연쇄는 위에서 **구간마다** 이미 봤다. 전체를 한 덩어리로 다시 보면
            # 1구간이 바꾼 것을 3구간의 부작용으로 오독한다 (오븐 3연쇄가 정확히
            # 그렇다 — 문을 열고 닫고 켜므로 Open 도 ToggledOn 도 다 바뀐다).
            if len(chain) > 1:
                side = {}
            if side:
                txt = " / ".join(f"{k}: {a} -> {b_}" for k, (a, b_) in side.items())
                print(f"[부작용] 시키지 않은 상태가 바뀌었다 — {txt}", flush=True)
                note_warn.append(
                    f"⚠️ **시키지 않은 상태까지 바뀌었다** — {txt}. 판정기는 지시문이 "
                    f"요구한 것({asked}) 하나만 보므로, 성공이 떴더라도 **깨끗한 "
                    f"성공이 아니다** (21.4 의 '같은 물체의 두 동작을 못 고른다')." )
        except Exception as e:
            print(f"[목표 물체 최종 상태] 못 읽었다: {type(e).__name__}: {e}", flush=True)
        print("영상:", video, flush=True)
        try:
            exp = expected_script(demo, bounds, A.skill, A.video_fps,
                                  held=held_at_handover, prompt=A.prompt)
        except Exception as e:
            exp = [f"(기대 동작을 적지 못했다: {type(e).__name__}: {e})"]
        _gi = (b["instruction"] if len(chain) == 1
               else " → ".join(chain) + f"  ({len(chain)}구간 연쇄)")
        write_guide(A.output_dir, video if A.write_video else None, A.task, inst,
                    A.demo, A.skill, _gi, all(s in _TOP for s in chain),
                    handover, n_steps, A.video_fps, ZERO_BASE, ok,
                    expected=exp, warnings=stage_warn, notes=note_warn,
                    judge=why, skill_ok=skill_ok,
                    segments=segrec)


if __name__ == "__main__":
    main()
