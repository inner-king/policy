"""수집 원본 해상도 평가 래퍼 — RGB 전용판과 RGB+depth 판.

왜 필요한가 (실측, v3.9.1):
- 기본 DefaultWrapper는 224x224라 학습 분포(head 720 / wrist 480 → 512 resize)와 어긋난다.
- 공식 RGBDFullResWrapper는 depth_linear 모달리티를 추가하고 env.load_observation_space()
  **전역** 재로드를 부르는데, 그 경로가 headless에서
  `AttributeError: 'NoneType' object has no attribute 'view'`로 죽는다.
- DefaultWrapper가 쓰는 **센서별** sensor.load_observation_space() 경로는 멀쩡하다.

그래서: DefaultWrapper의 작동 경로를 그대로 쓰되 해상도만 수집 원본으로 맞춘다.
모달리티만 다른 두 래퍼가 그 경로를 공유한다.

  RGBFullResWrapper   rgb            — SmolVLA, groot h16/h40 실행들
  RGBDFullResWrapper  rgb + depth    — groot-radioinit-h40-rgbd-200k

둘 다 `B1K_ACTION_LOG_DIR` 이 설정되면 적용된 액션을 에피소드별로 남긴다 (_ActionLog).
설정하지 않으면 아무 일도 하지 않는다.

챌린지 규정상 둘 다 허용 범위 안이다 (challenge track = RGB + depth + proprioception,
docs/challenge/evaluation.md). 공식 평가 래퍼는 RGBD 쪽이다.

사용:
    PYTHONPATH=$SRC/common \\
    python -m omnigibson.eval.eval ... --env-wrapper b1k_wrappers.RGBFullResWrapper
    python -m omnigibson.eval.eval ... --env-wrapper b1k_wrappers.RGBDFullResWrapper

    B1K_ACTION_LOG_DIR=$OUTDIR/actions ...   # 액션 로그까지 남기려면
"""
import atexit
import json
import os

import numpy as np

from omnigibson.envs import Environment, EnvironmentWrapper
from omnigibson.eval.utils.eval_utils import (
    HEAD_RESOLUTION,
    WRIST_RESOLUTION,
    get_robot_camera_names,
    set_sensor_modalities,
)
from omnigibson.utils.ui_utils import create_module_logger

logger = create_module_logger(module_name=__name__)


def _apply_full_res(env: Environment, modalities: set[str], label: str) -> None:
    """수집 원본 해상도로 맞추고 모달리티를 지정한다. 센서별 재로드 경로를 쓴다."""
    robot = env.robots[0]
    robot_eval_config = getattr(env, "_eval_robot_config", {})
    camera_roles_by_sensor_name = {
        camera_name.split("::")[1]: camera_id
        for camera_id, camera_name in get_robot_camera_names(robot.name, robot_eval_config).items()
    }
    for sensor_name, sensor in robot.sensors.items():
        if not hasattr(sensor, "image_height") or not hasattr(sensor, "image_width"):
            continue
        set_sensor_modalities(sensor, modalities)
        if camera_roles_by_sensor_name.get(sensor_name) == "head":
            sensor.image_height, sensor.image_width = HEAD_RESOLUTION
        else:
            sensor.image_height, sensor.image_width = WRIST_RESOLUTION
        # ⚠️ env.load_observation_space() 전역 재로드가 아니라 센서별 재로드를 쓴다
        #    (전역 경로는 NoneType.view로 죽는다 — 모듈 docstring 참고)
        sensor_space = sensor.load_observation_space()
        if env.observation_space is not None:
            env.observation_space.spaces[robot.name].spaces[sensor_name] = sensor_space
    logger.info(f"{label} full-res observation spaces reloaded!")


class _ActionLog:
    """롤아웃에 **실제로 적용된** 23차원 액션을 에피소드별로 남긴다.

    `B1K_ACTION_LOG_DIR` 이 설정되어 있을 때만 동작한다. 없으면 record/flush 가 즉시
    반환하므로 기존 평가와 같은 경로를 탄다.

    왜 필요한가: 평가 산출물(json 집계 · mp4)에는 그리퍼 명령이 없다. "첫 닫힘 뒤 다시
    여는가"(docs/analysis/2026-09-03_0737_multitask-for-recovery.md §5 지표 3)를 재려면
    적용된 액션이 있어야 한다. r1pro 액션 레이아웃은 right_gripper=[22:23],
    left_gripper=[14:15] 이고 그리퍼 명령은 연속값이다 (eval/r1pro.yaml: MultiFingerGripperController
    mode: smooth — [-1, 1] 을 손가락 관절 위치 범위로 선형 사상). ≥0 open / <0 close 로 나누는 것은
    분석 쪽 규약이지 컨트롤러 동작이 아니다.

    에피소드 경계는 `reset()` 이다. 마지막 에피소드 뒤에는 reset 이 없으므로 환경 `close()`
    (Evaluator.__exit__ 가 부른다)와 인터프리터 종료(atexit)에서 남은 스텝을 쓴다. SIGKILL·SIGTERM 처럼
    둘 다 못 타는 종료에 대비해 `FLUSH_EVERY` 스텝마다 같은 파일을 다시 쓰는 것도 유지한다
    (9,000스텝 × 23 float32 = 828KB).
    """

    FLUSH_EVERY = 500

    def __init__(self) -> None:
        self.dir = os.environ.get("B1K_ACTION_LOG_DIR") or None
        self.buf: list = []
        self.seq = 0
        self.instance_id = None  # watch_instance 가 잡은 값. 못 잡으면 None (0 으로 채우지 않는다)
        self.ended = False  # 마지막 step 이 terminated/truncated 를 돌려줬는가
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)
            atexit.register(self.close)
            logger.info(f"action log enabled: {self.dir}")

    def watch_instance(self, task) -> None:
        """평가 하네스가 싣는 인스턴스 id 를 잡는다.

        Evaluator.load_task_instance(instance_id) 는 TRO 상태만 덮어쓰고 task.activity_instance_id 는
        고치지 않는다 — 환경 생성 때 값 0 그대로다 (eval_utils.py:213). 그래서 그 속성을 읽던 예전
        메타는 전부 0 이었다. 하네스가 인스턴스 파일 이름을 만들 때 부르는
        task.get_cached_activity_scene_filename(..., activity_instance_id=...) (evaluator.py:285)을
        task 객체의 인스턴스 속성으로 감싸 그 인자를 기록한다. 클래스와 omnigibson/ 파일은 건드리지 않는다.

        순서는 eval.py 기준 reset(직전 에피소드 flush) → load_task_instance(여기서 갱신) → reset → step
        이라서 flush 시점의 값은 항상 그 에피소드의 인스턴스다. 같은 인스턴스를 여러 번 도는 경우
        (--num-rollouts)에는 load_task_instance 가 다시 불리지 않으므로 flush 에서 지우지 않는다.
        """
        if not self.dir or task is None or not hasattr(task, "get_cached_activity_scene_filename"):
            return
        orig = task.get_cached_activity_scene_filename

        def hooked(*args, **kwargs):
            iid = kwargs.get("activity_instance_id", args[3] if len(args) > 3 else None)
            if iid is not None:
                self.instance_id = int(iid)
            return orig(*args, **kwargs)

        task.get_cached_activity_scene_filename = hooked

    def record(self, action) -> None:
        if not self.dir:
            return
        a = action.detach().cpu().numpy() if hasattr(action, "detach") else action
        self.buf.append(np.asarray(a, dtype=np.float32).reshape(-1))
        if len(self.buf) % self.FLUSH_EVERY == 0:
            self._write(final=False)

    def _write(self, final: bool) -> None:
        if not self.dir or not self.buf:
            return
        arr = np.stack(self.buf)
        np.save(os.path.join(self.dir, f"ep{self.seq:02d}.npy"), arr)
        with open(os.path.join(self.dir, f"ep{self.seq:02d}.json"), "w") as f:
            json.dump(
                {
                    "seq": self.seq,
                    "steps": int(arr.shape[0]),
                    "action_dim": int(arr.shape[1]),
                    "instance_id": self.instance_id,
                    "complete": final,
                },
                f,
            )

    def flush(self, final: bool = True) -> None:
        """에피소드를 확정하고 다음 에피소드로 넘어간다."""
        if not self.dir or not self.buf:
            return
        self._write(final=final)
        logger.info(f"action log: ep{self.seq:02d} {len(self.buf)} steps complete={final}")
        self.buf = []
        self.ended = False
        self.seq += 1

    def close(self) -> None:
        """환경 close / 인터프리터 종료 시점. reset 이 오지 않는 마지막 에피소드를 여기서 쓴다.

        에피소드가 끝까지 갔는지는 마지막 step 의 terminated/truncated 로 판단한다. 중간에 끊겨(SIGINT 등)
        닫힌 경우에도 그때까지의 스텝은 전부 쓰되 complete 는 false 로 둔다. 두 번 불려도 된다 (buf 가 비면 반환).
        """
        self.flush(final=self.ended)


class _ActionLogMixin:
    """`_ActionLog` 를 step/reset/close 경로에 끼운다.

    evaluator.py:254 가 래퍼의 step() 을 부르므로(load_env 가 래퍼를 반환한다)
    여기서 잡는 액션이 시뮬레이터에 들어간 그 액션이다.
    """

    def _init_action_log(self) -> None:
        self._alog = _ActionLog()
        self._alog.watch_instance(getattr(self.env, "task", None))

    def step(self, action, n_render_iterations: int = 1):
        self._alog.record(action)
        out = self.env.step(action, n_render_iterations=n_render_iterations)
        self._alog.ended = bool(out[2]) or bool(out[3])
        return out

    def reset(self):
        self._alog.flush()
        return self.env.reset()

    def close(self):
        # EnvironmentWrapper 에는 close 가 없어 예전에는 __getattr__ 로 env.close() 에 바로 갔다
        self._alog.close()
        return self.env.close()


class RGBFullResWrapper(_ActionLogMixin, EnvironmentWrapper):
    """head 720x720 / wrist 480x480, RGB 전용."""

    def __init__(self, env: Environment):
        super().__init__(env=env)
        _apply_full_res(env, {"rgb"}, "RGB")
        self._init_action_log()


class RGBDFullResWrapper(_ActionLogMixin, EnvironmentWrapper):
    """head 720x720 / wrist 480x480, RGB + depth_linear.

    depth_linear 은 (H, W) float32 미터값이고 범위가 0.0~inf 다
    (vision_sensor.py:961). 8비트 3채널로 바꾸는 일은 정책 쪽
    B1KPolicyWrapper.process_input 이 한다 — 학습이 본 양자화와 맞춰야 하기 때문이다.
    """

    def __init__(self, env: Environment):
        super().__init__(env=env)
        _apply_full_res(env, {"rgb", "depth_linear"}, "RGB+depth")
        self._init_action_log()
