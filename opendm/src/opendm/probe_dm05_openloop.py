"""DM0.5 base 청크를 B1K 데모 한 에피소드에 대고 개루프로 비교한다.

앵커 프레임 t = start + j·stride 마다 데모 관측(카메라 3개 + proprio)으로 DM0.5 에 청크를 한 번
요청하고, serve_dm05_b1k.py 와 같은 매핑으로 23차원으로 바꾼 청크의 앞 --exec 개를 정답
actions[t:t+exec] 와 비교한다. t + exec 가 에피소드 길이를 넘는 앵커는 뺀다.
기준선은 "현재 자세 유지"(hold) 다: 팔은 앵커 시점 qpos, 그리퍼는 앵커 시점 개방도, 나머지는 정답 그대로.

지표 (ARM = DM0.5 가 실제로 움직이는 팔 관절 12개, 23차원 인덱스 7..12 · 15..20.
13·21 은 브리지가 고정하므로 뺀다. GRIP = 14·22):
    arm/grip MAE     pred−gt, hold−gt 각각
    pred_motion      mean|pred − hold| (ARM) — DM0.5 가 현재 자세에서 얼마나 움직이라고 하는가
    gt_motion        mean|gt − hold| (ARM)   — 데모가 같은 구간에서 실제로 움직인 양
    latency_ms       HTTP 요청 왕복 시간

⚠️ DM0.5 base 는 R1Pro 를 학습한 적이 없다 (DOS W1·Aloha, 팔 6관절). 매핑도 임의로 정했다
(serve_dm05_b1k.py 모듈 docstring 참고). 여기서 나오는 수치는 배관과 크기 확인용 — 요청이 오가는가,
출력이 관절 범위 안에서 현재 자세 근처에 있는가 — 이지 정책 품질 지표가 아니다.

데이터는 task0093-rgb 루트를 쓴다. data/b1k/clean_a_keyboard_en 뷰는 에피소드 인덱스를 위치로
해석해서 18600 을 못 찾는다.

실행 (DM0.5 서비스가 7891 에 떠 있어야 한다)::

    vla/.venv/bin/python src/opendm/probe_dm05_openloop.py --n-anchors 20 \\
        --out runs/eval/dm05base-b1k-smoke/openloop.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serve_dm05_b1k as bridge  # noqa: E402

CAPSTONE = Path(__file__).resolve().parents[2]

# ds[t] 카메라 키 → DM0.5 images 슬롯 (head / 왼손목 / 오른손목)
CAMERA_KEYS = {
    "observation.rgb.zed_link_camera_0": "1",
    "observation.rgb.left_realsense_link_camera_0": "2",
    "observation.rgb.right_realsense_link_camera_0": "3",
}

# 23차원 액션 인덱스
ARM = list(range(7, 13)) + list(range(15, 21))   # DM0.5 가 구동하는 팔 12관절 (13·21 은 고정)
GRIP = [14, 22]

DEFAULT_PROMPT = ("Push the chair under the desk, pick up the duster brush, "
                  "and wipe the dust off the keyboard with the brush.")


def to_hwc_uint8(x: torch.Tensor) -> np.ndarray:
    """(3,H,W) float [0,1] → (H,W,3) uint8."""
    return (x * 255).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()


def gripper_cmd(finger_qpos: np.ndarray) -> float:
    """R1Pro 손가락 qpos(2,) → 23차원 그리퍼 명령 [-1, 1] (+1 완전 개방)."""
    v = float(np.mean(np.asarray(finger_qpos, dtype=np.float64)))
    return float(np.clip(v / bridge.R1PRO_FINGER_MAX, 0.0, 1.0) * 2.0 - 1.0)


def hold_baseline(gt: np.ndarray, proprio: np.ndarray) -> np.ndarray:
    """현재 자세 유지 기준선 (exec,23). 팔·그리퍼만 앵커 시점 값으로 덮고 나머지는 정답."""
    hold = gt.copy()
    hold[:, 7:14] = proprio[bridge.ARM_LEFT_QPOS]
    hold[:, 15:22] = proprio[bridge.ARM_RIGHT_QPOS]
    hold[:, 14] = gripper_cmd(proprio[bridge.GRIPPER_LEFT_QPOS])
    hold[:, 22] = gripper_cmd(proprio[bridge.GRIPPER_RIGHT_QPOS])
    return hold


def mae(a: np.ndarray, b: np.ndarray, idx: list[int]) -> float:
    return float(np.mean(np.abs(a[:, idx] - b[:, idx])))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=CAPSTONE / "data" / "smolvla" / "task0093-rgb")
    parser.add_argument("--repo-id", default="behavior-1k/2026-challenge-demos")
    parser.add_argument("--episode", type=int, default=18600)
    parser.add_argument("--start", type=int, default=0, help="첫 앵커 프레임")
    parser.add_argument("--stride", type=int, default=200, help="앵커 간격 (프레임)")
    parser.add_argument("--n-anchors", type=int, default=20)
    parser.add_argument("--exec", dest="exec_steps", type=int, default=25,
                        help="청크 앞쪽 몇 개를 비교할지 (serve_dm05_b1k --exec-steps 와 같은 뜻)")
    parser.add_argument("--dm05-url", default="http://127.0.0.1:7891/v1/infer")
    parser.add_argument("--robot-type", default="DOS W1")
    parser.add_argument("--control-mode", default="joint")
    parser.add_argument("--speed", default="0.5", help="OpenDM 기본값과 같은 문자열")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--http-timeout", type=float, default=60.0)
    parser.add_argument("--out", type=Path, default=None, help="결과 JSON 경로 (선택)")
    args = parser.parse_args()

    if not 1 <= args.exec_steps <= bridge.DM_CHUNK:
        parser.error(f"--exec 범위: 1..{bridge.DM_CHUNK} (받은 값 {args.exec_steps})")
    if args.stride < 1:
        parser.error(f"--stride 는 1 이상 (받은 값 {args.stride})")
    if args.n_anchors < 1:
        parser.error(f"--n-anchors 는 1 이상 (받은 값 {args.n_anchors})")
    if args.start < 0:
        parser.error(f"--start 는 0 이상 (받은 값 {args.start})")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    t0 = time.monotonic()
    ds = LeRobotDataset(args.repo_id, root=str(args.root),
                        episodes=[args.episode], tolerance_s=0.01)
    actions = np.stack([np.asarray(a) for a in ds.hf_dataset["action"]]).astype(np.float32)
    T = len(ds)
    assert actions.shape == (T, bridge.ACTION_DIM), f"actions {actions.shape} != ({T}, {bridge.ACTION_DIM})"
    print(f"에피소드 {args.episode}: {T} 프레임, 로드 {time.monotonic() - t0:.1f} s")

    anchors = [args.start + j * args.stride for j in range(args.n_anchors)]
    anchors = [t for t in anchors if t + args.exec_steps <= T]
    if not anchors:
        sys.exit(f"유효한 앵커 없음: start={args.start}, stride={args.stride}, "
                 f"exec={args.exec_steps}, T={T}")
    print(f"앵커 {len(anchors)}개 (요청 {args.n_anchors}), exec {args.exec_steps}, "
          f"DM0.5 {args.dm05_url}")

    client = bridge.DM05Client(
        url=args.dm05_url,
        robot_type=args.robot_type,
        control_mode=args.control_mode,
        speed=args.speed,
        jpeg_quality=args.jpeg_quality,
        timeout=args.http_timeout,
    )

    print(f"{'t':>6} {'lat_ms':>7} {'arm_dm':>8} {'arm_hold':>8} "
          f"{'grip_dm':>8} {'grip_hold':>9} {'pred_mot':>8} {'gt_mot':>8}")
    records = []
    for t in anchors:
        s = ds[t]
        images = {slot: bridge._encode_jpeg(to_hwc_uint8(s[key]), args.jpeg_quality)
                  for key, slot in CAMERA_KEYS.items()}
        proprio = s["observation.state"].numpy().astype(np.float32).reshape(-1)
        state14 = bridge.build_state14(proprio)

        t_req = time.monotonic()
        actions14, meta = client.infer(args.prompt, state14, images)
        latency_ms = (time.monotonic() - t_req) * 1000.0
        if actions14.shape[0] < args.exec_steps:
            raise RuntimeError(
                f"t={t}: DM0.5 청크가 --exec 보다 짧음: {actions14.shape[0]} < {args.exec_steps}")

        pred = np.stack([bridge.dm14_to_r1pro23(a, proprio)
                         for a in actions14[: args.exec_steps]], axis=0)
        gt = actions[t: t + args.exec_steps]
        hold = hold_baseline(gt, proprio)

        rec = {
            "t": t,
            "latency_ms": latency_ms,
            "n_chunk": int(actions14.shape[0]),
            "arm_mae_dm05": mae(pred, gt, ARM),
            "arm_mae_hold": mae(hold, gt, ARM),
            "grip_mae_dm05": mae(pred, gt, GRIP),
            "grip_mae_hold": mae(hold, gt, GRIP),
            "pred_motion": mae(pred, hold, ARM),
            "gt_motion": mae(gt, hold, ARM),
            "metadata": meta,
        }
        records.append(rec)
        print(f"{t:>6} {latency_ms:>7.0f} {rec['arm_mae_dm05']:>8.4f} {rec['arm_mae_hold']:>8.4f} "
              f"{rec['grip_mae_dm05']:>8.4f} {rec['grip_mae_hold']:>9.4f} "
              f"{rec['pred_motion']:>8.4f} {rec['gt_motion']:>8.4f}")

    def avg(key: str) -> float:
        return float(np.mean([r[key] for r in records]))

    summary = {
        "n_anchors": len(records),
        "n_frames": len(records) * args.exec_steps,
        "arm_mae_dm05": avg("arm_mae_dm05"),
        "arm_mae_hold": avg("arm_mae_hold"),
        "grip_mae_dm05": avg("grip_mae_dm05"),
        "grip_mae_hold": avg("grip_mae_hold"),
        "pred_motion": avg("pred_motion"),
        "gt_motion": avg("gt_motion"),
        "latency_ms_mean": avg("latency_ms"),
    }
    summary["arm_mae_ratio_dm05_over_hold"] = (
        summary["arm_mae_dm05"] / summary["arm_mae_hold"] if summary["arm_mae_hold"] > 0 else None)

    print()
    print(f"요약: 앵커 {summary['n_anchors']}개, 비교 프레임 {summary['n_frames']}개, "
          f"평균 지연 {summary['latency_ms_mean']:.0f} ms")
    print(f"  {'MAE':<6}{'dm05':>9}{'hold':>9}")
    print(f"  {'arm':<6}{summary['arm_mae_dm05']:>9.4f}{summary['arm_mae_hold']:>9.4f}")
    print(f"  {'grip':<6}{summary['grip_mae_dm05']:>9.4f}{summary['grip_mae_hold']:>9.4f}")
    print(f"  pred_motion {summary['pred_motion']:.4f}   gt_motion {summary['gt_motion']:.4f}")
    ratio = summary["arm_mae_ratio_dm05_over_hold"]
    if ratio is None:
        print("  arm MAE dm05/hold: 계산 불가 (hold MAE = 0)")
    else:
        print(f"  arm MAE dm05/hold: {ratio:.3f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "records": records,
            "summary": summary,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
        print(f"저장: {args.out}")


def _json_default(obj):
    """json.dump 가 모르는 numpy 값 변환 (DM0.5 metadata 에 섞여 올 수 있다)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"JSON 직렬화 불가: {type(obj).__name__}")


if __name__ == "__main__":
    main()
