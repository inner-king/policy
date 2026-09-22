#!/usr/bin/env python3
"""원격 시뮬레이터(super2)용 평가 진입점 — `python -m omnigibson.eval.eval` 과 같은 인자를 받는다.

super2 ↔ 이 머신 회선이 2.0 MB/s (2026-09-19 실측, Tailscale direct) 라서 전체 해상도 관측
(머리 720² + 손목 480² × 2, RGBA ≈ 3.9 MB/스텝) 을 그대로 보내면 스텝마다 약 2초가 더 든다.
정책 클라이언트(WebsocketPolicy.forward)만 바꿔 치워, 보내기 직전에 RGB 를 224×224 로 면적 평균
축소하고 JPEG(품질 90)으로 줄인다. 깊이·분할 채널은 정책이 안 쓰므로 뺀다.
평가 하네스 파일은 건드리지 않는다. 영상 기록(evaluator._write_video)은 원본 관측을 그대로 쓴다.

serve_pi07.py 의 Pi07Runtime._img 가 1차원 uint8 배열을 JPEG 으로 보고 복호화한다.
"""
from __future__ import annotations

import runpy
import sys

import cv2
import numpy as np

import omnigibson.eval.policies as _pol

JPEG_QUALITY = 90
SIZE = 224
_orig_forward = _pol.WebsocketPolicy.forward


def _to_numpy(v):
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v)


def _compress(obs: dict) -> dict:
    out = {}
    for k, v in obs.items():
        if isinstance(k, str) and k.endswith("::rgb"):
            a = _to_numpy(v)[..., :3]
            if a.dtype != np.uint8:
                a = np.clip(a * (255.0 if a.max() <= 1.5 else 1.0), 0, 255).astype(np.uint8)
            if a.shape[0] != SIZE or a.shape[1] != SIZE:
                a = cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(a[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            assert ok
            out[k] = buf.reshape(-1).astype(np.uint8)
        elif isinstance(k, str) and ("::depth" in k or "::seg" in k):
            continue
        else:
            out[k] = v
    return out


def forward(self, obs: dict, *args, **kwargs):
    if "need_new_action" in obs and not obs["need_new_action"] and self.last_action is not None:
        return self.last_action
    return _orig_forward(self, _compress(obs), *args, **kwargs)


_pol.WebsocketPolicy.forward = forward

if __name__ == "__main__":
    sys.argv = ["omnigibson.eval.eval"] + sys.argv[1:]
    runpy.run_module("omnigibson.eval.eval", run_name="__main__", alter_sys=True)
