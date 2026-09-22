"""B1K mp4 에서 임의 프레임 디코딩 (PyAV).

B1K 비디오는 HEVC, GOP 8, B-프레임 없음이라 키프레임 탐색 후 최대 7장만 디코딩하면 된다.
LeRobot v3.0 은 에피소드 여러 개를 한 파일에 이어 붙이므로 프레임 t 의 시각 = from_ts + t / fps.

메모리 방침 (groot-mt-200k 의 워커 OOM 두 원인에 대한 대응):
- 디코더 스레드를 명시적으로 준다 (기본 0 = 코어 수 → 디코더당 16 스레드였다).
- 열어 둔 컨테이너 수를 LRU 로 제한한다. 캐시가 아무것도 쫓아내지 않던 것이 첫 번째 원인이었다.
"""
from __future__ import annotations

import collections
import os

import av
import numpy as np

_THREADS = int(os.environ.get("PI07_FFMPEG_THREADS", "1"))
_MAX_OPEN = int(os.environ.get("PI07_MAX_OPEN_VIDEOS", "6"))


class FrameReader:
    """워커 프로세스마다 하나. 열린 컨테이너를 최대 _MAX_OPEN 개 유지한다."""

    def __init__(self, fps: int = 30, max_open: int = _MAX_OPEN, threads: int = _THREADS):
        self.fps = fps
        self.max_open = max_open
        self.threads = threads
        self._open: collections.OrderedDict[str, av.container.InputContainer] = collections.OrderedDict()

    def _container(self, path: str):
        c = self._open.pop(path, None)
        if c is None:
            c = av.open(path)
            s = c.streams.video[0]
            s.thread_type = "SLICE" if self.threads > 1 else "NONE"
            s.thread_count = self.threads
            while len(self._open) >= self.max_open:
                _, old = self._open.popitem(last=False)
                old.close()
        self._open[path] = c
        return c

    def close(self):
        for c in self._open.values():
            c.close()
        self._open.clear()

    def read(self, path: str, ts: float, size: int | None = None) -> np.ndarray:
        """시각 ts(초)에 해당하는 프레임을 HWC uint8 RGB 로. size 가 있으면 정사각 리사이즈."""
        c = self._container(path)
        s = c.streams.video[0]
        tb = float(s.time_base)
        target = int(round(ts / tb))
        half = int(round(0.5 / self.fps / tb))
        c.seek(target, stream=s, backward=True, any_frame=False)
        frame = None
        for f in c.decode(s):
            frame = f
            if f.pts is not None and f.pts >= target - half:
                break
        if frame is None:
            raise RuntimeError(f"decode failed: {path} @ {ts:.3f}s")
        if size is not None and (frame.width != size or frame.height != size):
            frame = frame.reformat(width=size, height=size, format="rgb24", interpolation="AREA")
            return frame.to_ndarray()
        return frame.to_ndarray(format="rgb24")
