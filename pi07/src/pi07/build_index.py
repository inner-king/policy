#!/usr/bin/env python3
"""π0.7 재현용 B1K 인덱스 + 저차원 memmap 생성.

입력 (읽기 전용):
  data/b1k/<task>/meta/episodes/*/*.parquet   에피소드 메타 (뷰 전용 사본)
  data/b1k/<task>/data/chunk-XXX/*.parquet     action / observation.state
  data/b1k/<task>/annotations/episode_*.json   스킬 구간 주석 (task-93 은 behavior/ 아래)

출력 ($DATA/pi07/):
  index.json    에피소드 목록 — 태스크 문장, 길이, 비디오 경로·시작 시각, 세그먼트(문장·mistake), speed/quality
  lowdim.npy    float32 [전체 프레임, 48] = action 23 + state 25 (GR00T r1pro 와 같은 state 부분집합)
  stats.json    q01/q99/mean/std (학습 분할에서만 계산)

⚠️ 뷰의 meta/ 는 /data/b1k 미러로 가는 심링크일 수 있다. 이 스크립트는 뷰에 아무것도 쓰지 않는다.

메모리: parquet 파일을 하나씩 두 컬럼만 읽어 memmap 에 흘려 쓴다. 최대 RSS ~1.5 GB.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subtask_text import segment_key, subtask_sentence  # noqa: E402

CAPSTONE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(CAPSTONE, "data")

# BDDL goal 기준 영어 지시문 — src/groot/prepare_mt_views.sh 의 INSTR 과 같다 (groot-mt-200k 학습 문자열)
INSTR = {
    "can_meat": "Put the bratwurst into the jars, close them, and place the jars in the cabinet.",
    "make_microwave_popcorn": "Cook the popcorn in the microwave.",
    "cook_cabbage": "Cook the diced cabbage and chili in the frying pan.",
    "chop_an_onion": "Chop the onion and put the diced onion into the bowl.",
    "slicing_vegetables": "Slice the zucchini, bell pepper, and beet.",
    "cook_hot_dogs": "Cook the hot dogs.",
    "cook_bacon": "Cook the bacon in the frying pan.",
    "freeze_pies": "Put the apple pies into the containers and place them in the freezer.",
    "canning_food": "Dice the steak and pineapple, and put each into a separate bowl.",
    "make_pizza": "Make the pizza and place it on the baking sheet.",
    "freeze_fruit": "Put the apples and strawberries into the containers and place them in the freezer.",
    "cook_a_brisket": "Cook the brisket in the oven.",
    "halve_an_egg": "Cut the hard-boiled egg in half.",
    "thawing_frozen_food": "Thaw the frozen chicken breast.",
    "cook_a_frozen_pie": "Cook the frozen apple pie on the tray.",
    "cook_brussels_sprouts": "Halve the brussels sprouts and cook them in the oven.",
    "cook_broccolini": "Cook the broccolini and garlic in the frying pan on the stove.",
    # clean_a_keyboard_en/meta/tasks.parquet row 93 (groot-radioinit-h40-200k 이후 학습 문자열)
    "clean_a_keyboard": "Push the chair under the desk, pick up the duster brush, and wipe the dust off the keyboard with the brush.",
}
TASK93_ANN = os.path.join(CAPSTONE, "behavior", "task-0093-clean_a_keyboard", "annotations")

CAMERAS = {  # π0.7 뷰 이름 → B1K 비디오 키
    "head": "observation.rgb.zed_link_camera_0",
    "left_wrist": "observation.rgb.left_realsense_link_camera_0",
    "right_wrist": "observation.rgb.right_realsense_link_camera_0",
}
# observation.state(61) 에서 쓰는 25개 — groot modality.json 과 같은 순서
STATE_SLICES = [(0, 3), (53, 57), (3, 10), (24, 26), (28, 35), (49, 51)]
STATE_DIM = sum(b - a for a, b in STATE_SLICES)  # 25
ACTION_DIM = 23
FPS = 30
SPEED_BIN = 500  # π0.7 V-C: 길이를 500 스텝 단위로 반올림


def speed_bin(length: int) -> int:
    return int(round(length / SPEED_BIN) * SPEED_BIN)


# π0.7 의 quality(1~5)·mistake 는 사람이 붙인 라벨이다. B1K 데모에는 없다.
# 연속 반복 세그먼트(같은 스킬·객체·공간 수식어)를 재시도로 보고 mistake 를 유도해 봤으나,
# 1,601건 중 1,596건이 freeze_fruit(200/200 에피소드)·halve_an_egg(200/200)에서 나왔고
# 스킬 하나를 동일 라벨 두 조각으로 나눈 주석 관행이었다 — 실패가 아니다. 그래서 상수로 둔다.
QUALITY = 5


def load_segments(ann_path: str, length: int) -> tuple[list[dict], int]:
    d = json.load(open(ann_path))
    raw = []
    seen = set()
    for s in d["skill_annotation"]:
        fd = s["frame_duration"]
        pieces = fd if isinstance(fd[0], list) else [fd]  # task-0004 에 두 조각짜리 83개
        for st, en in pieces:
            st, en = int(st), min(int(en), length)  # 34개 에피소드가 길이를 넘는다 (최대 784 프레임)
            if en <= st:
                continue
            key = (st, en) + segment_key(s)
            if key in seen:
                continue
            seen.add(key)
            raw.append((st, en, s))
    raw.sort(key=lambda z: (z[0], z[1]))
    segs = []
    for i, (st, en, s) in enumerate(raw):
        nxt = raw[i + 1][2] if i + 1 < len(raw) else None
        repeat = bool(nxt is not None and segment_key(nxt) == segment_key(s))
        segs.append(dict(start=st, end=en, text=subtask_sentence(s), skill=s["skill_description"][0],
                         repeat_next=repeat, mistake=False))
    n_repeat = sum(g["repeat_next"] for g in segs)
    return segs, n_repeat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "pi07"))
    ap.add_argument("--val-per-task", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    manifest = json.load(open(os.path.join(DATA, "b1k", "manifest.json")))
    tasks = [(t["task_name"], t["task_index"]) for t in manifest["tasks"]] + [("clean_a_keyboard", 93)]

    episodes = []
    for name, tidx in tasks:
        root = os.path.join(DATA, "b1k", name)
        files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "*", "*.parquet")))
        cols = ["episode_index", "length", "data/chunk_index", "data/file_index", "dataset_from_index",
                "task_index", "annotation_path", "task_instance_id"]
        for key in CAMERAS.values():
            cols += [f"videos/{key}/chunk_index", f"videos/{key}/file_index", f"videos/{key}/from_timestamp"]
        rows = []
        for f in files:
            rows += pq.read_table(f, columns=cols).to_pylist()
        rows = [r for r in rows if r["task_index"] == tidx]
        assert len(rows) == 200, (name, len(rows))
        ann_dir = TASK93_ANN if tidx == 93 else os.path.join(root, "annotations")
        for r in sorted(rows, key=lambda z: z["episode_index"]):
            L = int(r["length"])
            ann = os.path.join(ann_dir, os.path.basename(r["annotation_path"]))
            segs, n_repeat = load_segments(ann, L)
            videos = {}
            for cam, key in CAMERAS.items():
                ci, fi = r[f"videos/{key}/chunk_index"], r[f"videos/{key}/file_index"]
                path = os.path.join(root, "videos", key, f"chunk-{ci:03d}", f"file-{fi:03d}.mp4")
                videos[cam] = dict(path=path, from_ts=float(r[f"videos/{key}/from_timestamp"]))
            episodes.append(dict(
                task=name, task_index=tidx, instruction=INSTR[name], episode_index=int(r["episode_index"]),
                instance=int(r["task_instance_id"]), length=L,
                data_path=os.path.join(root, "data", f"chunk-{r['data/chunk_index']:03d}",
                                       f"file-{r['data/file_index']:03d}.parquet"),
                dataset_from_index=int(r["dataset_from_index"]),
                videos=videos, segments=segs, n_repeat=n_repeat,
                speed=speed_bin(L), quality=QUALITY))
        print(f"{name:24s} task={tidx:3d} eps={len(rows)} frames={sum(int(r['length']) for r in rows)}", flush=True)

    # 분할: 태스크마다 무작위 val_per_task 개를 검증용으로 뺀다
    rng = np.random.default_rng(args.seed)
    by_task = {}
    for i, e in enumerate(episodes):
        by_task.setdefault(e["task"], []).append(i)
    for t, idx in by_task.items():
        val = set(rng.choice(idx, size=args.val_per_task, replace=False).tolist())
        for i in idx:
            episodes[i]["split"] = "val" if i in val else "train"

    # 저차원 memmap — 에피소드 순서대로 이어 붙인다
    off = 0
    for e in episodes:
        e["offset"] = off
        off += e["length"]
    total = off
    lowdim_path = os.path.join(args.out, "lowdim.npy")
    mm = np.lib.format.open_memmap(lowdim_path, mode="w+", dtype=np.float32, shape=(total, ACTION_DIM + STATE_DIM))
    filled = np.zeros(len(episodes), dtype=bool)
    by_file = {}
    for i, e in enumerate(episodes):
        by_file.setdefault(e["data_path"], []).append(i)
    t0 = time.time()
    for k, (path, idxs) in enumerate(sorted(by_file.items())):
        tb = pq.read_table(path, columns=["episode_index", "frame_index", "action", "observation.state"])
        ep = tb.column("episode_index").to_numpy()
        fr = tb.column("frame_index").to_numpy()
        act = np.asarray(tb.column("action").combine_chunks().flatten(), dtype=np.float32).reshape(len(ep), -1)
        st = np.asarray(tb.column("observation.state").combine_chunks().flatten(),
                        dtype=np.float32).reshape(len(ep), -1)
        assert act.shape[1] == ACTION_DIM and st.shape[1] == 61, (act.shape, st.shape)
        st = np.concatenate([st[:, a:b] for a, b in STATE_SLICES], axis=1)
        del tb
        for i in idxs:
            e = episodes[i]
            sel = np.nonzero(ep == e["episode_index"])[0]
            assert len(sel) == e["length"], (path, e["episode_index"], len(sel), e["length"])
            order = sel[np.argsort(fr[sel])]
            assert (fr[order] == np.arange(e["length"])).all()
            mm[e["offset"]:e["offset"] + e["length"], :ACTION_DIM] = act[order]
            mm[e["offset"]:e["offset"] + e["length"], ACTION_DIM:] = st[order]
            filled[i] = True
        del act, st, ep, fr
        print(f"  lowdim {k + 1}/{len(by_file)} {os.path.basename(os.path.dirname(path))}/"
              f"{os.path.basename(path)} {time.time() - t0:.0f}s", flush=True)
    mm.flush()
    assert filled.all(), f"lowdim 미기록 에피소드 {(~filled).sum()}"

    # 정규화 통계 — 학습 분할에서 균등 표본 2M 프레임
    tr_rows = np.concatenate([np.arange(e["offset"], e["offset"] + e["length"]) for e in episodes
                              if e["split"] == "train"])
    samp = np.sort(rng.choice(tr_rows, size=min(2_000_000, len(tr_rows)), replace=False))
    x = np.asarray(mm[samp])
    stats = {}
    for name, sl in (("action", slice(0, ACTION_DIM)), ("state", slice(ACTION_DIM, None))):
        v = x[:, sl]
        stats[name] = dict(q01=np.quantile(v, 0.01, axis=0).tolist(), q99=np.quantile(v, 0.99, axis=0).tolist(),
                           mean=v.mean(0).tolist(), std=v.std(0).tolist(),
                           min=v.min(0).tolist(), max=v.max(0).tolist())
    json.dump(stats, open(os.path.join(args.out, "stats.json"), "w"), indent=1)

    meta = dict(fps=FPS, cameras=CAMERAS, state_slices=STATE_SLICES, action_dim=ACTION_DIM, state_dim=STATE_DIM,
                total_frames=total, speed_bin=SPEED_BIN, episodes=episodes)
    json.dump(meta, open(os.path.join(args.out, "index.json"), "w"))
    n_tr = sum(e["split"] == "train" for e in episodes)
    print(f"episodes={len(episodes)} (train {n_tr}, val {len(episodes) - n_tr}) frames={total} "
          f"segments={sum(len(e['segments']) for e in episodes)} repeats={sum(e['n_repeat'] for e in episodes)}")


if __name__ == "__main__":
    main()
