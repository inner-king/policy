"""변환된 LeRobot 데이터셋의 무결성 점검 (업로드 전 게이트).

    uv run --project third_party/openpi python scripts/verify_dataset.py
    uv run --project third_party/openpi python scripts/verify_dataset.py --video-sample 300

## 데이터를 읽지 않고 전수 검사한다

에피소드가 6.7만 개라 parquet을 실제로 읽으면 시간이 오래 걸린다. 하지만
pyarrow는 컬럼별 min/max 통계를 **푸터에** 넣어 두므로, 푸터만 읽으면
행 수와 `index`/`episode_index`/`task_index` 의 범위를 전부 알 수 있다.
그래서 전수 검사가 파일당 수백 마이크로초로 끝난다.

`index` 는 build_dataset이 `np.arange(n_fr, n_fr+n)` 로 쓰므로 에피소드 안에서는
연속이 보장된다. 확인해야 하는 것은 **에피소드 경계에서 이어지는가**이고,
그건 min/max로 충분하다.

## 영상은 표본만 본다

mp4 프레임 수 확인은 ffprobe 프로세스를 20만 번 띄우는 일이라 전수는 비현실적이다.
개수(= 에피소드 x 3)는 전수로 세고, 프레임 수 일치는 `--video-sample` 개
에피소드를 무작위로 뽑아 확인한다. 프레임 정렬 자체는 build 단계에서
탐색 없는 순차 디코드로 구조적으로 보장되고 기준 프레임 MAE로 이미 검증했다.
"""
import argparse, json, os, random, subprocess, sys
from collections import Counter

import pyarrow.parquet as pq

# 데이터를 만든 기기의 저장소 루트다. 데이터셋 자체는 `--root` 로 받으므로
# 이 상수에 걸리는 것은 빌드 산출물(failed_segments.jsonl) 하나뿐이고,
# 그건 빌드하지 않은 환경(Colab)에는 애초에 없다 — 아래 참조.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 이 파일의 두 단계 위 (공개하며 절대경로를 바꿈)
OUT = f"{ROOT}/data/lerobot/cooking-vla/behavior-cooking-224"
CAMS = ("image", "left_wrist_image", "right_wrist_image")
CHUNK = 1000

fails = []


def chk(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)
    return cond


def nframes(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-sample", type=int, default=200,
                    help="mp4 프레임 수를 확인할 에피소드 표본 수 (0이면 생략)")
    ap.add_argument("--root", default=OUT)
    a = ap.parse_args()
    root = a.root

    info = json.load(open(f"{root}/meta/info.json"))
    eps = [json.loads(l) for l in open(f"{root}/meta/episodes.jsonl")]
    tasks = [json.loads(l) for l in open(f"{root}/meta/tasks.jsonl")]
    nstats = sum(1 for _ in open(f"{root}/meta/episodes_stats.jsonl"))
    n = len(eps)
    total_len = sum(e["length"] for e in eps)

    print(f"\n=== meta ===  에피소드 {n:,} / 프레임 {total_len:,} / 지시문 {len(tasks)}종")
    chk([e["episode_index"] for e in eps] == list(range(n)), "episode_index 0..N-1 연속")
    chk(nstats == n, f"episodes_stats {nstats:,} == 에피소드 {n:,}")
    chk([t["task_index"] for t in tasks] == list(range(len(tasks))), "task_index 0..T-1 연속")
    chk(info["total_episodes"] == n, f"info.total_episodes {info['total_episodes']:,} == {n:,}")
    chk(info["total_frames"] == total_len, f"info.total_frames {info['total_frames']:,} == {total_len:,}")
    chk(info["total_tasks"] == len(tasks), f"info.total_tasks {info['total_tasks']} == {len(tasks)}")
    chk(info["total_videos"] == n * 3, f"info.total_videos {info['total_videos']:,} == {n*3:,}")
    chk(info["total_chunks"] == max(1, (n + CHUNK - 1) // CHUNK), "info.total_chunks 일치")
    chk(info["splits"] == {"train": f"0:{n}"}, "info.splits 일치")
    for c in CAMS:
        f = info["features"][c]
        chk(f.get("info", {}).get("video.height") == 224
            and f.get("info", {}).get("video.width") == 224, f"{c} 224x224")

    tidx = {t["task"]: t["task_index"] for t in tasks}

    print(f"\n=== parquet 전수 (푸터) ===")
    stat_of = lambda rg, name: next(
        rg.column(i).statistics for i in range(rg.num_columns)
        if rg.column(i).path_in_schema == name)
    off, bad, seen_tasks = 0, [], Counter()
    for e in eps:
        i = e["episode_index"]
        p = f"{root}/data/chunk-{i//CHUNK:03d}/episode_{i:06d}.parquet"
        if not os.path.exists(p):
            bad.append((i, "파일 없음")); off += e["length"]; continue
        m = pq.read_metadata(p)
        rg = m.row_group(0)
        if m.num_rows != e["length"]:
            bad.append((i, f"행 {m.num_rows} != length {e['length']}"))
        if m.num_row_groups != 1:
            bad.append((i, f"row_group {m.num_row_groups}개 — 통계 검사 불완전"))
        s = stat_of(rg, "index")
        if s.min != off or s.max != off + e["length"] - 1:
            bad.append((i, f"index [{s.min},{s.max}] != [{off},{off+e['length']-1}]"))
        s = stat_of(rg, "episode_index")
        if s.min != i or s.max != i:
            bad.append((i, f"episode_index [{s.min},{s.max}] != {i}"))
        s = stat_of(rg, "task_index")
        want = tidx.get(e["tasks"][0])
        if s.min != s.max or s.min != want:
            bad.append((i, f"task_index [{s.min},{s.max}] != {want}"))
        seen_tasks[s.min] += 1
        off += e["length"]
        if i % 5000 == 0:
            print(f"    {i:,}/{n:,}", flush=True)
    chk(not bad, f"parquet {n:,}개 행수/index/episode_index/task_index 정합")
    for i, m in bad[:20]:
        print(f"        ep{i}: {m}")
    if len(bad) > 20:
        print(f"        ... 외 {len(bad)-20}건")
    chk(off == total_len, f"index 전역 연속 0..{total_len-1:,}")
    unused = [t["task_index"] for t in tasks if t["task_index"] not in seen_tasks]
    chk(not unused, f"쓰이지 않는 task_index 없음 ({len(unused)}개)")

    print(f"\n=== 파일 개수 ===")
    npq = sum(len([f for f in os.listdir(f"{root}/data/{d}") if f.endswith(".parquet")])
              for d in sorted(os.listdir(f"{root}/data")))
    chk(npq == n, f"parquet {npq:,} == 에피소드 {n:,}")
    for c in CAMS:
        nv = sum(len(os.listdir(f"{root}/videos/{d}/{c}"))
                 for d in sorted(os.listdir(f"{root}/videos"))
                 if os.path.isdir(f"{root}/videos/{d}/{c}"))
        chk(nv == n, f"{c} mp4 {nv:,} == 에피소드 {n:,}")

    # 이 파일은 build_dataset.py 가 남기는 것이라 **빌드한 기기에만** 있다.
    # 없을 때 0건으로 세어 `ok` 를 찍으면, 검사한 적이 없는 것을 통과로
    # 오해하게 된다 (Colab 에서 실제로 그렇게 찍히고 있었다).
    # 그래서 없으면 건너뛴 사실을 그대로 밝힌다.
    fs = f"{ROOT}/data/failed_segments.jsonl"
    if os.path.exists(fs):
        nf = sum(1 for _ in open(fs))
        chk(nf == 0, f"실패 세그먼트 {nf}건")
    else:
        print(f"  건너뜀 실패 세그먼트 — {fs} 없음 (빌드한 기기에서만 확인된다)")

    if a.video_sample:
        print(f"\n=== mp4 프레임 수 표본 {a.video_sample}개 ===")
        random.seed(0)
        vbad = []
        for k, e in enumerate(random.sample(eps, min(a.video_sample, n))):
            i = e["episode_index"]
            for c in CAMS:
                got = nframes(f"{root}/videos/chunk-{i//CHUNK:03d}/{c}/episode_{i:06d}.mp4")
                if got != e["length"]:
                    vbad.append((i, c, got, e["length"]))
            if (k + 1) % 50 == 0:
                print(f"    {k+1}/{a.video_sample}", flush=True)
        chk(not vbad, f"표본 mp4 프레임 수 == parquet 행 수")
        for i, c, g, w in vbad[:20]:
            print(f"        ep{i} {c}: {g} != {w}")

    print("\n" + ("모든 점검 통과 — 업로드 가능" if not fails
                  else f"실패 {len(fails)}건:\n  - " + "\n  - ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
