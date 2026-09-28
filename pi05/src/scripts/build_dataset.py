"""BEHAVIOR-1K 요리 세그먼트 -> openpi 학습용 LeRobot 데이터셋 (v2.1 포맷).

태스크 단위로 [다운로드 -> 224p 추출 -> 원본 삭제]를 반복해 피크 디스크를
27.5GB 이하로 억제한다(태스크 간 파일 공유가 없어 재다운로드도 없다).
세그먼트 하나가 출력 데이터셋의 에피소드 하나가 되고, 지시문이 task로 들어간다.

  uv run --project third_party/openpi python scripts/build_dataset.py
  uv run --project third_party/openpi python scripts/build_dataset.py --tasks 40 --max-episodes 2

## 영상 처리는 전부 ffmpeg에 맡긴다

이 파이프라인이 하는 일의 본질은 8천만 장(26.6M 프레임 x 카메라 3대)을
720p hevc -> 224p h264로 트랜스코딩하는 것이고, 전체 소요 시간은 거의
전적으로 여기서 결정된다. 그래서 프레임이 파이썬을 통과하지 않게 하는 것이
유일하게 중요한 설계 원칙이다.

거쳐 온 두 번의 낭비 (같은 실수의 다른 형태다):

1. lerobot의 `add_frame()` 은 프레임마다 카메라별 PNG를 디스크에 쓰고
   `save_episode()` 가 그 디렉터리를 ffmpeg으로 다시 읽는다.
   PNG 쓰기가 84 f/s로 벽시계의 88%를 차지했다 (디코드 508, ffmpeg 540 f/s).
2. PNG를 없애고 torchcodec 디코드 -> torch 리사이즈 -> ffmpeg stdin으로 바꾸니
   196 f/s가 됐지만, 이번엔 torch 리사이즈가 프레임당 1.54ms를 먹었다.

지금은 디코드/리사이즈/인코드를 전부 ffmpeg이 하고 파이썬은 바이트를
세그먼트별 인코더로 배분만 한다. 동시 실행 종합 처리량 실측(카메라-f/s):

    프로세스 수            4      8     12
    순수 ffmpeg (CPU)   1172   1983   2332   <- 채택
    NVDEC + CPU 인코드  1261   1638      -

NVDEC(GPU 디코드)는 12프로세스 CPU보다 느리고, NVENC는 같은 화질에서 파일이
2배(1,461 vs 727 B/frame) 커진다. **GPU는 쓰지 않는다.**

리사이즈를 swscale에 맡긴 부수 효과: torch의 `interpolate(mode="bilinear")` 는
antialias를 켜지 않으면 3.2배 축소에서 에일리어싱이 생기는데, swscale의
bicubic은 축소 배율에 맞춰 커널을 늘리므로 이 문제가 없다.

## 탐색(seek)이 없다

`from_timestamp` 는 에피소드마다 하나(파일 내 시작 오프셋)이고 에피소드는
전부 단일 파일 안에 있음을 확인했다(3000/3000). 게다가 한 파일에서 쓰는 프레임
범위가 사실상 파일 전체다(중앙값 73,329 / 파일 길이 72,576).

그래서 작업 단위는 에피소드가 아니라 **(카메라, 소스 파일)** 이다. 파일을
처음부터 한 번만 순차로 읽으면서 구간에 든 프레임만 세그먼트별 인코더로
흘려보낸다. 탐색이 없으니 프레임이 밀릴 여지가 구조적으로 없고, 파일당
디코드도 정확히 한 번뿐이다. 마지막 세그먼트가 끝나면 디코더를 죽인다.

## lerobot의 프레임 단위 API를 쓰지 않는 또 하나의 이유

`save_episode()` 는 에피소드 수에 대해 O(N^2)라 6.7만 에피소드에는 못 쓴다:
매 호출마다 (a) `concatenate_datasets` 로 누적 Arrow 테이블 전체를 다시 만들고
(b) `root.rglob("*.mp4")` 로 트리를 통째로 훑어 개수를 assert 한다.
그래서 parquet/meta 도 직접 쓴다 — 다만 스키마와 통계는 lerobot 자신의
`get_hf_features_from_features` / `get_feature_stats` 로 만들어 포맷 정합성을 지킨다.

## 메모리

프레임을 쌓아 두지 않고 150KB짜리 한 장씩 흘려보내므로 워커당 상주 메모리는
수십 MB다. (이전 구조에서는 `get_frames_in_range` 로 세그먼트 전체를 받아
`.float()` 하는 바람에 최장 세그먼트 3,326프레임에서 720p float32 20.7GB가
잡혀 exit 137(OOM kill)이 났다.)
"""
import argparse, json, os, shutil, subprocess, sys, time
import multiprocessing as mp
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 이 파일의 두 단계 위 (공개하며 절대경로를 바꿈)
DATA = f"{ROOT}/data"
REPO = "behavior-1k/2026-challenge-demos"
OUT_REPO = "cooking-vla/behavior-cooking-224"
RAW = f"{DATA}/raw"                      # 임시 원본 (처리 후 삭제)
OUT = f"{DATA}/lerobot/{OUT_REPO}"
STAGE = f"{OUT}/.staging"                # 워커가 쓰는 임시 mp4
STATE_F = f"{DATA}/build_state.json"     # 재개용

CAMS = {  # 출력 이름 -> 원본 feature 이름
    "image":             "observation.rgb.zed_link_camera_0",
    "left_wrist_image":  "observation.rgb.left_realsense_link_camera_0",
    "right_wrist_image": "observation.rgb.right_realsense_link_camera_0",
}
SIZE, FPS, WIN, CHUNK = 224, 30, 64, 1000

# 인코더 설정. lerobot 기본값(libsvtav1/g=2/crf=30)은 g=2가 2프레임마다 키프레임이라
# 압축이 사실상 없다. 실측 224p 654프레임: 기본값 2,783 B/frame @186 f/s vs
# 아래 설정 523 B/frame @858 f/s — 5.3배 작고 4.6배 빠르다.
ENC = ["-vcodec", "libx264", "-g", "15", "-crf", "23", "-pix_fmt", "yuv420p"]


# ---------------------------------------------------------------- 워커

FRAME_BYTES = SIZE * SIZE * 3


def _decoder(path):
    """소스 파일을 처음부터 순차 디코드하며 224p raw 프레임을 stdout으로 흘린다.

    `-vsync 0` 은 프레임 복제/누락 없이 입력 프레임과 1:1로 내보내기 위한 것이고,
    리사이즈를 ffmpeg의 swscale에 맡기는 것이 이 파이프라인의 핵심이다(아래 참조).
    """
    return subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-threads", "1", "-i", path, "-vsync", "0",
         "-vf", f"scale={SIZE}:{SIZE}:flags=bicubic",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=FRAME_BYTES * 4)


def _encoder(path):
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-threads", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE}x{SIZE}",
         "-r", str(FPS), "-i", "pipe:0", "-threads", "1", *ENC, path],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def encode_stream(job):
    """(카메라, 소스 파일) 하나를 순차 디코드하며 세그먼트별 mp4로 갈라 쓴다.

    반환: [(pid, cam, ok, 이미지 통계), ...]

    `from_timestamp` 는 에피소드마다 하나이고 에피소드는 전부 단일 파일 안에
    들어 있음을 확인했다(3000/3000). 게다가 한 파일에서 우리가 쓰는 프레임 범위가
    사실상 파일 전체다. 그래서 탐색 없이 파일을 처음부터 한 번만 읽으면 되고,
    파일 내 절대 프레임 번호가 곧 정렬 근거가 된다 — 프레임이 밀릴 여지가 없다.

    ⚠️ 세그먼트는 겹칠 수 있다. BEHAVIOR의 skill_annotation에는 구간이 겹치는
    스킬이 있어서(전체 66,892개 중 314개 스트림에서 발생) 활성 세그먼트를
    하나만 두면 겹치는 순간 `i` 가 다음 시작점을 지나쳐 버리고, 그 스트림의
    **나머지 세그먼트가 전부 유실된다.** 실제로 그렇게 태스크 4에서 283개를
    잃었고 전체로는 50.2% 유실이 예측됐다. 그래서 활성 세그먼트를 리스트로
    두고 같은 프레임을 여러 인코더에 동시에 흘려보낸다.
    """
    from lerobot.common.datasets.compute_stats import sample_indices, get_feature_stats

    cam, path, segs = job          # segs: [(pid, a, b)] 파일 내 절대 프레임, a로 정렬됨
    dec = _decoder(path)
    out, si, act, i = [], 0, [], 0

    def finish(c):
        ok = c["ok"]
        try:
            c["proc"].stdin.close()
            if c["proc"].wait() != 0:
                ok = False
        except Exception:
            ok = False
        st = {}
        if ok and len(c["keep"]) == len(set(c["idxs"])):
            arr = np.stack([c["keep"][k] for k in c["idxs"]]).transpose(0, 3, 1, 2)
            f = get_feature_stats(arr, axis=(0, 2, 3), keepdims=True)
            st = {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in f.items()}
        else:
            ok = False
        out.append((c["pid"], cam, ok, st))

    try:
        while si < len(segs) or act:
            buf = dec.stdout.read(FRAME_BYTES)
            if len(buf) < FRAME_BYTES:
                break                                  # 파일이 예상보다 짧다
            while si < len(segs) and segs[si][1] == i:  # 같은 프레임에서 여러 개 시작 가능
                pid, a, b = segs[si]
                idxs = sample_indices(b - a)
                act.append({"pid": pid, "a": a, "b": b, "idxs": idxs, "want": set(idxs),
                            "keep": {}, "proc": _encoder(f"{STAGE}/{pid:07d}_{cam}.mp4"),
                            "ok": True})
                si += 1
            for c in act:
                try:
                    c["proc"].stdin.write(buf)
                except Exception:
                    c["ok"] = False
                k = i - c["a"]
                if k in c["want"]:
                    c["keep"][k] = np.frombuffer(buf, np.uint8).reshape(SIZE, SIZE, 3).copy()
            for c in [c for c in act if i == c["b"] - 1]:
                finish(c); act.remove(c)
            i += 1
    finally:
        for c in act:                                  # 중간에 끊긴 것
            c["ok"] = False; finish(c)
        done = {o[0] for o in out}
        for pid, a, b in segs:                         # 도달하지 못한 것
            if pid not in done:
                out.append((pid, cam, False, {}))
        try:
            dec.kill(); dec.wait()                     # 남은 프레임은 디코드하지 않는다
        except Exception:
            pass
    return out


# ---------------------------------------------------------------- 메타/포맷

def features_dict():
    img = lambda: {"dtype": "video", "shape": (SIZE, SIZE, 3),
                   "names": ["height", "width", "channel"]}
    return {"image": img(), "left_wrist_image": img(), "right_wrist_image": img(),
            "state":   {"dtype": "float32", "shape": (61,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (23,), "names": ["actions"]}}


def create_skeleton():
    """lerobot에게 info.json 초기본을 만들게 한 뒤 객체는 버린다.

    DEFAULT_FEATURES(timestamp/frame_index/episode_index/index/task_index)와
    기본 필드들을 직접 하드코딩하지 않기 위해서다. 이후 단계는 전부 평범한
    json/jsonl 조작이라 lerobot 객체를 다시 만들지 않는다 (Hub 조회 회피).
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    shutil.rmtree(OUT, ignore_errors=True)
    LeRobotDataset.create(repo_id=OUT_REPO, root=OUT, fps=FPS, robot_type="r1_pro",
                          features=features_dict(), use_videos=True,
                          image_writer_threads=0, image_writer_processes=0)


def ref_schema(hf_features):
    """lerobot이 쓰는 parquet 스키마(HF 메타데이터 포함)를 한 번만 만들어 재사용한다.

    `datasets.Dataset.to_parquet` 은 에피소드당 26.7ms(6.7만개에 30분)인데
    같은 스키마로 pyarrow 직접 쓰기는 4.3ms다. 스키마는 metadata까지 동일함을 확인.
    """
    import datasets, pyarrow.parquet as pq
    datasets.disable_progress_bars()
    dummy = {"state": np.zeros((1, 61), np.float32), "actions": np.zeros((1, 23), np.float32),
             "timestamp": np.zeros(1, np.float32), "frame_index": np.zeros(1, np.int64),
             "episode_index": np.zeros(1, np.int64), "index": np.zeros(1, np.int64),
             "task_index": np.zeros(1, np.int64)}
    tmp = f"{DATA}/.schema.parquet"
    datasets.Dataset.from_dict({k: dummy[k] for k in hf_features},
                               features=hf_features, split="train").to_parquet(tmp)
    s = pq.read_schema(tmp)
    os.remove(tmp)
    return s


def write_parquet(path, cols, schema):
    import pyarrow as pa, pyarrow.parquet as pq
    arrs = {}
    for k in schema.names:
        v = cols[k]
        if v.ndim == 2:
            arrs[k] = pa.FixedSizeListArray.from_arrays(pa.array(v.ravel(), pa.float32()), v.shape[1])
        else:
            arrs[k] = pa.array(v)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.table(arrs).cast(schema), path)


# ---------------------------------------------------------------- 다운로드

def load_state():
    return json.load(open(STATE_F)) if os.path.exists(STATE_F) else {"done": []}


def save_state(s):
    json.dump(s, open(STATE_F, "w"), indent=1)


def files_for(sub):
    fs = set()
    for _, r in sub[["data/chunk_index", "data/file_index"]].drop_duplicates().iterrows():
        fs.add("data/chunk-%03d/file-%03d.parquet"
               % (int(r["data/chunk_index"]), int(r["data/file_index"])))
    for c in CAMS.values():
        for _, r in sub[["%s/chunk_index" % c, "%s/file_index" % c]].drop_duplicates().iterrows():
            fs.add("videos/%s/chunk-%03d/file-%03d.mp4"
                   % (c, int(r["%s/chunk_index" % c]), int(r["%s/file_index" % c])))
    return sorted(fs)


def download(files):
    args = ["hf", "download", REPO, "--repo-type", "dataset", "--local-dir", RAW]
    for f in files:
        args += ["--include", f]
    env = dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="1")
    subprocess.run(args, check=True, env=env, stdout=subprocess.DEVNULL)


# ---------------------------------------------------------------- 메인

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, nargs="*", default=None)
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="태스크당 소스 에피소드 상한 (기본: 전부, 200)")
    ap.add_argument("--nav-keep", type=float, default=0.25,
                    help="'move to' 세그먼트 유지 비율 (기본 25%%)")
    ap.add_argument("--keep-raw", action="store_true",
                    help="원본을 지우지 않는다 (스모크 반복용, 디스크 주의)")
    ap.add_argument("--workers", type=int, default=12,
                    help="인코딩 워커 수 (8코어/16스레드에서 12가 최적)")
    a = ap.parse_args()

    from lerobot.common.datasets.utils import (
        get_hf_features_from_features, serialize_dict)
    from lerobot.common.datasets.compute_stats import get_feature_stats

    seg = pd.read_parquet(f"{DATA}/segments.parquet")
    if a.tasks:
        seg = seg[seg.task_index.isin(a.tasks)]

    # move to 서브샘플링: 전체 프레임의 ~16% 감소, 잃는 정보 없음
    rng = np.random.default_rng(0)
    nav = seg.skill == "move to"
    drop = nav & (rng.random(len(seg)) >= a.nav_keep)
    print(f"'move to' 서브샘플링: {nav.sum():,} -> {(nav & ~drop).sum():,}개 유지")
    seg = seg[~drop].reset_index(drop=True)

    state = load_state()
    fresh = not state["done"] or not os.path.exists(f"{OUT}/meta/info.json")
    if fresh:
        state = {"done": []}
        create_skeleton()
        tasks = []                                  # 지시문 -> task_index (등장 순)
        n_ep = n_fr = 0
    else:
        tasks = [t["task"] for t in map(json.loads, open(f"{OUT}/meta/tasks.jsonl"))]
        eps = [json.loads(l) for l in open(f"{OUT}/meta/episodes.jsonl")]
        n_ep, n_fr = len(eps), sum(e["length"] for e in eps)
        print(f"재개: 에피소드 {n_ep:,} / 프레임 {n_fr:,}")
    tidx = {t: i for i, t in enumerate(tasks)}

    info = json.load(open(f"{OUT}/meta/info.json"))
    # get_hf_features_from_features는 shape을 튜플로 비교한다(`ft["shape"] == (1,)`).
    # info.json에서는 리스트로 실려 오므로 lerobot의 load_info와 동일하게 변환해야
    # 스칼라 컬럼이 Sequence(length=1)로 잘못 잡히지 않는다.
    hf_features = get_hf_features_from_features(
        {k: {**v, "shape": tuple(v["shape"])}
         for k, v in info["features"].items() if v["dtype"] != "video"})
    schema = ref_schema(hf_features)
    os.makedirs(STAGE, exist_ok=True)

    fh = {k: open(f"{OUT}/meta/{k}.jsonl", "a") for k in ("tasks", "episodes", "episodes_stats")}
    t_start = time.time()

    for task, tg in seg.groupby("task_index"):
        if int(task) in state["done"]:
            print(f"[task {task}] 완료됨, 건너뜀"); continue
        files = files_for(tg)
        print(f"\n[task {task}] 세그먼트 {len(tg):,} / 파일 {len(files)} 다운로드 중...")
        td = time.time(); download(files); dl = time.time() - td
        tp = time.time()

        # 소스 에피소드별 state/action을 numpy로 뽑아 둔다 (워커에는 넘기지 않는다)
        arrs = {}
        for f in files:
            if not f.endswith(".parquet"):
                continue
            df = pd.read_parquet(f"{RAW}/{f}")
            for ei, g in df.groupby("episode_index"):
                g = g.sort_values("frame_index")
                fi = g["frame_index"].to_numpy()
                if fi[0] != 0 or not np.array_equal(fi, np.arange(len(fi))):
                    print(f"  !! ep{ei}: frame_index가 0..N-1이 아님, 건너뜀"); continue
                arrs[int(ei)] = (np.stack(g["observation.state"].to_numpy()).astype(np.float32),
                                 np.stack(g["action"].to_numpy()).astype(np.float32))
            del df

        eps_src = sorted(tg.episode_index.unique())
        if a.max_episodes:
            eps_src = eps_src[:a.max_episodes]

        # pid는 (에피소드, frame_start) 순으로 결정론적으로 매긴다.
        # 최종 episode_index는 이 순서대로 메인이 다시 부여한다.
        plan, pid = {}, 0
        streams = {}                        # (out_cam, 파일경로) -> [(pid, a, b)]
        for ei in eps_src:
            if int(ei) not in arrs:
                continue
            eg = tg[tg.episode_index == ei].sort_values("frame_start")
            nst = len(arrs[int(ei)][0])
            # itertuples는 '/'·'.'가 든 컬럼명을 _1,_2로 바꿔 버리므로 dict로 순회한다
            for d in eg.to_dict("records"):
                f0, f1 = int(d["frame_start"]), int(d["frame_end"])
                if f1 > nst:
                    continue
                for out, cam in CAMS.items():
                    path = "%s/videos/%s/chunk-%03d/file-%03d.mp4" % (
                        RAW, cam, int(d["%s/chunk_index" % cam]), int(d["%s/file_index" % cam]))
                    off = int(round(float(d["%s/from_timestamp" % cam]) * FPS))
                    streams.setdefault((out, path), []).append((pid, off + f0, off + f1))
                plan[pid] = (int(ei), f0, f1, d["instruction"])
                pid += 1

        jobs = [(cam, path, sorted(v, key=lambda x: x[1]))
                for (cam, path), v in streams.items()]
        jobs.sort(key=lambda j: -len(j[2]))            # 큰 작업부터 -> 꼬리 지연 감소

        print(f"[task {task}] 다운로드 {dl/60:.1f}분 / 스트림 {len(jobs)} / 세그먼트 {pid:,} 인코딩 중...")
        okc = {}                                       # pid -> 성공한 카메라 수
        stats_by = {}                                  # pid -> {카메라: 통계}
        nstream, camfr = 0, 0
        with mp.Pool(a.workers) as pool:
            for res in pool.imap_unordered(encode_stream, jobs):
                for p, cam, ok, st in res:
                    if ok:
                        okc[p] = okc.get(p, 0) + 1
                        stats_by.setdefault(p, {})[cam] = st
                        camfr += plan[p][2] - plan[p][1]
                nstream += 1
                el = time.time() - tp
                # 카메라-f/s 로 보여 준다. 세그먼트는 카메라 3대가 모두 끝나야 완성되므로
                # 세그먼트 기준으로 세면 첫 스트림들에서 0으로 보여 진행이 멈춘 듯하다.
                done = sum(1 for c in okc.values() if c == len(CAMS))
                print(f"    스트림 {nstream}/{len(jobs)}  세그먼트 {done:,}/{pid:,}"
                      f"  {camfr/max(el,1):.0f} 카메라-f/s  경과 {el/60:.1f}분", flush=True)
        got = {p: (okc.get(p, 0) == len(CAMS), stats_by.get(p, {})) for p in range(pid)}

        # 최종 인덱스 부여: 성공한 세그먼트만 순서대로 -> episode_index에 구멍 없음
        nfail, failed = 0, []
        for p in range(pid):
            ok, istats = got.get(p, (False, None))
            if not ok:
                nfail += 1
                ei, f0, f1, instr = plan[p]
                failed.append({"task": int(task), "episode": ei, "frame_start": f0,
                               "frame_end": f1, "instruction": instr,
                               "cams_ok": okc.get(p, 0)})
                continue
            ei, f0, f1, instr = plan[p]
            n = f1 - f0
            if instr not in tidx:
                tidx[instr] = len(tasks); tasks.append(instr)
                fh["tasks"].write(json.dumps({"task_index": tidx[instr], "task": instr},
                                             ensure_ascii=False) + "\n")
            ti = tidx[instr]
            e, ch = n_ep, n_ep // CHUNK
            st_a, ac_a = arrs[ei]
            cols = {"state": st_a[f0:f1], "actions": ac_a[f0:f1],
                    "timestamp": (np.arange(n) / FPS).astype(np.float32),
                    "frame_index": np.arange(n), "episode_index": np.full(n, e),
                    "index": np.arange(n_fr, n_fr + n), "task_index": np.full(n, ti)}
            write_parquet(f"{OUT}/data/chunk-{ch:03d}/episode_{e:06d}.parquet", cols, schema)
            for c in CAMS:
                d = f"{OUT}/videos/chunk-{ch:03d}/{c}"
                os.makedirs(d, exist_ok=True)
                os.replace(f"{STAGE}/{p:07d}_{c}.mp4", f"{d}/episode_{e:06d}.mp4")

            stats = dict(istats)
            for k, v in cols.items():
                stats[k] = get_feature_stats(v, axis=0, keepdims=(v.ndim == 1))
            fh["episodes"].write(json.dumps(
                {"episode_index": e, "tasks": [instr], "length": n}, ensure_ascii=False) + "\n")
            fh["episodes_stats"].write(json.dumps(
                {"episode_index": e, "stats": serialize_dict(stats)}) + "\n")
            n_ep += 1; n_fr += n
        for f in fh.values():
            f.flush()
        if failed:   # 실패는 조용히 넘기지 않는다 — 원인을 볼 수 있게 남긴다
            with open(f"{DATA}/failed_segments.jsonl", "a") as ff:
                for r in failed:
                    ff.write(json.dumps(r, ensure_ascii=False) + "\n")

        # info.json 갱신 (lerobot은 에피소드마다 쓰지만 태스크마다면 충분하다)
        info["total_episodes"], info["total_frames"] = n_ep, n_fr
        info["total_tasks"], info["total_videos"] = len(tasks), n_ep * len(CAMS)
        info["total_chunks"] = max(1, (n_ep + CHUNK - 1) // CHUNK)
        info["splits"] = {"train": f"0:{n_ep}"}
        if n_ep and not info["features"]["image"].get("info"):
            from lerobot.common.datasets.video_utils import get_video_info
            for c in CAMS:
                info["features"][c]["info"] = get_video_info(
                    f"{OUT}/videos/chunk-000/{c}/episode_000000.mp4")
        json.dump(info, open(f"{OUT}/meta/info.json", "w"), indent=4, ensure_ascii=False)

        if not a.keep_raw:
            shutil.rmtree(RAW, ignore_errors=True)
        for f in os.listdir(STAGE):
            os.remove(f"{STAGE}/{f}")
        state["done"].append(int(task)); save_state(state)
        proc = time.time() - tp
        nfr_t = sum(plan[p][2] - plan[p][1] for p in got if got[p][0])
        print(f"[task {task}] 에피소드 {pid - nfail:,}개(실패 {nfail}) / 프레임 {nfr_t:,} / "
              f"처리 {proc/60:.1f}분 / {nfr_t/max(proc,1):.0f} f/s"
              + ("" if a.keep_raw else " / 원본 삭제됨"))

    for f in fh.values():
        f.close()
    shutil.rmtree(STAGE, ignore_errors=True)
    print(f"\n완료: {OUT}\n  에피소드 {n_ep:,} / 프레임 {n_fr:,} / 지시문 {len(tasks)}종"
          f" / 총 {(time.time()-t_start)/3600:.1f}시간")


if __name__ == "__main__":
    main()
