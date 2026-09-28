"""15개 요리 태스크의 에피소드를 스킬 세그먼트로 분할하고 지시문을 생성한다.

출력: data/segments.parquet
  - 세그먼트별 프레임 범위(에피소드 상대 + 데이터셋 절대)
  - 자연어 지시문
  - parquet/mp4 위치 (선별 다운로드 및 재패킹에 사용)
"""
import json, glob, os, sys, collections
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment_lib import make_instruction, spans, unknown_hashes

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")  # 공개하며 절대경로를 바꿈
FINAL = [4, 40, 41, 42, 43, 45, 46, 48, 49, 51, 62, 70, 72, 95, 96]
RGB = ["observation.rgb.left_realsense_link_camera_0",
       "observation.rgb.right_realsense_link_camera_0",
       "observation.rgb.zed_link_camera_0"]

# 1) 에피소드 인덱스 로드 (parquet/mp4 위치의 원천)
ep = pd.concat([pd.read_parquet(f) for f in
                glob.glob(f"{ROOT}/meta/episodes/**/*.parquet", recursive=True)],
               ignore_index=True)
ep = ep[ep.task_index.isin(FINAL)]
print(f"대상 에피소드: {len(ep)}개 ({ep.task_index.nunique()}개 태스크)")

tasks = {json.loads(l)["task_index"]: json.loads(l)
         for l in open(f"{ROOT}/meta/tasks.jsonl")}

rows, no_instr = [], collections.Counter()
for r in ep.itertuples(index=False):
    ann_path = os.path.join(ROOT, r.annotation_path)
    if not os.path.exists(ann_path):
        continue
    a = json.load(open(ann_path))
    valid = (a.get("meta_data") or {}).get("valid_duration") or [0, r.length]
    v0, v1 = int(valid[0]), int(valid[1])

    for s in a.get("skill_annotation", []):
        instr = make_instruction(s)
        desc = (s.get("skill_description") or ["?"])[0]
        if not instr:
            no_instr[desc] += 1
            continue
        for (f0, f1) in spans(s):
            f0, f1 = max(f0, v0), min(f1, min(v1, r.length))
            if f1 - f0 < 15:            # 0.5초 미만 세그먼트는 버림
                continue
            row = dict(
                task_index=r.task_index,
                task_name=tasks[r.task_index]["task_name"],
                episode_index=r.episode_index,
                skill_idx=s.get("skill_idx"),
                skill=desc,
                skill_type=(s.get("skill_type") or [None])[0],
                instruction=instr,
                frame_start=f0, frame_end=f1,
                n_frames=f1 - f0,
                duration_s=(f1 - f0) / 30.0,
                # 데이터셋 절대 행 인덱스 (data/*.parquet 슬라이싱용)
                abs_start=int(r.dataset_from_index) + f0,
                abs_end=int(r.dataset_from_index) + f1,
                data_chunk=int(getattr(r, "_3")) if not hasattr(r, "data/chunk_index") else None,
            )
            rows.append(row)

df = pd.DataFrame(rows)

# itertuples가 'data/chunk_index' 같은 이름을 위치인자로 바꾸므로 원본에서 직접 조인
loc = ep[["episode_index", "data/chunk_index", "data/file_index",
          "dataset_from_index", "dataset_to_index", "length"]].copy()
for c in RGB:
    loc[f"{c}/chunk_index"] = ep[f"videos/{c}/chunk_index"].values
    loc[f"{c}/file_index"] = ep[f"videos/{c}/file_index"].values
    loc[f"{c}/from_timestamp"] = ep[f"videos/{c}/from_timestamp"].values
df = df.drop(columns=["data_chunk"]).merge(loc, on="episode_index", how="left")

os.makedirs(f"{ROOT}", exist_ok=True)
df.to_parquet(f"{ROOT}/segments.parquet", index=False)

print(f"\n세그먼트 {len(df):,}개 / 총 {df.duration_s.sum()/3600:.1f}시간 "
      f"/ 평균 {df.duration_s.mean():.1f}초")
if unknown_hashes:
    print("!! 미등록 해시:", unknown_hashes)
if no_instr:
    print("!! 지시문 생성 실패:", dict(no_instr.most_common(10)))
print(f"\n저장: {ROOT}/segments.parquet")
