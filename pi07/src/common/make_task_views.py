"""HDD에 받아둔 B1K 데모 미러 위에 태스크별 LeRobot 루트 뷰를 심링크로 깐다.

원본은 100개 태스크가 한 저장소에 섞여 있고 청크 인덱스로만 갈린다.
학습 스크립트는 태스크 하나짜리 루트를 원하므로 태스크 이름 디렉토리를 만들고
해당 청크만 링크한다. 원본은 건드리지 않는다.

meta/ 는 100개 태스크 전체 분량을 그대로 링크한다 — LeRobotDataset 은
`episodes=[...]` 로 어차피 걸러 읽고, info.json 을 태스크별로 다시 쓰면
stats/episodes 와 어긋나 CastError 가 난다 (make_rgb_only_root.py 주석 참고).

실행:  vla/.venv/bin/python src/common/make_task_views.py
출력:  data/b1k/<task_name>/  +  data/b1k/manifest.json

뷰 하나는 data/ · videos/ · meta/ · annotations/ 네 개로 이뤄진다.
"""
import json
import os
from pathlib import Path

CAPSTONE = Path(__file__).resolve().parents[2]
MIRROR = Path(os.environ.get("B1K_HDD", "/data/b1k")) / "2026-challenge-demos"
DST_ROOT = CAPSTONE / "data" / "b1k"
EPISODES_PER_TASK = 200

META_LINKS = ("info.json", "stats.json", "tasks.parquet", "tasks.jsonl", "episodes")


def link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)


names = {
    r["task_index"]: r["task_name"]
    for r in map(json.loads, (MIRROR / "meta" / "tasks.jsonl").read_text().splitlines())
}

chunks = sorted(int(p.name.split("-")[1]) for p in (MIRROR / "data").glob("chunk-*"))
print(f"미러에서 찾은 청크 {len(chunks)}개: {chunks}")

manifest = []
for c in chunks:
    name = names[c]
    dst = DST_ROOT / name
    if dst.exists() and not dst.is_symlink() and not (dst / "data" / f"chunk-{c:03d}").exists():
        # 이름은 같은데 다른 원본을 보고 있는 기존 뷰 — 덮지 않는다
        print(f"  ⚠️  {name}: 이미 있고 chunk-{c:03d} 를 안 가리킨다. 건너뜀")
        continue

    link(MIRROR / "data" / f"chunk-{c:03d}", dst / "data" / f"chunk-{c:03d}")
    cams = []
    for cam_dir in sorted((MIRROR / "videos").iterdir()):
        # hf download 가 실패한 글롭 패턴을 디렉토리 이름 그대로 만들어 둘 때가 있다
        if not cam_dir.name.startswith("observation."):
            continue
        chunk_dir = cam_dir / f"chunk-{c:03d}"
        if chunk_dir.exists():
            link(chunk_dir, dst / "videos" / cam_dir.name / f"chunk-{c:03d}")
            cams.append(cam_dir.name)
    for f in META_LINKS:
        if (MIRROR / "meta" / f).exists():
            link(MIRROR / "meta" / f, dst / "meta" / f)

    # 스킬 구간 라벨. 경로가 chunk-NNN 이 아니라 task-NNNN 이다 (에피소드당 json 1개).
    ann = MIRROR / "annotations" / f"task-{c:04d}"
    n_ann = len(list(ann.glob("*.json"))) if ann.exists() else 0
    if n_ann:
        link(ann, dst / "annotations")

    lo = c * EPISODES_PER_TASK
    manifest.append({
        "task_name": name,
        "task_index": c,
        "chunk": f"chunk-{c:03d}",
        "episodes": [lo, lo + EPISODES_PER_TASK - 1],
        "root": str(dst),
        "cameras": cams,
        "annotations": n_ann,
    })
    print(f"  ✓ {name:26} chunk-{c:03d}  ep {lo}–{lo + EPISODES_PER_TASK - 1}  "
          f"cam {len(cams)}종  annot {n_ann}개")

(DST_ROOT / "manifest.json").write_text(json.dumps({
    "repo_id": "behavior-1k/2026-challenge-demos",
    "mirror": str(MIRROR),
    "episodes_per_task": EPISODES_PER_TASK,
    "tasks": manifest,
}, indent=2, ensure_ascii=False))
print(f"\n✅ 뷰 {len(manifest)}개  ·  {DST_ROOT / 'manifest.json'}")
