"""체크포인트가 **실제로 열리는지** 검사한다 — 학습 로그의 "저장 완료" 는 이것을 보장하지 않는다.

    python scripts/verify_checkpoint.py <run 폴더 또는 스텝 폴더> [...] [--read-params]

## 왜 필요한가 (HANDOFF 39.12)

2026-09-15 run8/20999 는 orbax 가 `Finished saving checkpoint` / `No errors found` 를
찍었는데 **Drive 에는 목차 파일(`params/d/<해시>`, 2,570바이트)이 빠진 채** 올라갔다.
가중치 본문 10 GiB 는 다 있었지만 목차가 없어 열리지 않았고, `rclone check` 도
**Drive 에 있는 파일끼리만** 대조하므로 통과했다. 런타임이 이미 반납된 뒤라 복구하지
못하고 18000 으로 평가했다. 학습 프로그램은 로컬 폴더(= Drive FUSE)에 쓴 것까지만
확인하고, 실제 업로드는 FUSE 가 뒤에서 따로 한다 — 그 사이에서 사라진 것으로 본다.

## 무엇을 보나

1. params·train_state 각각의 OCDBT 목차를 열어 키를 나열한다 (목차가 빠지면 여기서 실패).
2. 모양 정보(`.zarray`) 수가 `array_metadatas` 의 배열 수와 같은지.
3. `--read-params` 면 params 의 모든 배열을 끝까지 읽는다 (본문 파일이 잘렸으면 실패).
   params 는 v4 기준 약 10 GiB 라 Drive 에서 몇 분 걸린다. train_state(46 GiB)는 안 읽는다.

종료코드: 전부 통과 0 / 하나라도 실패 1.
"""
import json
import pathlib
import sys

import tensorstore as ts


def list_keys(part: pathlib.Path):
    kv = ts.KvStore.open({"driver": "ocdbt", "base": {"driver": "file", "path": str(part) + "/"}}).result()
    return kv.list().result()


def read_all(part: pathlib.Path, names):
    total = 0
    for name in names:
        arr = ts.open({"driver": "zarr",
                       "kvstore": {"driver": "ocdbt", "base": {"driver": "file", "path": str(part) + "/"},
                                   "path": name + "/"}}).result()
        total += arr.read().result().nbytes
    return total


def check_step(step: pathlib.Path, read_params: bool) -> bool:
    ok = True
    if not (step / "_CHECKPOINT_METADATA").exists():
        print(f"  🔴 {step.name}: _CHECKPOINT_METADATA 없음 — 저장이 끝나지 않았다")
        return False
    for part_name in ("params", "train_state"):
        part = step / part_name
        if not part.exists():
            if part_name == "params":
                print(f"  🔴 {step.name}/params 없음"); ok = False
            continue
        try:
            names = [m["array_metadata"]["param_name"] for m in
                     json.loads((part / "array_metadatas/process_0").read_text())["array_metadatas"]]
            keys = list_keys(part)
        except Exception as e:  # noqa: BLE001 — 어떤 에러든 "못 연다" 로 보고한다
            print(f"  🔴 {step.name}/{part_name}: 열기 실패 — {str(e)[:160]}")
            ok = False
            continue
        n_meta = sum(k.endswith(b".zarray") for k in keys)
        if n_meta != len(names):
            print(f"  🔴 {step.name}/{part_name}: 모양 정보 {n_meta}개 / 배열 {len(names)}개 — 목차가 불완전")
            ok = False
            continue
        msg = f"  ✅ {step.name}/{part_name}: 배열 {len(names)}개, 키 {len(keys)}개"
        if read_params and part_name == "params":
            try:
                msg += f", 전부 읽음 {read_all(part, names) / 2**30:.1f} GiB"
            except Exception as e:  # noqa: BLE001
                print(f"  🔴 {step.name}/params: 읽기 실패 — {str(e)[:160]}")
                ok = False
                continue
        print(msg)
    return ok


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    read_params = "--read-params" in sys.argv
    if not args:
        sys.exit(__doc__)
    all_ok = True
    for a in args:
        p = pathlib.Path(a)
        steps = [p] if (p / "params").exists() or (p / "_CHECKPOINT_METADATA").exists() else \
            sorted((d for d in p.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name))
        if not steps:
            print(f"🔴 {p}: 스텝 폴더가 없다"); all_ok = False
        for s in steps:
            all_ok &= check_step(s, read_params and s == steps[-1])   # 전부 읽기는 마지막 스텝만
    print("\n결과:", "✅ 전부 열린다" if all_ok else "🔴 열리지 않는 체크포인트가 있다 — 런타임을 끄기 전에 처리할 것")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
