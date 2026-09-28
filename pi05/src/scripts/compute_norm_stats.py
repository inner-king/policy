"""정규화 통계 계산 래퍼 — cooking-vla 설정을 등록한 뒤 openpi 스크립트로 넘긴다.

    uv run --project third_party/openpi python scripts/compute_norm_stats.py pi05_cooking

통계는 repack + data_transforms 를 **적용한 뒤** 계산되므로 델타 변환이 반영된
분포로 q01/q99가 잡힌다. 별도로 할 일은 없다.
"""
import pathlib, runpy, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))

import lerobot_patch
lerobot_patch.apply()          # 6.7만 에피소드에서 데이터셋 생성 321초/25GB -> 188초/10.8GB

import task_subset
task_subset.apply()            # asset_id 에 필터가 등록된 설정이면 인덱스를 걸러 준다

import cooking_config
cooking_config.register()

runpy.run_path(str(ROOT / "third_party/openpi/scripts/compute_norm_stats.py"), run_name="__main__")
