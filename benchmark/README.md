# OmniGibson 자원 사용량 벤치마크

OmniGibson을 시나리오별로 헤드리스 실행하면서 RAM, VRAM, CPU, GPU 사용량과 속도를 0.5초 간격으로 기록하는 도구와 측정 결과입니다.

## 파일

| 파일 | 설명 |
|---|---|
| `monitor.py` | 명령을 실행하고 프로세스 트리, 시스템, GPU 사용량을 기록합니다. 출력: `<name>.json`(요약), `<name>.csv`(시계열), `<name>.log` |
| `scenario.py` | 씬, 로봇, 카메라, 태스크 조합을 헤드리스로 띄워 N step을 실행하고, `@@MARK`/`@@STAT` 줄로 단계별 시각과 속도를 출력합니다 |
| `vec_scenario.py` | `og.VectorEnvironment`로 여러 환경을 동시에 돌리는 확장성 측정 |
| `results/summary_rtx3060.csv` | 검증을 거친 시나리오별 요약 |
| `results/raw/` | 실행별 원본 요약(JSON)과 시계열(CSV) |

## 실행 방법

```bash
source scripts/og_env.sh
cd benchmark
python monitor.py --out results --name S2_rs_int_r1pro -- python scenario.py --scene Rs_int --robot R1Pro --cam 128 --steps 300
python monitor.py --out results --name S5_eval_default -- python -m omnigibson.eval.eval --task-name turning_on_radio --policy local --max-steps 300 --no-write-video --output-dir ./eval_out
python monitor.py --out results --name S8_vec_env_4 -- python vec_scenario.py --num-envs 4 --steps 2000
```

- **시뮬레이터는 한 번에 하나만** 실행하세요. 동시에 돌리면 GPU 측정값이 섞입니다.
- 수치를 읽는 법:
  - RAM은 `sys_used_delta_mb`(실행 전 대비 시스템 사용량 증가분)를 봅니다. `pss_mb`로 교차 확인할 수 있습니다.
  - `rss_mb`는 약 27개 자식 프로세스의 공유 페이지를 중복 합산하므로 사용량으로 쓰면 안 됩니다.
- 구간 정의:
  - 정상 상태: `warmup_done → steps_done` 구간
  - 로딩 시간: 실행 시작부터 `warmup_done` 마크까지

## 결과

RTX 3060 12 GB(i9-12900K, RAM 32 GB) 측정 결과와 해석은 위키 [Resource Usage](https://github.com/inner-king/policy/wiki/Resource-Usage)에 있습니다.

- 원본 데이터: `results/summary_rtx3060.csv`(시나리오별 요약), `results/raw/`(실행별 JSON 요약과 0.5초 간격 CSV 시계열)
- `S7_rs_int_gpu_dynamics_fail1`, `_fail2`는 GPU dynamics + R1Pro 조합이 첫 step에서 크래시한 실행입니다(CUDA error 700).
- 요약: 큰 씬이나 공식 평가에서 RAM 약 6 GB, VRAM 약 8 GB를 쓰고, GPU는 거의 항상 100%입니다.
