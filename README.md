# policy

[BEHAVIOR-1K](https://behavior.stanford.edu/) / OmniGibson 시뮬레이터 환경 구축과 실행을 위한 스크립트 모음입니다.

자세한 설명은 **[Wiki](https://github.com/inner-king/policy/wiki)** 에 있습니다.

- [OmniGibson 설치](https://github.com/inner-king/policy/wiki/OmniGibson-Setup)
- [OmniGibson 실행](https://github.com/inner-king/policy/wiki/Running-OmniGibson)
- [BEHAVIOR 데이터셋 다운로드](https://github.com/inner-king/policy/wiki/Dataset-Download)
- [자원 사용량](https://github.com/inner-king/policy/wiki/Resource-Usage)
- [문제 해결](https://github.com/inner-king/policy/wiki/Troubleshooting)

실험별 코드와 설정은 하위 디렉토리에 있고, 결과는 [이슈](https://github.com/inner-king/policy/issues)로 올립니다.

- [π0.7 재현](pi07/README.md) — HL·LL·월드 모델 학습과 폐루프 평가 ([실험 세팅](pi07/SETTINGS.md))

## 빠른 시작

```bash
# 0. 사전 준비: NVIDIA 드라이버 580+ 설치 후 재부팅, git curl tmux (wiki: OmniGibson Setup 2.1)
tmux new -A -s install                      # SSH로 설치한다면 필수 (끊겨도 계속 진행)
git clone https://github.com/inner-king/policy.git ~/policy && cd ~/policy

# 1. 설치: Miniforge + conda env + BEHAVIOR-1K v3.9.2 + Isaac Sim 5.1 + 데이터셋
#    디스크 여유 100 GB 권장, 30분~수 시간. BEHAVIOR-1K는 ~/BEHAVIOR-1K에 설치됨
bash setup/install_behavior.sh --dry-run                                  # 실행할 명령만 확인
bash setup/install_behavior.sh --accept-nvidia-eula --accept-dataset-tos  # 라이선스를 읽고 동의한 경우

# 2. 환경 활성화와 점검 (새 셸마다 source 필요)
source scripts/og_env.sh
bash setup/check_env.sh --sim

# 3. 실행
python -u scripts/quickstart.py                 # Rs_int + R1Pro, 카메라 이미지를 ~/og_outputs에 저장
python -u scripts/behavior_task_demo.py         # BEHAVIOR 태스크 (picking_up_trash)
bash scripts/run_eval_smoke.sh                  # 2026 챌린지 평가 파이프라인 (zero-action 정책)
```

## 구성

```
setup/
  install_behavior.sh    BEHAVIOR-1K 전체 설치 (공식 setup.sh 래퍼, CuRobo용 CUDA 12.8을 conda env 안에 설치)
  download_datasets.sh   데이터셋만 따로 다운로드
  check_env.sh           설치 점검 (--sim: 헤드리스 시뮬레이션 실행)
scripts/
  og_env.sh              conda 환경 활성화 + OMNIGIBSON_HEADLESS=1 (source로 실행)
  quickstart.py          씬 + 로봇 최소 예제, 카메라 이미지 저장
  behavior_task_demo.py  BEHAVIOR 태스크 예제 (--list로 가능한 활동 확인)
  run_eval_smoke.sh      공식 평가(omnigibson.eval.eval) 스모크 테스트
benchmark/
  monitor.py, scenario.py, vec_scenario.py   자원 사용량 측정 도구
  results/                                   RTX 3060 12 GB 측정 결과
pi07/                    π0.7 재현 실험 (코드 · run.yaml · 환경 목록, pi07/README.md)
```

설치 후 디렉터리:

| 경로 | 내용 |
|---|---|
| `~/policy` | 이 레포. `setup/…`, `scripts/…` 명령은 모두 여기서 실행 |
| `~/BEHAVIOR-1K` | 설치 스크립트가 clone. OmniGibson 소스, `datasets/`, 캐시 `OmniGibson/appdata/` 포함 |
| `~/miniforge3` | conda. 환경 이름은 `behavior` |

## 검증 환경

| 항목 | 값 |
|---|---|
| OS | Ubuntu 24.04, NVIDIA driver 580.178 |
| 하드웨어 | i9-12900K, RAM 32 GB, RTX 3060 12 GB |
| 소프트웨어 | BEHAVIOR-1K v3.9.2, Isaac Sim 5.1.0, Python 3.11, torch 2.7.0+cu128, CUDA 12.8 (conda-forge) |

## 라이선스 주의

- BEHAVIOR 데이터셋(BEHAVIOR Data Bundle)은 **비상업적 학술 연구 용도**로만 쓸 수 있습니다.
- 데이터와 복호화 키(`omnigibson.key`)는 재배포할 수 없습니다. 이 레포에는 포함하지 않으며, 커밋하지 마세요.
- Isaac Sim은 NVIDIA EULA를 따릅니다.
