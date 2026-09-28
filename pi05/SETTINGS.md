# π0.5 요리 스킬 정책 — 실험 세팅

π0.5 base 를 BEHAVIOR-1K 요리 데모의 스킬 구간으로 파인튜닝한 실험의 설정이다. 결과와 해석은 [이슈](https://github.com/inner-king/policy/issues)(`[pi05]`)로 따로 올린다.
코드·환경·실행 방법은 [README](README.md) 에 있고, 이 문서의 `configs/`·`scripts/` 경로는 [`ers345/pi05`](https://github.com/inner-king/policy/tree/ers345/pi05/pi05) 브랜치의 `pi05/src/` 기준이다.
수치는 코드 기본값, `runs/*/run.yaml`, 학습·평가 로그에서 가져왔다. 기록이 없는 값은 "기록 없음"으로 적는다.

## 1. 구성

| 항목 | 내용 |
|---|---|
| 정책 | π0.5 하나. 3뷰 RGB + 프롬프트(스킬 문장 + 이산 state) → 액션 청크. 상위 정책·서브골 생성 없음 |
| 출력 | 32 스텝 × 23차원 액션 청크 (1차만 50 스텝) |
| 런타임 | 청크 32 스텝 중 앞 16 스텝 실행 후 다시 묻는다(receding horizon). 디노이징 10 스텝(openpi 기본). 30 Hz |
| 평가 때 프롬프트 | 데모의 해당 스킬 주석 문장(사람이 쓴 정답 문장). 일반화 검사에서만 바꿔 쓴 문장 |
| 계보 | π0.5 base → 2차 → 3 → 4 → 5 → 6 → 7 → 9 → 10. 1차는 따로(π0.5 base 에서), 8차는 7차에서 갈라졌다가 닫았다 |

## 2. 모델

| 항목 | 값 |
|---|---|
| 가중치 | openpi `pi05_base` (`gs://openpi-assets/checkpoints/pi05_base/params`), openpi 커밋 `215abfb`, JAX |
| VLM | PaliGemma: SigLIP 414.8M + Gemma 2B 2,508.5M (width 2048, 18층) |
| action expert | Gemma 300M 변형 427.9M (width 1024, 18층, mlp 4096) |
| LoRA | `gemma_2b_lora` rank 16 / α 16 (27.9M), `gemma_300m_lora` rank 32 / α 32 (22.1M). attention·FFN. 설정 이름을 바꾸면 파라미터 트리가 달라져 이어받기가 깨지므로 v3 이후에도 그대로 둔다 |
| 전체 | 3,403.4M (체크포인트에서 직접 셈) |
| state 입력 | π0.5 기본대로 정규화 후 256구간 정수로 바꿔 텍스트 토큰으로 넣는다(`discrete_state_input=True`) |
| 텍스트 길이 상한 | `max_token_len` 280 (v1) / 150 (v2 부터). 기본값 200 이면 v1 프롬프트가 100% 잘린다(state 61개가 약 200토큰) |
| 흐름 매칭 | openpi 기본: x_t = t·ε + (1−t)·a, 목표 u = ε − a, t = 0.001 + 0.999·Beta(1.5, 1) |
| 다표본 흐름 매칭 (v3 부터) | 프리픽스(연산의 약 99.5%)를 한 번 통과시키고 잡음·시간 표본 **8개**로 손실을 낸다(`configs/pi0_patch.py`, 값 동일성은 `scripts/check_pi0_patch.py`) |
| EMA | 없음 (v1·v2) / 0.99 (v3 부터). 체크포인트의 `params/` 는 EMA 가중치다 |

설정 판별 학습 범위 (`configs/cooking_config.py` 의 `freeze_filter`, 수치는 `scripts/check_freeze_filter.py`):

| 판 | 얼린 것 | 학습하는 것 | 학습 파라미터 |
|---|---|---|---:|
| v1·v2 | Gemma 2B 본체, action expert 본체 | SigLIP, LoRA 둘, 선형층(2.2M) | 467.0M |
| v3·v5 | SigLIP, Gemma 2B 본체 | action expert 본체, LoRA 둘, 선형층 | 480.1M |
| v4 | SigLIP | Gemma 2B 본체까지 전부 | 2,988.6M |

v3·v5 에서 문장을 읽는 쪽의 학습은 Gemma 2B 본체가 얼어 있으므로 **언어부 LoRA 27.9M 안에만** 쌓인다.

## 3. 프롬프트와 지시문

```
Task: {스킬 문장}, State: {0~255 정수 25개};
Action: 
```

| 항목 | 값 |
|---|---|
| 문장 생성 | B1K `skill_annotation` 한 구간 = 세그먼트 하나. `skill_description` + `object_id` 를 영어 명령문으로 바꾼다(`scripts/segment_lib.py`). 물체 id 끝 번호와 모델 해시(명시 목록 17개)를 지운다 |
| 수 | 에피소드 3,000개 → 세그먼트 81,278개(평균 13.1초) → 고유 문장 472종. 이동 서브샘플 뒤 데이터셋에는 466종 |
| 학습 범위 | v1 466종 전부 / v2·v3·v4 `TOP20` / v5 `TOP32` = TOP20 + `NEW12` (문장 목록은 `configs/cooking_config.py`) |
| TOP20 | 에피소드 수 상위 20종. 화면은 거의 같고 문장만 다른 쌍(`pick up the brussels sprouts` / `pick up the half brussels sprouts`)을 포함한다 |
| NEW12 기준 | ① 자동 판정기가 붙는 형태(열기·닫기 / 켜기 / 집기 / 놓기 / 자르기) ② 일반화 검사용 미학습 문장 9종을 건드리지 않음 ③ 프레임 10만 이상 ④ 기존 20종과 문장으로만 갈리는 짝(같은 물체 다른 출처, 열기·켜기에 이어지는 넣기 등) |
| 드롭아웃 | 없음. 모든 학습 표본에 스킬 문장이 들어간다 |

생성 규칙에서 걸렸던 것: `pour` 의 첫 인자는 같은 물체 여러 개의 리스트라 평탄화하면 슬롯이 밀린다, `chop` 의 인자 순서는 `[도구, 대상]` 이다, 6글자 단어(`pepper`)를 모델 해시로 잘못 자르지 않게 해시를 목록으로 관리한다.

## 4. 관측 · 상태 · 액션 · 정규화

| 항목 | 값 |
|---|---|
| 이미지 | RGB 3뷰: head `zed_link` 720×720, 양손목 `left/right_realsense_link` 480×480 → 224×224 (ffmpeg bicubic, H.264 crf 23). depth 는 쓰지 않는다 |
| state | 원본 61차원 중 25개 (v2 부터): base_qvel 3 · torso 4 · 왼팔 7 · 왼그리퍼 2 · 오른팔 7 · 오른그리퍼 2. 인덱스 (0–2)(53–56)(3–9)(24,25)(28–34)(49,50), 공식 baseline 과 같은 부분집합. v1 은 61개 전부 |
| action | 23차원: base 속도 3 · torso 4 · 왼팔 7 · 왼그리퍼 1 · 오른팔 7 · 오른그리퍼 1. 시뮬레이터에는 절대 관절 목표로 보낸다. 모델 안에서 32차원으로 0 패딩 |
| 델타 액션 | 팔·몸통 18차원(action 3–13, 15–21)은 `action[t] − state(지금)` 으로 학습하고 출력 때 되돌린다. base 속도·그리퍼는 그대로 |
| 청크 | 50 스텝 (v1) / 32 스텝 (v2 부터). 실행은 앞 16 스텝 |
| 정규화 | openpi quantile 정규화(q01 → −1, q99 → +1). v1·v2 는 차원별 하나. **v3 부터 청크 위치별 (32, 23)**: 델타 17차원은 위치마다 따로, 나머지는 위치 공통 |
| 액션 6번 | torso 4번째 관절. q99 − q01 이 약 0.0018 rad 라 위치별 정규화에서 뺀다(키우면 잡음만 커진다) |
| 통계 | 설정마다 `asset_id` 를 따로 두고(`jiu121/behavior-cooking[-top20][-top20-v3][-top32-v5]`) 부분집합 필터와 같은 이름으로 묶는다. v3 통계는 40만 프레임(청크 399,872개) |

위치별 정규화를 넣은 이유: 델타 액션은 청크 뒤로 갈수록 커져서, 통합 통계로는 팔 관절의 정규화 후 표준편차가 청크 첫 스텝 0.061 · 마지막 스텝 0.434(분산 19배)였다. 실행하는 앞 16 스텝이 학습 신호를 가장 적게 받았다.

## 5. 데이터

| 항목 | 값 |
|---|---|
| 출처 | `behavior-1k/2026-challenge-demos` (LeRobot), 태스크당 200 에피소드 |
| 태스크 | 아래 15개 × 200 = 3,000 에피소드 |
| 단위 | 에피소드가 아니라 스킬 세그먼트. 세그먼트 하나가 LeRobot 에피소드 하나가 되고 문장이 task 로 들어간다 |
| 이동 구간 | `move to ...` 세그먼트는 시드 0 으로 25%만 남긴다 |
| 결과 | 66,892 세그먼트 / 26,579,202 프레임 / 65.6 GB (224p) |
| 지시문당 상한 (v3 부터) | 문장마다 에피소드를 섞어 10만 프레임까지만 담는다(`configs/task_subset.py`). TOP20 은 1,939,958 프레임(최대/최소 1.44배, 상한 없으면 26.8배), TOP32 는 3,148,879 프레임(NEW12 는 전부 상한에 걸린다) |
| 에폭 | TOP20 1 에폭 = 60,624 스텝 (배치 32) |
| 표집 | openpi 데이터 로더의 셔플. 재개할 때 데이터 위치를 복원하지 않으므로 차수마다 시드를 바꾼다 |

| # | 태스크 | task_index | # | 태스크 | task_index |
|---|---|---|---|---|---|
| 1 | can_meat | 4 | 9 | make_pizza | 49 |
| 2 | make_microwave_popcorn | 40 | 10 | cook_a_brisket | 51 |
| 3 | cook_cabbage | 41 | 11 | halve_an_egg | 62 |
| 4 | chop_an_onion | 42 | 12 | thawing_frozen_food | 70 |
| 5 | slicing_vegetables | 43 | 13 | cook_a_frozen_pie | 72 |
| 6 | cook_hot_dogs | 45 | 14 | cook_brussels_sprouts | 95 |
| 7 | cook_bacon | 46 | 15 | cook_broccolini | 96 |
| 8 | canning_food | 48 | | | |

태스크를 고른 기준은 "요리인가"가 아니라 **"이동·집기·놓기 외의 스킬을 공급하는가"**다. 후보 26개의 주석에서 스킬 구성을 세어 골랐고,
전체 데이터의 `chop` 12.0시간 중 9.9시간(82%)이 이 15개에 있다. 고른 뒤 비중은 chop 0.6% → 3.4%, pour 0.7% → 3.0%, 이동 38.4% → 21.9% 다.

## 6. 학습 설정

| 차수 | run-id | 설정 | 이어받은 가중치 | 스텝 | peak lr | 시드 | GPU | 초/스텝 | 벽시계 |
|---|---|---|---|---:|---:|---:|---|---:|---|
| 1 | `pi05v1-pi05base-10k` | `pi05_cooking` (v1) | π0.5 base | 10,000 | 5e-5 | 42 (기본값) | A100 80GB | 3.4 | 9.4 h |
| 2 | `pi05v2-pi05base-13k` | `pi05_cooking_top20` (v2) | π0.5 base | 13,000 | 5e-5 | 기록 없음 (기본값 42) | A100 80GB | 2.9 | 약 10.5 h (중간 재개 1회) |
| 3 | `pi05v3-run2-12.5k` | `pi05_cooking_top20_v3` (v3) | run2/12999 | 12,500 | 5e-5 | 42 | A100 80GB | 2.8 | 기록 없음 |
| 4 | `pi05v3-run3-11.5k` | v3 | run3/12499 | 11,500 (세션 끊김) | 1e-4 | 43 | A100 40GB | 기록 없음 | 기록 없음 |
| 5 | `pi05v3-run4-12.5k` | v3 | run4/11500 | 12,500 | 1e-4 | 44 | A100 40GB | 3.0 | 기록 없음 |
| 6 | `pi05v3-run5-25k` | v3 | run5/12499 | 25,000 (계획 27,000) | 1e-4 | 45 | A100 40GB | 3.0 | 23.2 h |
| 7 | `pi05v3-run6-26k` | v3 | run6/25000 | 26,000 | 1e-4 | 46 | A100 40GB | 기록 없음 | 기록 없음 |
| 8 | `pi05v4-run7-21k` | `pi05_cooking_top20_v4` (v4) | run7/25999 | 21,000 | 1e-4 | 47 | A100 80GB | 3.5 | 기록 없음 |
| 9 | `pi05v5-run7-24k` | `pi05_cooking_top32_v5` (v5) | run7/25999 | 24,000 | 1e-4 | 48 | A100 40GB | 3.0 | 20.1 h |
| 10 | `pi05v5-run9-24k` | v5 | run9/23999 | 24,000 | 1e-4 | 49 | 기록 없음 | 3.0 | 19.9 h |

- 6차는 26,500 스텝까지 돌았지만 마지막 세 번의 저장이 Drive 에 다 올라가지 못해 25,000 을 쓴다. 8차의 마지막 체크포인트 20999 는 목차 파일이 빠져 열리지 않아 평가는 18000 으로 했다.
- 이어받는 차수는 가중치(`params/`, EMA)만 넘기고 옵티마이저 상태와 lr 위치는 새로 시작한다. 그래서 차수마다 warmup 을 다시 한다.

공통 설정:

| 항목 | 값 |
|---|---|
| 배치 | 32 (GPU 한 장). 데이터 로더 워커 8 |
| 옵티마이저 | AdamW (openpi 기본값: β = (0.9, 0.95), eps 1e-8, weight decay 1e-10), 전역 그래디언트 클립 1.0 |
| lr 일정 | warmup min(1000, 스텝/10) → peak → 코사인으로 5e-6 까지. `decay_steps` = 총 스텝 |
| peak lr | 설정 기본값 5e-5 (1~3차). 4차에서 1e-4 로 올렸고, 오프라인 지시문·카메라 교란 배율이 3차보다 나아서(1.55 → 1.69, 2.20 → 3.33) 이후 유지했다 |
| 체크포인트 | 저장 간격 500~2,000 스텝. 한 개 약 18 GiB (v3·v5, `train_state` 포함), 추론용 `params`+`assets` 는 약 6 GiB. v4 는 46.1 GiB |

## 7. 평가 프로토콜

### 7.1 오프라인 — 정책이 문장을 읽는가

| 검사 | 사용 차수 | 방법 |
|---|---|---|
| `scripts/ablate_policy.py` | 1~5차 | 학습 데이터 50표본에서 지시문(또는 카메라)을 다른 표본 것으로 바꿨을 때 예측 오차가 정상의 몇 배가 되나. 2차 이후 판정 기준 1.40배 |
| `scripts/probe_language.py` | 3차~ | 학습 데이터 배치 16표본, 확산 잡음 고정(`jax.random.key(0)`). 조건: 정상 / **지시문만** 옆 표본 것으로 교체(state 토큰은 제 것) / 통째 교체(state 까지) / 잡음만 교체. 지표는 정규화 공간 액션 청크 L1 이 정상 대비 늘어난 비율 |

- 2026-09-25 전의 `probe_language.py` 는 "통째 교체"만 쟀다. π0.5 는 state 를 지시문과 같은 문자열에 넣으므로 그 값에는 자세를 바꾼 효과가 섞인다. 이후 값은 **지시문만** 교체한 것을 쓰고, 문자열로 되돌렸다가 다시 토큰화했을 때 원래 토큰과 1개라도 다르면 멈춘다(왕복 검사).
- 정규화 통계가 다른 모델끼리는 이 오프라인 오차를 견주지 않는다. 같은 배치를 상대 모델 기준으로 다시 정규화하면 완벽한 모델도 오차가 0.028~0.030 나와, 두 모델의 차이보다 크다. 7·9·10차는 같은 32종 배치 파일로 쟀다.

### 7.2 폐루프 — 시뮬레이터에서 스킬을 해내는가

`scripts/rollout_b1k.py` (OmniGibson, 공식 평가와 같은 부분 장면 로딩).

| 항목 | 값 |
|---|---|
| 장면 | 그 데모가 녹화된 장면 인스턴스(`task_instance_id`, train 분할). 인스턴스가 다르면 로봇 시작 위치와 물건 배치가 달라 인계 자세가 어긋난다 |
| 인계 | 데모의 23차원 액션을 스킬 k 의 시작 프레임까지 재생한 뒤(팔·몸통은 절대 위치, 바퀴는 속도) 정책에 넘긴다. 프롬프트는 스킬 k 의 주석 문장 |
| 무대 검사 | 인계 순간의 관절 자세와 **손에 쥔 물체**를 데모와 대조한다. 어긋난 판(재생이 물체를 놓친 경우 등)은 모델 성적에서 뺀다 |
| 정책 구간 | 2,700 스텝(90초). 데모의 스킬이 짧아도 같다. 조기 종료 끔 |
| 추론 | CPU, 호출당 약 9.5초. 판마다 정책 서버를 새로 띄운다(한 서버에서 이어 돌리면 난수열이 이어진다) |
| 판정기 | 문장 형태별 시뮬레이터 상태: 열기·닫기 = `Open` / 켜기 = `ToggledOn` / 집기 = 목표 물체 `is_grasping` / 놓기 = `OnTop`·`Inside` / 자르기 = 목표 물체가 잘려 사라짐. 15 스텝 연속 유지되면 성공. 문은 가동범위 대비 열린 비율(최대·끝)도 기록한다 |
| 대조군 | `--null-policy`: 같은 무대에서 정책에 묻지 않고 인계 자세를 유지한다. 여기서 성공이 찍히면 그 무대의 성공은 정책의 성과가 아니다 |
| 영상 판독 | 사람이 정책 구간의 영상을 본다. **판정기와 영상이 다르면 영상을 따른다** |

판정기의 알려진 한계(2026-09-28 기준, 고치지 않음):

- `Open` 은 관절 가동범위의 **5%** 만 넘어도 참이다. 문 열기는 열린 비율로 따로 본다.
- 같은 이름의 물체가 여럿이면 목록의 첫 번째를 목표로 잡는다(도마 둘, 피망 둘 등). 해당 판은 끝 좌표로 다시 센다.
- 판정기를 재생 전에 만들어서 `half X` 문장이 재생 중에 생기는 반쪽 물체가 아니라 통째 물체를 가리킬 수 있다(자르기·반쪽 집기).
- 무대 검사는 손에 쥔 것만 보고 로봇 위치는 보지 않는다(`brisket_pick_fridge` 는 인계 때 냉장고에서 3.8 m 떨어져 있었다).

### 7.3 평가 스위트

| 스위트 | 스크립트 | 판 | 내용 |
|---|---|---:|---|
| 기존 20종 | `rollout_all20.sh` | 21 | TOP20 각 1판 + 같은 무대에서 문장만 바꾼 A/B 1판(`micro_on_ab`) |
| 신규 12종 | `rollout_new12.sh` | 12 | NEW12 각 1판 |
| 일반화 | `rollout_unseen.sh` | 13 | 표현 바꾸기 4(무대는 학습 문장 판과 같고 문장만 바꿈) + 처음 보는 문장·물체 6 + 추가 3 |
| 대조군 | `rollout_null_control.sh`, `NULL_POLICY=1 rollout_new12.sh` | — | 위 무대에서 정책 없이 |

이슈에 나오는 판 이름과 문장:

| 판 이름 | 문장 | 판 이름 | 문장 |
|---|---|---|---|
| `fridge_open2` | open the door of the fridge | `knife_pick` | pick up the carving knife from the cutting board |
| `fridge_close` | close the door of the fridge | `knife_pick_counter` | pick up the carving knife from the countertop |
| `micro_open` | open the door of the microwave | `knife_back` | put the carving knife back on the countertop to the right of the cutting board |
| `micro_close` | close the door of the microwave | `board_pick` | pick up the cutting board from the countertop |
| `micro_on` · `micro_on_ab` | turn on the microwave | `half_pick` | pick up the half brussels sprouts from the cutting board |
| `oven_open` | open the door of the oven | `brussels_pick` | pick up the brussels sprouts from the cutting board |
| `oven_close` | close the door of the oven | `half_put` | put the half brussels sprouts on the baking sheet |
| `oven_on` | turn on the oven | `brussels_chop` | chop the brussels sprouts with the carving knife |
| `jar_put` | put the bratwurst in the hinged jar | `beet_chop` | chop the half beet with the parer |
| `bratwurst` | pick up the other bratwurst from the cutting board | `pepper_chop` | chop the half bell pepper with the parer |
| `board_pick_bar` | pick up the cutting board from the bar | `hotdog_put_micro` | put the hotdog in the microwave |
| `board_pick_sink` | pick up the cutting board from the drop in sink | `tray_put_oven` | put the tray in the oven |
| `knife_pick_bar` | pick up the carving knife from the bar | `beet_put_board2` | put the beet on the cutting board |
| `parer_pick_bar` | pick up the parer from the bar | `pepper_put_board2` | put the bell pepper on the cutting board |
| `plate_pick_fridge` | pick up the plate from the fridge | `board_back_counter` | put the cutting board back on the countertop |
| `brisket_pick_fridge` | pick up the brisket from the fridge | `jar_lid_open` | open the lid of the hinged jar |
| `para_micro_on` | switch on the microwave | `para_micro_open` | open the microwave |
| `para_knife_pick` | grab the knife off the cutting board | `para_fridge_close` | shut the fridge |

미학습 문장 9종(`unseen_*`, 학습에 절대 넣지 않는다): turn on the burner / chop the head cabbage with the carving knife /
pick up the head cabbage from the fridge on the high shelf / put the head cabbage on the cutting board / open the right door of the top cabinet /
pick up the chili from the fridge on the high shelf / close·open the door of the electric refrigerator / pick up the tupperware from the electric refrigerator.

판마다 데모·인스턴스·스킬 번호는 각 스크립트의 `PLAN` 에 있다.

## 8. 공식 baseline · 2025 1위와 다른 점

| 항목 | 공식 `pi05_b1k` | 2025 1위 (Robot Learning Collective) | 여기 |
|---|---|---|---|
| 정책 수 | 태스크당 1개 | 최종 제출 4개 | 1개 |
| 언어 | 태스크당 문장 1개 | 쓰지 않음 (태스크 임베딩) | 스킬 문장 20~32종으로 동작을 고른다 |
| 데이터 단위 | 에피소드 통째 | 에피소드 통째 + 단계 예측 보조 과제 | 스킬 세그먼트 |
| 학습 범위 | 전체 파인튜닝 | SigLIP 만 동결 | SigLIP·Gemma 2B 본체 동결 + LoRA (v3·v5) |
| 학습량 | 태스크당 50,000 스텝 × 32 | 8×H200 15일, 약 2 에폭 | 차수당 10k~26k 스텝 × 32. 3~7차 누적 87,500 스텝 = TOP20 표본 수 기준 약 1.4 에폭 |
| 정규화·손실 | openpi 기본 (청크 위치 공통) | 위치별 정규화, 다표본 15 | 위치별 정규화(v3~), 다표본 8 |
| state / 청크 | 25 / 32 | — | 25 / 32 (v2~) |

## 9. 하드웨어 · 자원

| 항목 | 값 |
|---|---|
| 학습 | Colab A100 80GB (1~3·8차) / A100 40GB (4~7·9차). v4 는 VRAM 최대 73.9 GiB 라 40 GB 에 안 들어가고 80 GB 에서 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.97` 이 필요했다 |
| 초/스텝 | v1 3.4 (A100 80GB) · v2 2.9 · v3 2.8 (80GB) / 3.0 (40GB) · v5 3.0 (40GB) · v4 3.5 (A100 80GB) / 3.3 (RTX PRO 6000 96GB) |
| Drive 업로드 | 약 11.5 MB/s. v4 체크포인트 하나에 약 68분 — 세션 끝 여유가 이보다 짧아 8차 마지막 체크포인트를 잃었다 |
| 로컬 평가 | RTX 5060 Laptop 8 GB: 시뮬레이터 약 5.5 GiB, 정책은 CPU(가중치 약 6.8 GiB). 추론 약 9.5초/회, 롤아웃 한 판 40~47분 |
| 한 차수 평가 | 언어 검사 3회 + 롤아웃 46판 ≈ 34시간 (`eval_chain.sh`) |
| 데이터 변환 | 15 태스크 16.0시간, 피크 디스크 27.5 GB (태스크 단위로 받고 지운다) |
