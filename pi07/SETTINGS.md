# π0.7 재현 — 실험 세팅

π0.7([arXiv:2604.15483](https://arxiv.org/abs/2604.15483))의 학습 방법을 24 GB GPU 2장 서버에 맞춰 옮긴 재현 실험의 설정이다. 결과와 해석은 [이슈](https://github.com/inner-king/policy/issues)로 따로 올린다. 코드·환경·실행 방법은 [README](README.md) 에 있다.
이 문서의 수치는 `runs/*/run.yaml`, 코드 기본값, 실행 로그에서 가져왔다. 기록이 뒤에 정정된 경우(subtask 문장 수, HL 전환 표본 비율, LL 검증 청크 길이)에는 정정값을 쓴다.

## 1. 구성

| 모듈 | 입력 → 출력 | 구현 | 실행 id |
|---|---|---|---|
| HL (고수준 정책) | 3뷰 관측 + 태스크 문장 + 지시 이력(최대 8개) → 지금 유효한 subtask 문장 | π0.5 base 의 PaliGemma + LoRA | `pi07hl-pi05base-8k` |
| LL (저수준 VLA) | 3뷰 관측 + π0.7 프롬프트 + 이산 state (+ 서브골 3뷰) → 50 스텝 × 23차원 액션 청크 | π0.5 base, action expert 만 학습 | `pi07ll-pi05base-30k`, `pi07ll-pi05base-200k` |
| WM (서브골 월드 모델) | 현재 3뷰 + 태스크 + subtask → 현재 세그먼트 끝의 3뷰 | FLUX.2 [klein] base 4B + LoRA | `pi07wm-flux2kleinbase-6k` |
| 생성 서브골 혼합 (π0.7 VI-C) | WM 이 만든 서브골 4,000장을 섞어 LL 추가 학습 | LL 30k 마지막 체크포인트에서 시작 | `pi07ll-pi05base-30k-genmix-5k` |

- 학습 순서: GPU1 에서 HL 8k → WM 6k(`chain.sh`) → 서브골 4,000장 생성, GPU0 에서 LL 30k → genmix 5k(`stage3.sh`). LL 200k 는 LL 30k 와 같은 레시피를 GPU 2장 데이터 병렬로 처음부터 다시 돌린다.
- 런타임(`src/pi07/serve_pi07.py`, π0.7 Algorithm 1): HL 은 30 스텝마다, WM 은 subtask 가 바뀌거나 4초가 지나면, LL 은 25 스텝마다 50 스텝 청크를 새로 뽑는다(디노이징 5회). HL 과 LL 은 PaliGemma 가중치 하나를 나눠 쓴다(HL = LoRA 켬, LL = LoRA 끔). 폐루프 평가에서는 WM 을 쓰지 않았다.

## 2. 모델

### 2.1 LL

| 항목 | 값 |
|---|---|
| 가중치 | `lerobot/pi05_base` (openpi π0.5 base 의 PyTorch 포트), 토크나이저는 `google/paligemma-3b-pt-224` |
| VLM | PaliGemma 3B = SigLIP 400M + Gemma 2B (width 2048, 18층). bf16, 동결 |
| action expert | Gemma 300M 변형 (width 1024, 18층, mlp 4096, head 8 / KV head 1) |
| 학습 파라미터 | 430.1M = expert + `action_in_proj`·`action_out_proj`·`time_mlp`. fp32 가중치 + bf16 autocast |
| prefix | 관측 3뷰 × 256 토큰 + 텍스트(상한 256 토큰, 실측 최대 170) [+ 서브골 3뷰 × 256] |
| VLM 경로 | prefix 를 `no_grad` 로 한 번 통과시켜 KV 캐시만 expert 에 넘긴다 |
| 서브골 블록 | 텍스트 뒤에 붙는 block-causal 블록. 서브골 토큰은 앞을 전부 보고 관측·텍스트는 서브골을 보지 않는다. 서브골이 없을 때 prefix 는 π0.5 사전학습과 같다 |
| 흐름 매칭 | x_t = t·ε + (1−t)·a, 목표 u = ε − a. MSE 는 23차원에만 건다(패딩 9차원 제외). 에피소드 끝을 넘는 스텝은 마스크 |
| 시간 표집 | t = 0.001 + 0.999·Beta(1.5, 1.0) (LeRobot π0.5 기본값) |
| 추론 | t: 1 → 0 Euler 5 스텝, 초기 잡음 [1, 50, 32] |

### 2.2 HL

| 항목 | 값 |
|---|---|
| 백본 | π0.5 base 의 PaliGemma 만 (action expert 제외) |
| LoRA | 언어 모델 18층 × {q, k, v, o, gate, up, down}_proj, r=32, α=32, dropout 0.05. A 는 kaiming uniform, B 는 0 으로 초기화 |
| 학습 파라미터 | 39.2M (LoRA 만). 나머지는 bf16 동결 |
| 구조 | prefix-LM: 이미지 3뷰 + 프롬프트는 양방향, 정답 문장 + EOS 는 인과. LM head(입력 임베딩과 묶임)는 정답 위치에서만 계산 |
| 메모리 | 층 단위 그래디언트 체크포인팅 |
| 디코딩 | 탐욕, 최대 32 토큰. 학습 중 검증은 제약 없이, 런타임은 태스크별 subtask 문장 트라이로 제약(7절) |

### 2.3 WM

| 항목 | 값 |
|---|---|
| 가중치 | `black-forest-labs/FLUX.2-klein-base-4B` (Apache-2.0, 증류하지 않은 base) |
| 입출력 | 3뷰를 머리 \| 왼손목 \| 오른손목 순으로 이어 붙인 672×224 한 장. 현재 타일을 FLUX.2 참조 이미지(위치 id T=10)로 넣고, 정답은 세그먼트 끝 타일 |
| 텍스트 | `Task: {태스크}. Subtask: {subtask}.` 를 Qwen3 채팅 템플릿에 넣고 Qwen3-4B 층 9/18/27 은닉 상태를 미리 계산해 캐시(671문장, 64토큰). 학습 중 텍스트 인코더는 올리지 않는다 |
| LoRA | double-stream q/k/v/out·add_q/k/v·to_add_out·ff, single-stream qkv_mlp·out. r=64, α=64 → 96.3M |
| 동결 | transformer bf16, VAE fp32. 그래디언트 체크포인팅 켬 |
| 잡음 일정 | σ 는 logit-normal 에 FLUX.2 `compute_empirical_mu`(시퀀스 588) 이동. 목표 속도 ε − x₀ |
| CFG 용 드롭 | 학습 시 10% 빈 문장 |
| 생성 | Euler, 기본 28 스텝 · CFG 4.0. 서브골 대량 생성은 12 스텝 · CFG 4.0 |

## 3. 프롬프트 · 드롭아웃 · 서브골

LL 프롬프트. 빠진 항목은 생략된다.

```
Task: {태스크 문장}. Subtask: {subtask}. Speed: {speed}. Quality: {quality}. Mistake: {true|false}. Control Mode: joint. State: {0~255 정수 25개};
Action:
```

HL 프롬프트와 정답. 이력이 비면 `none`, 이력 드롭아웃이면 `unknown`.

```
Task: {태스크 문장} Subtask history: {h1; …; h_last}. Current subtask:   →   " {subtask}<eos>"
```

드롭아웃·표집 비율:

| 대상 | 규칙 |
|---|---|
| LL 서브골 이미지 | 배치의 25% (마이크로배치 16 중 4). 서브골 유/무를 별도 데이터로더 스트림으로 채운다 |
| LL subtask | 서브골 있는 예시에서 30%, 없는 예시에서 10% 제거. Subtask 항이 없는 프롬프트는 학습 표본의 15% |
| LL 메타데이터 | 통째로 15% 제거, 남은 경우 speed·quality·mistake 항목마다 5% 제거 |
| LL Control Mode | 드롭하지 않음 (항상 `joint`) |
| HL 이력 | 10% 를 `unknown` 으로 |
| WM 텍스트 | 10% 빈 문장 |
| 검증 표본 | 드롭아웃 전부 0 |

메타데이터 값:

| 항목 | 학습 | 런타임 |
|---|---|---|
| Speed | 에피소드 길이를 500 스텝 단위로 반올림 | 태스크별 train 에피소드 길이의 15 백분위를 500 단위로 반올림 |
| Quality | 상수 5 | 5 |
| Mistake | 상수 false | false |
| Control Mode | joint | joint |

Quality·Mistake 를 상수로 둔 이유: B1K 데모에 해당 라벨이 없다. 연속 반복 세그먼트를 재시도로 보고 mistake 를 유도하면 1,601건 중 1,597건이 freeze_fruit·halve_an_egg 에서 나오는데, 이는 스킬 하나를 같은 라벨 두 조각으로 나눈 주석 관행이다.

subtask 문장:

| 항목 | 값 |
|---|---|
| 생성 | B1K 스킬 주석 한 세그먼트(스킬 19종 × object_id × memory/spatial 수식어)를 영어 명령문으로 바꾼다. 객체 id 끝의 숫자와 6글자 모델 id 는 지운다 |
| 수 | 세그먼트 94,438개 → 고유 문장 540종 (index.json 기준. 이전 기록의 542 는 정정됨) |
| LL 의 subtask | 프레임 t 에 활성인 세그먼트의 문장. 주석 구간 밖 프레임(프레임 기준 0.9%, 에피소드 평균 1.3%)은 다음 세그먼트, 마지막 세그먼트 뒤(0.4%)는 마지막 세그먼트 |

HL 표본:

| 항목 | 값 |
|---|---|
| 프레임 | 50% 는 에피소드 전체 균등, 50% 는 세그먼트 경계 ±1.5초 |
| 이력 | 직전 질의 시점 t − Δ 까지 내려진 지시(그 시점에 유효한 지시 포함), Δ ~ U[0.5, 1.5]초, 연속 중복은 하나로, 최대 8개. 런타임에서 HL 이 자기 직전 출력을 이력 끝에 두는 것과 같은 정의 |
| 정답 | 프레임 t 에 활성인 세그먼트 문장 (오프라인 주석) |
| 전환 비율 | 학습 표본의 20.0% 에서 이력의 마지막 ≠ 정답 (이전 기록의 20.6% 는 정정됨) |

서브골:

| 용도 | 규칙 |
|---|---|
| LL 학습 (실제 프레임) | 서브골 예시의 25% 는 현재 세그먼트의 마지막 프레임, 75% 는 t + U[0, 4]초 뒤 프레임. [t, 에피소드 끝] 으로 자른다 |
| WM 정답 | 현재 세그먼트의 마지막 프레임 (max(end − 1, t)) |
| genmix (생성 이미지) | 서브골 예시의 50% 를 WM step-006000 이 만든 이미지로 바꾼다. train 분할에서 균등하게 뽑은 (에피소드, t) 4,000개, 12 스텝, CFG 4.0, 뷰당 224 |
| 런타임 | subtask 가 바뀌거나 4초가 지나면 WM 을 부른다 (평가에서는 WM 미사용) |

## 4. 관측 · 상태 · 액션 · 정규화

| 항목 | 값 |
|---|---|
| 이미지 | RGB 3뷰: head(`zed_link_camera_0`), left_wrist(`left_realsense_link_camera_0`)·right_wrist(`right_realsense_link_camera_0`). 학습은 HEVC 영상에서 디코딩해 224×224 로 면적 축소, SigLIP 입력 [−1, 1] |
| 주기 | 30 Hz |
| state | 61차원 proprio 중 25차원: base_qvel 3 · torso 4 · left_arm 7 · left_gripper 2 · right_arm 7 · right_gripper 2 (GR00T r1pro 와 같은 부분집합, 인덱스 (0,3)(53,57)(3,10)(24,26)(28,35)(49,51)) |
| state 입력 방식 | 정규화 뒤 [−1, 1] 로 자르고 256구간 정수로 바꿔 텍스트 토큰으로 넣는다 (π0.5 방식) |
| action | 23차원: base 3 · torso 4 · left_arm 7 · left_gripper 1 · right_arm 7 · right_gripper 1. 모델 안에서는 32차원으로 0 패딩 |
| 청크 | 50 스텝 (1.67초). 에피소드 끝을 넘으면 마지막 액션 반복 + 패딩 마스크 |
| 정규화 | q01 → −1, q99 → +1, ±1.5 에서 자름. 통계는 train 분할에서 균등하게 뽑은 2M 프레임 (`data/pi07/stats.json`) |
| 역정규화 (런타임) | (a + 1) / 2 · (q99 − q01) + q01 |
| 퇴화 차원 | action 6번(torso 4번째)은 q01 = q99 = 0 이라 정규화값이 항상 −1 |

## 5. 데이터

| 항목 | 값 |
|---|---|
| 출처 | BEHAVIOR-1K 2026 챌린지 데모 `behavior-1k/2026-challenge-demos`, 태스크당 200 에피소드. clean_a_keyboard 주석은 task-0093 annotations |
| 에피소드 | 3,600 = 18 태스크 × 200 |
| 분할 | 태스크마다 무작위 10개를 val 로 (seed 0) → train 3,420 / val 180 |
| 프레임 | 전체 37,860,087, train 35,957,659 |
| 산출물 | `index.json`(에피소드·세그먼트·speed), `lowdim.npy`(float32 [37,860,087, 48] = action 23 + state 25, memmap), `stats.json` |
| 표집 | LL·WM 은 프레임 균등 (에피소드를 길이 비례로 고르고 t 균등). 에폭 개념 없이 무한 표집 |

18개 태스크 (17 요리 + clean_a_keyboard). `Task:` 에 들어가는 문장은 `src/pi07/build_index.py` 에 있다.

| # | 태스크 | task_index | 프레임 | 태스크 문장 |
|---|---|---|---|---|
| 1 | can_meat | 4 | 2,369,415 | Put the bratwurst into the jars, close them, and place the jars in the cabinet. |
| 2 | make_microwave_popcorn | 40 | 647,555 | Cook the popcorn in the microwave. |
| 3 | cook_cabbage | 41 | 2,828,962 | Cook the diced cabbage and chili in the frying pan. |
| 4 | chop_an_onion | 42 | 1,279,960 | Chop the onion and put the diced onion into the bowl. |
| 5 | slicing_vegetables | 43 | 2,968,988 | Slice the zucchini, bell pepper, and beet. |
| 6 | cook_hot_dogs | 45 | 1,828,971 | Cook the hot dogs. |
| 7 | cook_bacon | 46 | 1,535,909 | Cook the bacon in the frying pan. |
| 8 | freeze_pies | 47 | 2,491,046 | Put the apple pies into the containers and place them in the freezer. |
| 9 | canning_food | 48 | 4,595,098 | Dice the steak and pineapple, and put each into a separate bowl. |
| 10 | make_pizza | 49 | 3,837,385 | Make the pizza and place it on the baking sheet. |
| 11 | freeze_fruit | 50 | 2,523,756 | Put the apples and strawberries into the containers and place them in the freezer. |
| 12 | cook_a_brisket | 51 | 1,466,999 | Cook the brisket in the oven. |
| 13 | halve_an_egg | 62 | 1,275,408 | Cut the hard-boiled egg in half. |
| 14 | thawing_frozen_food | 70 | 1,645,924 | Thaw the frozen chicken breast. |
| 15 | cook_a_frozen_pie | 72 | 1,733,592 | Cook the frozen apple pie on the tray. |
| 16 | cook_brussels_sprouts | 95 | 3,132,354 | Halve the brussels sprouts and cook them in the oven. |
| 17 | cook_broccolini | 96 | 923,552 | Cook the broccolini and garlic in the frying pan on the stove. |
| 18 | clean_a_keyboard | 93 | 775,213 | Push the chair under the desk, pick up the duster brush, and wipe the dust off the keyboard with the brush. |

학습 노출량 (스텝 × 유효 배치):

| 실행 | 표본 수 | 에폭 |
|---|---|---|
| LL 30k | 960,000 (서브골 있음 240,000) | 0.0267 |
| HL 8k | 256,000 | 0.0071 |
| WM 6k | 96,000 | 0.0027 |
| genmix 5k | 160,000 (생성 이미지 4,000장 × 평균 5.0회) | 0.0044 |

## 6. 학습 설정

실행별 하이퍼파라미터. 유효 배치 = 마이크로배치 × 누적 × GPU 수.

| 실행 | 모듈 | 시작 가중치 | 학습 파라미터 | 스텝 | 마이크로배치 × 누적 × GPU = 유효 배치 | lr (peak → final) | warmup | save / val / milestone | 워커 | GPU | 벽시계 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `pi07hl-pi05base-8k` | HL | π0.5 base PaliGemma | LoRA 39.2M | 8,000 | 16 × 2 × 1 = 32 | 1e-4 → 1e-5 | 300 | 500 / 1,000 / 2,000 | 4 | GPU1 (3090) | 20.0 h (수동 정지·재개 2회 포함, 크래시 재시작 0) |
| `pi07ll-pi05base-30k` | LL | π0.5 base | 430.1M | 30,000 | 16 (서브골 없음 12 + 있음 4) × 2 × 1 = 32 | 2.5e-5 → 2.5e-6 | 1,000 | 1,000 / 2,000 / 10,000 | 5 | GPU0 (3090 Ti) | 28.0 h (크래시 재시작 0) |
| `pi07wm-flux2kleinbase-6k` | WM | FLUX.2 klein base 4B | LoRA 96.3M | 6,000 | 8 × 2 × 1 = 16 | 1e-4 → 1e-5 | 300 | 500 / 1,000 / 2,000 | 3 | GPU1 (3090) | 15.4 h (09-19 09:12 → 09-20 00:38) |
| `pi07ll-pi05base-30k-genmix-5k` | LL | LL 30k step-030000 의 학습 파라미터 (옵티마이저 새로) | 430.1M | 5,000 | 16 (12 + 4) × 2 × 1 = 32 | 1e-5 → 2.5e-6 | 200 | 1,000 / 1,000 / 5,000 | 5 | GPU0 (3090 Ti) | 4.6 h. 앞서 서브골 생성 4.5 h (GPU1, 0.25 장/s) |
| `pi07ll-pi05base-200k` | LL | π0.5 base (처음부터) | 430.1M | 200,000 | rank 당 16 (12 + 4) × 1 × 2 = 32 | 2.5e-5 → 2.5e-6 | 1,000 | 2,000 / 2,000 / 10,000 | rank 당 4 | GPU0 + GPU1 (torchrun 2 rank) | 진행 중 (run.yaml 예상 97 h) |

공통 설정:

| 항목 | 값 |
|---|---|
| 옵티마이저 | AdamW, β = (0.9, 0.95), eps 1e-8, weight decay 0.01 |
| lr 일정 | 선형 warmup ((step + 1) / warmup) 뒤 peak → final 코사인 |
| 기본 lr | 인자를 주지 않으면 HL 1e-4 → 1e-5, LL 2.5e-5 → 2.5e-6 (openpi π0.5 파인튜닝 기본값), WM 1e-4 → 1e-5 |
| 그래디언트 클립 | 전역 norm 1.0 (세 모드 공통) |
| 정밀도 | 동결 부분 bf16 (WM VAE 는 fp32), 학습 파라미터 fp32 + bf16 autocast, TF32 켬 |
| LL 두 스트림 | 서브골 없음 / 있음 데이터로더를 따로 돌리고 손실을 0.75 / 0.25 로 가중한다. 로그의 train loss 는 두 스트림의 단순 평균 |
| LL 액션 지평 / 추론 디노이징 | 50 / 5 |
| 데이터 병렬 (200k) | DDP 모듈 없이 그래디언트를 평탄화 all-reduce 후 평균하고, 클립은 평균 뒤에 건다. rank 별 데이터 시드가 다르다. 검증·저장은 rank 0 |
| 시드 | `--seed 0`. 데이터 시드는 (base seed, rank, 워커, 재개 스텝)에서 유도 |
| 체크포인트 | 학습 파라미터 + 옵티마이저 + 스케줄러. 마지막 2개 + milestone 배수를 남기고 milestone 은 옵티마이저 상태를 지운다. 크기 LL 5.2 GB, HL 0.5 GB |
| 재개 | 같은 launch 명령으로 다시 띄우면 최신 체크포인트에서 이어서 돈다 |
| 메모리 가드 | MemAvailable < 2 GB 이면서 SwapFree < 8 GB 이면 체크포인트를 쓰고 exit 75 |

학습 중 검증:

| 모듈 | 방법 |
|---|---|
| HL | 검증 에피소드, 질의 간격 Δ = 1초. 1초 격자 균등 256 + 경계 직후 0~1초 64 + 경계 직전 0~1초 64 = 384 표본 고정 (전환 표본 76). 제약 없는 탐욕 생성의 exact match 를 전체·전환·유지로 나눠 잰다. 이 집합에서 "이력의 마지막 지시 유지" 기준선은 0.802, 균등 256 부분만은 0.9375 |
| LL (현재 코드) | 서브골 없음 96 + 있음 32. 있음 32 는 없음 앞 32 와 같은 (에피소드, t) 에 서브골만 붙인 짝이다. 드롭아웃 0, 잡음 시드 고정(flow 1234 / 샘플링 4321). flow 손실 + 5 스텝 Euler 로 뽑은 50 스텝 청크의 L1(정규화 공간, 관절군별·앞 5/25 스텝) + 직전 액션 반복 기준선 |
| LL (결과를 낸 코드) | 30k·genmix 는 수정 전 코드로 돌았다. L1 은 50 스텝 청크 기준(5 는 디노이징 횟수)이고, 스트림별 마지막 배치 16 표본 값만 기록됐으며, 두 스트림은 표본이 겹치지 않고 샘플링 잡음 시드가 없었다 |
| WM | 검증 32 표본 고정. σ ∈ {0.25, 0.5, 0.75} 고정 잡음 손실 + 첫 배치 8장을 28 스텝 · CFG 4.0 으로 생성한 PSNR. 기준선은 현재 관측을 그대로 서브골로 쓰는 것(복사) |

## 7. 평가 프로토콜

BEHAVIOR-1K 챌린지 하네스(`python -m omnigibson.eval.eval`)가 websocket 으로 `serve_pi07.py` 에 붙는다. 서버 하나가 포트별로 조건을 나눈다: 8010 = HL 켬, 8011 = HL 끔(태스크 문장만, 플랫 기준선).

| 실행 id | 태스크 | 조건 | 인스턴스 (public_test 인덱스) | 실행 스텝 | 메타 Speed |
|---|---|---|---|---|---|
| `pi07ll-pi05base-30k-at30k-thawing_frozen_food-ll-hl` | thawing_frozen_food | A: LL 30k + HL 8k, WM 없음 (포트 8010) | 0–4 로컬, 5–9 원격 머신 | 12,345 | 6500 |
| `pi07ll-pi05base-30k-at30k-thawing_frozen_food-ll` | thawing_frozen_food | B: LL 30k 단독, Subtask 항 없음 (포트 8011) | 0–4 로컬, 5–9 원격 머신 | 12,345 | 6500 |
| `pi07ll-pi05base-30k-at30k-clean_a_keyboard-ll-hl` | clean_a_keyboard | A (포트 8010) | 0–9 (인스턴스 301–310) 전부 로컬 | 5,815 | 2500 |
| `pi07ll-pi05base-30k-at30k-clean_a_keyboard-ll` | clean_a_keyboard | B (포트 8011) | 0–9 (인스턴스 301–310) 전부 로컬 | 5,815 | 2500 |
| `pi07ll-pi05base-30k-at30k-thawing_frozen_food-ll-hl-smoke1` | thawing_frozen_food | A, 배관 확인용 1 롤아웃 (`run_eval.sh`, 시뮬·서버 GPU0) | 0 | 하네스 기본값 | — |

공통 설정:

| 항목 | 값 |
|---|---|
| 체크포인트 | LL `pi07ll-pi05base-30k/checkpoints/step-030000`, HL `pi07hl-pi05base-8k/checkpoints/step-008000` |
| 실행 구간 Ĥ | 25 스텝 (50 스텝 청크 중 앞 25 스텝 실행) |
| LL 디노이징 | 5 스텝 |
| HL 주기 | 30 스텝 (30 Hz 에서 1초) |
| HL 이력 | 서로 다른 연속 출력, 최대 8개 |
| HL 제약 디코딩 | 태스크별 train 분할 subtask 문장 집합의 토큰 트라이로 탐욕 디코딩을 제한한다 |
| subtask 전환 시 | 남은 청크를 버리고 새 지시로 즉시 다시 뽑는다 |
| 메타 프롬프트 | `Speed: {태스크 15 백분위}. Quality: 5. Mistake: false. Control Mode: joint.` |
| 메타 CFG | 0 (끔) |
| WM | 미사용 |
| 환경 래퍼 | `b1k_wrappers.RGBFullResWrapper` (head 720×720, wrist 480×480, RGB 만). `B1K_ACTION_LOG_DIR` 가 있으면 적용된 액션을 에피소드별로 남긴다 |
| 관측 축소 | 로컬: 전체 해상도를 서버가 224×224 로 면적 축소. 원격 머신: 클라이언트가 224×224 면적 축소 + JPEG 품질 90 으로 보낸다(회선 2.0 MB/s), depth·seg 채널 제외 |
| 모드 · 인스턴스 | `public_test`, 공개 분할 20개 중 인덱스 0–9 만 (공식 태스크 점수 아님) |
| max_steps | 하네스 기본값 = int(사람 데모 평균 × 1.5): thawing 8229.62 × 1.5 → 설정 12,344, keyboard 3,876 × 1.5 → 설정 5,814. 하네스는 설정보다 1스텝 더 돌아 실행 스텝은 12,345 / 5,815 다 |
| 로컬 / 원격 분할 | thawing 은 두 조건을 같은 인스턴스 분할(0–4 로컬, 5–9 원격 머신)로 돌려 하드웨어·관측 경로 차이가 조건 비교에 섞이지 않게 했다. keyboard 는 전부 로컬 |
| 실행 순서 | 같은 머신에서 A → B 순차 (thawing 로컬 09-19 18:14–20:45, keyboard 09-20 13:37–15:36 UTC) |
| 서버 GPU | thawing 은 GPU0 (로컬 시뮬과 같은 3090 Ti). keyboard 는 로그에 남지 않았다 |

개루프 분석 스크립트 기본값: `src/analysis/probe_pi07_ll.py` 는 검증 에피소드 192 표본, 배치 16, 표본 K=8, 디노이징 5/10/20 비교.

## 8. 논문과 다른 점

| 항목 | π0.7 | 여기 |
|---|---|---|
| LL 백본 | Gemma3 4B VLM + MEM 비디오 인코더 + 860M flow-matching expert | π0.5 base: PaliGemma 3B (SigLIP 400M + Gemma 2B) + 300M expert |
| VLM 학습 | knowledge insulation: VLM 을 FAST 토큰 CE 로 학습, expert 그래디언트는 VLM 으로 가지 않음 | VLM 동결, action expert 만 학습. KI 와 달리 VLM 이 로봇 데이터로 학습되지 않는다 |
| 이력 프레임 | MEM 6프레임, 1초 간격 | 없음 (24 GB 에서 배치 불가) |
| 학습 시 RTC | 지연 0~12 스텝 시뮬레이션 | 없음. 평가 하네스가 추론 동안 시뮬레이터를 멈춘다 |
| state 입력 | 선형 투영 | π0.5 식 256구간 이산 텍스트 토큰 |
| quality / mistake | 사람 라벨 | 상수 5 / false |
| 실패·자율 롤아웃 데이터 | 사용 | 없음 (B1K 데모만) |
| 서브골 없는 예시의 subtask 제거 | 수치 없음 | 10% 추가 |
| HL | π0.7 자체(Gemma3 4B)를 사람 코칭 데이터로 파인튜닝, 자유 문장 출력 | π0.5 base PaliGemma + LoRA 39.2M, 정답은 오프라인 세그먼트 주석, 학습 때 이력은 정답 이력(추론 때는 자기 출력). 런타임 출력은 학습 문장 집합으로 제약 |
| WM | BAGEL-14B, 뷰당 512×384, 메타데이터 조건, 웹·사람 영상 데이터 | FLUX.2 klein base 4B + LoRA r64, 뷰당 224×224 (3뷰 가로 타일), 메타데이터 조건 없음, B1K 데모만 |
| 생성 서브골 혼합 | "a large number" | 4,000장 (12 스텝) 으로 LL 5,000 스텝 추가 학습 |
| 런타임 | HL·WM 을 별도 스레드에서 비동기 | 동기 (하네스가 응답을 기다리는 동안 시뮬레이터가 멈춘다) |
| 추론 설정 | 5 디노이징, Ĥ ∈ {15, 25}, 4초마다 서브골, 메타 CFG β ∈ {1.3, 1.7, 2.2} | 5 디노이징, Ĥ = 25, 평가에서 WM·메타 CFG 미사용 |

## 9. 하드웨어 · 소프트웨어 · 자원

| 항목 | 값 |
|---|---|
| 학습·로컬 평가 머신 | GPU0 RTX 3090 Ti 24 GB, GPU1 RTX 3090 24 GB, RAM 48 GB, 스왑 64 GB |
| 원격 평가 머신 | RTX 3060 12 GB, RAM 31 GB, OmniGibson 3.9.2, isaacsim 5.1.0 |
| 로컬 평가 하네스 | OmniGibson 3.9.1, isaacsim 5.1.0.0 (Python 3.11) |
| 학습·서빙 환경 | Python 3.12, torch 2.11.0+cu128, transformers 5.5.4, lerobot 0.5.2, diffusers 0.40.0 (`--no-deps` 로 추가) |
| 실행 방식 | systemd 사용자 유닛(`launch.sh`): Restart=on-failure (120초 뒤, 2시간에 5회 상한), KillMode=mixed, TimeoutStopSec=600, MemoryMax 18G (200k 는 26G), MemorySwapMax 8G |
| 프로세스 환경변수 | OMP/MKL 스레드 4, 영상 디코더 스레드 1, 워커당 열린 영상 파일 6개 상한 |
| VRAM (학습, 2026-09-18 실측) | LL 16.9 GB, HL 13.1 GB. WM 스모크는 배치 12 에서 최대 16.1 GB |
| 스텝 시간 | HL 8.76 s, LL 30k 3.30 s, WM 9.17 s, genmix 3.24 s (각 마지막 스텝 로그). LL 200k 스모크 1.74 s (2 GPU) |
| 서빙 VRAM | HL + LL + WM 을 한 GPU 에 올렸을 때 최대 19.0 GB (24 GB 한 장, 챌린지 제출 조건 안) |
