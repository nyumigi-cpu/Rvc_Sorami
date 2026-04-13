# Rvc_Sorami - 실시간 음성변조 프로젝트

## 프로젝트 목표
세계 최고 수준의 실시간 음성변조 시스템 구축
- "진짜 여자와 구별 불가능" 수준의 남→여 음성 변환
- 모든 인간 소리 변환 (말, 웃음, 한숨, 속삭임, 감탄 등)
- 50ms 이하 지연, CPU 전용
- 게임뿐 아니라 모든 환경(통화, 조용한 방 등)에서 통할 수준
- 특정 사용자(고음/중성적 남성) → 특정 여성 1:1 전용 모델

## 작업 원칙

### 1. 솔직하고 현실적인 피드백
- 기분을 맞추기 위한 호의적 대답 금지
- 현실적이고 솔직한 의견을 제시할 것
- 기술적으로 문제가 있으면 직접적으로 지적할 것

### 2. 불가능은 없다 - 돌파구를 찾아라
- 하나의 접근법에 매몰되지 말 것
- 시야를 넓게 두고 다양한 방법론을 탐색할 것
- 막히면 우회로를 찾고, 다른 분야의 기술도 적극 차용할 것

---

## 사용자 컨텍스트

- **코딩 지식 없음** — 모든 코드를 클로드가 작성. 사용자는 클릭/피드백만
- **환경**: PC (Windows), 게임 개발도 병행 중 (PC 클코는 게임용)
- **학습 GPU**: Google Colab (무료 T4 또는 Pro A100)
- **데이터**: 목표 여성 목소리 녹음 보유 (대화+웃음+감정 포함)
- **사용자 목소리**: 고음/중성적 남성 (남→여 변환에 가장 유리한 조건)
- **이전 경험**: Beatrice Colab 학습 경험 있음. 오카다 기반 자체 VC 시도 → 실패 (삐이이이 소리)

## 이전 실패 교훈
오카다(w-okada) voice-changer 코드 위에 ONNX 모델을 끼워넣으려 시도 → 오디오 포맷/버퍼 불일치로 노이즈만 발생. 남의 코드 위에 올리는 접근은 실패함.
**이번에는 한 줄도 남의 코드를 쓰지 않고 바닥부터 자체 구축한다.**

---

## 핵심 설계: 뉴럴 오디오 코덱 기반 VC

### 왜 코덱 기반인가
기존 VC (RVC, Beatrice)는 "언어적 콘텐츠(발음/단어)"를 추출 → 다른 음색으로 재합성.
문제: 웃음, 한숨, 속삭임은 "언어적 콘텐츠"가 아니라서 깨지거나 무시됨.

**우리 접근**: 뉴럴 오디오 코덱은 원래 모든 소리를 압축/복원하도록 설계됨.
코덱에 음색 교체를 내장하면, 말이든 웃음이든 상관없이 음색만 바뀜.
비유: "편지 내용을 읽고 다시 쓰는 것"이 아니라 "사진 전체에서 색깔만 바꾸는 것".

### 추론 파이프라인

```
마이크 (48kHz, WASAPI 128샘플 = 2.67ms)
    ↓
[리샘플링] 48kHz → 24kHz
    ↓
[코덱 인코더] (경량 인과적 CNN, ~0.5M params)
  → 소리 전체를 토큰으로 압축
  → 음색 토큰과 콘텐츠/구조 토큰 분리 (DeCodec 직교 투영 방식)
    ↓
[음색 교체 모듈] (경량 프로젝션 네트워크, ~0.3M params)
  → 소스 남성 음색 → 타겟 여성 음색으로 교체
  → 나머지(콘텐츠, 운율, 비언어 구조)는 그대로 보존
    ↓
[F0 변환] (PESTO 기반, 130K params)
  → 피치를 남성 → 여성 범위로 자연스럽게 이동
    ↓
[코덱 디코더] (경량 인과적 CNN + iSTFT, ~1.5M params)
  → 변환된 토큰을 24kHz 파형으로 복원
    ↓
[FIR 포스트필터] (64 taps)
    ↓
[리샘플링] 24kHz → 48kHz
    ↓
스피커 출력 (2.67ms)
```

### 레이턴시 예산

| 구간 | 시간 |
|------|------|
| 오디오 입력 버퍼 | 2.67ms |
| 코덱 인코더 + 분리 | 3-5ms |
| 음색 교체 + F0 변환 | 2-3ms |
| 코덱 디코더 (iSTFT) | 5-8ms |
| FIR 필터 | <1ms |
| 오디오 출력 버퍼 | 2.67ms |
| 시스템 오버헤드 | 2-3ms |
| **합계** | **~20-25ms** |

### 모델 크기

| 모듈 | 파라미터 | INT8 ONNX |
|------|----------|-----------|
| 코덱 인코더 | ~0.5M | ~0.5MB |
| 음색 교체 네트워크 | ~0.3M | ~0.3MB |
| F0 추정 (PESTO) | 0.13M | ~0.15MB |
| 코덱 디코더 | ~1.5M | ~1.5MB |
| FIR 필터 | 64 | ~256B |
| **합계** | **~2.5M** | **~2.5MB** |

### Beatrice v2 대비 목표

| 항목 | Beatrice v2 | Sorami |
|------|-------------|--------|
| 레이턴시 | 50ms | **20-25ms** |
| 출력 | 16kHz | **24kHz** |
| 모델 크기 | ~30MB | **~2.5MB (INT8)** |
| CPU RTF | 0.2 | **<0.1** |
| 비언어 변환 | 불가능 | **웃음/한숨/속삭임 가능** |
| 콘텐츠 분리 | VQ | **직교 투영 (DeCodec)** |
| 보코더 | WaveGen | **코덱 디코더 (iSTFT)** |

---

## 연구 결과 요약 (핵심 참고 시스템)

### 아키텍처 핵심 참고

1. **VChangeCodec (ICLR 2025)** — 코덱에 VC 내장, 40ms, <1M params. Scalar Quantization으로 RVQ 대체. 코덱 기반이라 모든 소리 자연 처리.
2. **LLVC (Koe AI, arXiv 2311.00873)** — CPU 20ms 미만, RTF 2.8x. 지식 증류(대형 RVC→소형 Waveformer) + GAN. 마스크 기반 접근(파형 직접 생성 대신 입력에 마스크). Any-to-One.
3. **DeCodec (arXiv 2509.09201)** — 직교 투영(SOP)으로 음성을 {시맨틱, 파라언어적} 토큰으로 수학적 분리. 파라언어적 토큰의 음색만 교체 → 웃음 구조 보존.
4. **Takin-VC (ACL 2025)** — Adaptive Hybrid Content Encoder (WavLM + HybridFormer). 호흡/울음/감정 보존하는 최초의 VC. NMOS 3.98, SMOS 4.11.
5. **StreamVC (Google, ICASSP 2024)** — SoundStream 기반 인과적 합성곱. Pixel 7 CPU에서 70.8ms. 20ms 프레임 단위 스트리밍 설계.
6. **Beatrice v2** — wav2vec 2.0→VQ→WaveGenerator. 50ms, CPU 단일 스레드, 30MB. StreamVC 영감. 16kHz 한계.
7. **RVC** — HuBERT→FAISS→VITS→NSF-HiFi-GAN. 고음질이지만 90-300ms, GPU 필수. 오프라인 설계.
8. **StyleStream (UC Berkeley, arXiv 2602.20113)** — FSQ 45코드 정보병목 + DiT + Causal Vocos. 음색+억양+감정 동시 변환. ~1초 레이턴시. 16kHz.

### CPU 최적화 핵심

9. **MS-Wavehax** — 0.332M params 보코더, CPU RTF 0.04. HiFi-GAN의 2.4% 크기.
10. **Vocos** — iSTFT 기반 보코더. CPU 169.63x 실시간. ConvNeXt + iFFT 업샘플링.
11. **PESTO** — 130K params 피치 추정. 10ms 미만.
12. **SpecTokenizer (Interspeech 2025)** — DAC의 0.6% params로 동등 품질. CNN+RNN.

### 비언어 소리 관련

13. **voc2vec** — 비언어 전용 파운데이션 모델 (125시간, wav2vec 2.0 기반)
14. **NaturalVoices** — 5,049시간 팟캐스트 (웃음/기침/한숨/신음 포함)
15. **SynParaSpeech** — 118.75시간, 6 파라언어 카테고리, 타임스탬프
16. **SenseVoice-Small** — 70ms에 오디오 이벤트 감지 (웃음/숨소리/말 구분)

### GitHub 프로젝트 참고

17. **DDSP-SVC** (github.com/yxlllc/DDSP-SVC) — harmonic/noise 분해, 비언어 자연 통과, CPU 친화
18. **QuickVC** (github.com/quickvc/QuickVC-VoiceConversion) — iSTFT 디코더로 보코더 제거, 연산량 수배 절감
19. **seed-vc-rs** (github.com/thewh1teagle/seed-vc-rs) — Rust ONNX 추론 참고
20. **LLVC** (github.com/KoeAI/LLVC) — CPU 실시간 VC의 가장 실용적 레퍼런스
21. **cached_conv** (IRCAM) — 비인과 모델을 스트리밍 모델로 변환

---

## 기술 스택

- **학습**: Python + PyTorch 2.x (Google Colab)
- **추론**: Rust + ONNX Runtime (ort 크레이트)
- **오디오 I/O**: CPAL (Rust, WASAPI/CoreAudio/ALSA)
- **GUI**: Tauri (Rust + Svelte)
- **링 버퍼**: ringbuf 크레이트 (lock-free SPSC)
- **피치**: PESTO (130K params)
- **양자화**: ONNX Runtime INT8 동적 양자화

---

## 학습 3단계

### Phase 1: Teacher 데이터 생성
- Seed-VC 또는 Takin-VC를 Teacher로 사용 (비언어 보존 능력)
- 다양한 남성 음성 → 목표 여성 음성으로 변환한 합성 병렬 데이터 생성
- 웃음, 한숨, 속삭임도 포함

### Phase 2: 코덱 + VC 학습
- DeCodec 방식으로 음색/콘텐츠/파라언어 분리 학습
- VChangeCodec 방식으로 인코더에 음색 적응 네트워크 통합
- GAN 학습 (Multi-Period + Multi-Scale Discriminator)
- 손실: L_mel + L_adversarial + L_feat_match + L_ASR_CTC + L_D4C

### Phase 3: 경량화 & 스트리밍 최적화
- LLVC 방식 지식 증류 (Phase 2 모델 → 극경량 스트리밍 모델)
- ONNX INT8 양자화
- 인과적 상태 캐싱 적용

---

## 프로젝트 구조

```
Rvc_Sorami/
├── CLAUDE.md                      # 이 파일
├── training/                      # 학습 (Python, Colab용)
│   ├── models/
│   │   ├── __init__.py            # [완성] 모듈 export
│   │   ├── codec_encoder.py       # [완성] 경량 인과적 코덱 인코더
│   │   ├── codec_decoder.py       # [미완성] 인과적 코덱 디코더 + iSTFT
│   │   ├── timbre_adapter.py      # [미완성] 음색 교체 네트워크
│   │   ├── pitch_estimator.py     # [미완성] PESTO 경량 피치
│   │   ├── discriminator.py       # [미완성] GAN 판별기
│   │   └── pipeline.py            # [미완성] 전체 파이프라인
│   ├── losses/                    # [미완성] mel, adversarial, d4c, ctc
│   ├── trainers/                  # [미완성] phase1~3 학습 루프
│   ├── data/                      # [미완성] 데이터 처리 + 증강
│   ├── export/                    # [미완성] ONNX 내보내기 + 양자화
│   └── configs/                   # [미완성] YAML 설정
├── inference/                     # [미완성] Rust 추론 엔진
│   └── src/
│       ├── audio_io.rs            # CPAL 오디오 I/O
│       ├── ring_buffer.rs         # Lock-free 링 버퍼
│       ├── onnx_engine.rs         # ONNX Runtime 래퍼
│       ├── pipeline.rs            # 추론 파이프라인
│       └── state_cache.rs         # 인과적 상태 캐시
├── app/                           # [미완성] GUI (Tauri + Svelte)
├── tools/                         # [미완성] 벤치마크, 학습 GUI (Gradio)
├── tests/                         # [미완성] 테스트
└── docs/                          # [미완성] 문서
```

---

## 개발 로드맵 (STEP별)

### STEP 1: 코드 작성 (사용자 할 일 없음)
- [ ] codec_decoder.py — 인과적 CNN + iSTFT 헤드 (Vocos 방식)
- [ ] timbre_adapter.py — AdaIN 기반 음색 교체 (화자 임베딩 → scale/shift)
- [ ] pitch_estimator.py — PESTO 경량 피치 추정
- [ ] discriminator.py — Multi-Period + Multi-Scale Discriminator
- [ ] pipeline.py — 전체 모듈 결합 + forward/loss 계산
- [ ] losses/ — mel_loss, adversarial_loss, d4c_loss, ctc_loss
- [ ] trainers/ — Phase 1 (증류), Phase 2 (GAN), Phase 3 (경량화)
- [ ] data/ — 데이터셋 클래스, 전처리, 증강
- [ ] configs/ — 하이퍼파라미터 YAML
- [ ] export/ — ONNX 내보내기 + INT8 양자화
- [ ] tools/ — Colab 노트북, 벤치마크, 오디오 테스트

### STEP 2: 데이터 준비 (사용자: 녹음 파일 업로드)
- 사용자의 여자 녹음을 구글 드라이브에 올림
- 형식/길이/품질 가이드를 제공할 것

### STEP 3: AI 학습 (사용자: Colab에서 버튼 클릭)
- Colab 노트북 링크 제공
- 드라이브 연결 → ▶ 실행 → 모델 파일 다운로드

### STEP 4: 첫 테스트 (사용자: 듣고 피드백)
- Python 테스트 스크립트 또는 간이 GUI 제공

### STEP 5: 튜닝 반복 (사용자: 듣고 피드백)
- 피드백 기반으로 데이터/모델/파라미터 조정

### STEP 6: Rust 추론 엔진 + GUI 앱 완성

---

## 위험 요소 & 대안

1. **코덱 VC 품질 부족** → LLVC 마스크 기반으로 전환 (검증된 20ms CPU)
2. **비언어 변환 품질** → 하이브리드 라우팅 (SenseVoice 감지 → 음성/비음성 분기)
3. **Colab 학습 한계** → 2.5M 모델로 T4에서도 학습 가능
4. **ONNX 호환성** → 고정 프레임 크기 480 samples, 비호환 연산자 회피
5. **24kHz 품질 문제** → 16kHz 버전 먼저 검증 후 확장

---

## codec_encoder.py 설계 메모 (이미 구현된 파일)

- CausalConv1d: 왼쪽(과거)에만 패딩하는 인과적 1D 합성곱
- DepthwiseSeparableConv: LLVC에서 사용, 연산량 ~60% 절감
- EncoderBlock: Dilated Causal Conv + DSC, dilation_base=2로 수용 범위 확장
- CodecEncoder: strides=(4,4,5,3) → 총 다운샘플 240배 (24kHz→100fps)
  - content_bottleneck: 256→64 차원으로 병목 (음색 정보 제거)
  - timbre_proj: 시간축 평균 풀링으로 전역 화자 벡터 추출

---

## 품질 향상 로드맵 (Level 시스템)

- **Level 1 (60-70%)**: 기본 코덱 VC — 말소리가 여자로 나옴
- **Level 2 (80%)**: 약점 제거 — 웃음/숨소리/감탄사 각각 집중 개선
- **Level 3 (90%)**: 미세 디테일 — 입술소리, 성대프라이, 기식성, 말 끝 처리
- **Level 4 (95%+)**: 개인 최적화 — 사용자 전용 1:1 매핑, 약점 집중 훈련
- **Level 5 (98-99%)**: 인간 피드백 루프 — "여기 이상해" 반복 수정

핵심 전략: 범용 70% → "너 전용" 95% → 반복 튜닝 99%
