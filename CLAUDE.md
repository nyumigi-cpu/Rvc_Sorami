# Rvc_Sorami - 실시간 음성변조 프로젝트

## 프로젝트 목표
세계 최고 수준의 실시간 음성변조 시스템 구축
- "진짜 여자와 구별 불가능" 수준의 남→여 음성 변환
- 모든 인간 소리 변환 (말, 웃음, 한숨, 속삭임, 감탄 등)
- 50ms 이하 지연, CPU 전용, 게임/방송 보이스챗

## 작업 원칙

### 1. 솔직하고 현실적인 피드백
- 기분을 맞추기 위한 호의적 대답 금지
- 현실적이고 솔직한 의견을 제시할 것
- 기술적으로 문제가 있으면 직접적으로 지적할 것

### 2. 불가능은 없다 - 돌파구를 찾아라
- 하나의 접근법에 매몰되지 말 것
- 시야를 넓게 두고 다양한 방법론을 탐색할 것
- 막히면 우회로를 찾고, 다른 분야의 기술도 적극 차용할 것

## 핵심 설계 방향
- **뉴럴 오디오 코덱 기반 VC**: 코덱에 보이스체인저를 내장하여 모든 소리를 자연 변환
- **참고 시스템**: VChangeCodec(40ms/<1M), LLVC(20ms/CPU), DeCodec(직교 분리), Beatrice v2
- **학습**: PyTorch → ONNX INT8, Google Colab (T4/A100)
- **추론**: Rust + ONNX Runtime + CPAL
- **GUI**: Tauri + Svelte

## 기술 스택
- 학습: Python + PyTorch 2.x
- 추론: Rust + ONNX Runtime (ort)
- 오디오: CPAL (WASAPI/CoreAudio/ALSA)
- GUI: Tauri + Svelte
- 피치: PESTO (130K params)
