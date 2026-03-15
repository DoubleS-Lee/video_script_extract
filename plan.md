# 영상 대화 내용 JSON 대사집 변환 프로그램 구현 계획 (Implementation Plan)

## 1. 프로젝트 개요
`research.md`를 바탕으로, 영상 파일에서 오디오를 추출하고, 화자가 구분된 대사집(JSON)을 생성하며, 이를 GUI와 함께 단일 실행 파일(.exe)로 배포하기 위한 구체적인 기술적 구현 계획입니다.

## 2. 기술 스택 (Tech Stack)
*   **언어:** Python 3.10+ (호환성 및 PyTorch 안정성 고려)
*   **핵심 AI 라이브러리:**
    *   `faster-whisper`: STT (음성 -> 텍스트 변환)
    *   `pyannote.audio`: Diarization (화자 분리)
    *   `torch`, `torchaudio`: 딥러닝 백엔드
*   **미디어 처리:**
    *   `ffmpeg-python`: 영상에서 오디오 추출 (16kHz, mono, wav 변환)
*   **GUI 프레임워크:**
    *   `customtkinter`: 현대적인 UI 구현 (드래그 앤 드롭, CPU/GPU 토글 스위치 등)
*   **패키징 및 배포:**
    *   `PyInstaller`: 단일 `.exe` 파일 빌드 및 종속성 번들링

## 3. 파일 및 디렉토리 구조 (Directory Structure)
```text
stt_portable_project/
│
├── main.py                 # 프로그램 진입점 및 GUI 구동
├── config.py               # 모델 경로, 기본 설정, 환경 변수 관리
│
├── core/                   # 핵심 비즈니스 로직
│   ├── audio_processor.py  # 영상 오디오 추출 및 전처리 (FFmpeg)
│   ├── stt_engine.py       # faster-whisper 구동 (텍스트/타임스탬프 추출)
│   ├── diarization.py      # pyannote 구동 (화자 구간 식별)
│   └── script_builder.py   # STT와 Diarization 결과 병합 및 JSON 생성
│
├── ui/                     # 사용자 인터페이스
│   └── app_window.py       # CustomTkinter 기반 GUI 클래스
│
├── utils/                  # 유틸리티 함수
│   ├── gpu_helper.py       # CUDA 가용성 체크 및 DLL 경로 강제 주입
│   └── logger.py           # 작업 진행 상황 및 에러 로깅
│
├── requirements.txt        # 라이브러리 의존성 목록
└── build.spec              # PyInstaller 빌드 상세 설정
```

## 4. 핵심 모듈별 구현 계획

### Phase 1: 오디오 전처리 (`audio_processor.py`)
*   `ffmpeg`를 호출하여 원본 영상에서 목소리 인식에 최적인 **16kHz, Mono, 16-bit WAV** 파일을 추출합니다.
*   임시 작업 폴더를 생성하고 작업 완료 후 자동으로 정리하는 로직을 포함합니다.

### Phase 2: AI 엔진 구성 (`stt_engine.py`, `diarization.py`)
*   **Device 선택 로직:** 사용자가 GUI에서 선택한 값에 따라 `device="cuda"` 또는 `device="cpu"`를 동적으로 결정합니다.
*   **Whisper 최적화:** `compute_type="float16"`(GPU 시) 또는 `"int8"`(CPU 시)을 적용하여 속도와 메모리를 최적화합니다.
*   **HuggingFace 인증:** Diarization 모델 로드 시 필요한 토큰 정보를 안전하게 관리합니다.

### Phase 3: 결과 병합 및 저장 (`script_builder.py`)
*   텍스트의 시작/종료 시간과 화자별 발화 시간 구간을 대조하여, 겹치는 시간이 가장 긴 화자를 해당 대사의 주인공으로 매핑합니다.
*   최종 결과물은 `research.md`에서 정의한 표준 JSON 포맷으로 출력합니다.

### Phase 4: GUI 및 비동기 처리 (`ui/app_window.py`)
*   **사용자 경험:** 드래그 앤 드롭으로 파일을 입력받고, 실시간 프로그레스바를 통해 진행 상황을 공유합니다.
*   **스레딩:** AI 연산 중 GUI가 멈추지 않도록 `threading` 또는 `concurrent.futures`를 사용하여 백그라운드에서 작업을 수행합니다.

## 5. 구현 시 주의점 및 해결책

### 1) 윈도우 CUDA DLL 오류 해결
*   `stt.md`에서의 경험을 바탕으로, 프로그램 시작 시 `nvidia` 관련 폴더의 `bin` 경로를 탐색하여 `os.environ["PATH"]`에 강제로 추가하는 코드를 `gpu_helper.py`에 작성합니다.

### 2) 실행 파일(.exe) 용량 및 호환성
*   PyInstaller 빌드 시 `faster-whisper` 모델 파일을 내장할지, 혹은 첫 실행 시 다운로드할지 결정합니다. (내장 시 약 1.5GB~2GB 예상)
*   `hidden-import` 설정을 통해 AI 라이브러리들이 누락되지 않도록 `build.spec`을 꼼꼼히 작성합니다.

### 3) 메모리 관리 (C++ Crash 방지)
*   모델 객체를 함수 내 지역 변수로 두지 않고, 클래스 멤버나 전역 캐시로 관리하여 가비지 컬렉션 시 발생하는 C++ 레벨의 충돌을 방지합니다.

### 4) OpenMP 충돌 방지
*   `main.py` 최상단에 `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` 설정을 추가하여 멀티 스레딩 충돌을 예방합니다.
