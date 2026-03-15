# STT 및 화자 분리(Speaker Diarization) 구현 리뷰

본 문서에서는 `capcut_agents_260309` 프로젝트 내에서 영상의 대화 내용을 화자별로 구분해주는 STT(Speech-to-Text) 프로그램을 구현할 때 사용한 방법, 기술(스킬), 그리고 겪었던 문제점 및 해결 과정을 정리합니다.

## 1. 구현 방법 및 작동 원리

현재 시스템은 영상 파일에서 오디오를 추출한 뒤, 음성 인식과 화자 분리를 각각 독립적인 모델로 병렬 처리하고 최종적으로 두 결과를 병합하는 방식으로 구현되어 있습니다.

1. **오디오 추출**: 원본 영상(`mp4`, `mov`)에서 `moviepy`를 사용하여 오디오 트랙만 임시 폴더에 `wav` (pcm_s16le 포맷) 파일로 추출합니다.
2. **텍스트 변환 (STT)**: 추출된 오디오를 `faster-whisper` 모델에 입력하여 자막(텍스트)과 시간 정보(시작/종료 시간)를 추출합니다.
3. **화자 분리 (Diarization)**: 동일한 오디오 파일을 `pyannote.audio` 파이프라인에 입력하여 누가 언제 말했는지에 대한 타임스탬프 기반 화자 분리 구간 데이터를 추출합니다.
4. **결과 병합**: Whisper에서 추출된 텍스트 구간(Segment)과 Pyannote에서 추출된 화자 구간(Turn)을 교차 비교하여, 텍스트 구간과 가장 많이 겹치는(Intersection) 시간을 가진 화자(Speaker)를 해당 텍스트의 화자로 매핑(Mapping)합니다.

## 2. 사용된 스킬 및 라이브러리

*   **faster-whisper**: `ctranslate2` 기반으로 최적화된 Whisper 모델입니다. `base` 모델을 사용하며, `float16` 연산과 CUDA (GPU 가속)를 통해 빠른 처리 속도를 확보했습니다. (`beam_size=5` 적용)
*   **pyannote.audio**: Hugging Face의 최신 화자 분리 파이프라인(`pyannote/speaker-diarization-3.1`)을 사용하여 고도화된 화자 클러스터링을 수행합니다.
*   **moviepy**: 복잡한 영상 처리 도구 없이 파이썬 네이티브 환경에서 빠르게 영상 내 오디오만 파싱하고 파일로 저장하기 위해 사용했습니다.
*   **CrewAI (BaseTool)**: 에이전트 프레임워크와의 연동을 위해 `STTTool` 클래스를 `BaseTool`로 래핑하여, LLM 기반 에이전트가 필요할 때 도구로써 호출할 수 있도록 모듈화하였습니다.

## 3. 겪었던 어려움과 해결책

STT와 화자 분리를 통합하는 과정에서, 특히 윈도우(Windows) 및 C++ 기반 라이브러리(CUDA, ctranslate2 등)가 혼합되면서 여러 가지 환경적, 런타임 오류가 발생했습니다.

### 문제점 1: OpenMP 라이브러리 충돌 및 강제 종료
*   **어려움**: 여러 파이썬 패키지(PyTorch, ctranslate2 등)가 각자의 OpenMP 런타임을 로드하려다 중복 로드 에러가 발생하여 파이썬 프로세스 자체가 조용히 강제 종료되는 현상이 있었습니다.
*   **해결 방법**: 스크립트 최상단에 `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` 환경변수를 설정하여 중복 로드를 허용하도록 조치했습니다.

### 문제점 2: NVIDIA cuBLAS 및 cuDNN DLL 로드 실패 (ctranslate2 런타임 충돌)
*   **어려움**: `faster-whisper`가 내부적으로 사용하는 `ctranslate2` 엔진이 윈도우 환경에서 cuBLAS, cuDNN 라이브러리를 찾지 못해 초기화에 실패하는 문제가 있었습니다.
*   **해결 방법**: `site` 모듈을 이용해 파이썬 패키지가 설치된 경로(`site-packages`)를 동적으로 탐색하고, `nvidia` 폴더 내부의 `cublas`, `cudnn`의 `bin` 디렉터리 절대 경로를 찾아 시스템의 `PATH` 환경변수 최상단에 강제로 주입했습니다.

### 문제점 3: 파이썬 가비지 컬렉터(GC)에 의한 C++ 소멸자(Destructor) 크래시
*   **어려움**: `_run` 함수가 종료되면서 함수 내부 지역변수로 선언된 Whisper 모델이나 Pyannote 파이프라인 인스턴스가 메모리에서 해제될 때, C++ 레벨의 소멸자가 호출되면서 메모리 접근 오류(Segmentation Fault)를 일으키며 프로그램이 튕기는 현상이 있었습니다.
*   **해결 방법**: 모델 인스턴스들을 전역 변수(`_whisper_model_cache`, `_diarization_pipeline_cache`)에 다루어 프로세스가 살아있는 동안 메모리에 계속 유지되도록 하여 생명주기(Lifecycle) 충돌을 방지했습니다. 이는 재호출 시 모델 로딩 시간을 없애주는 성능 향상 효과도 가져왔습니다.

### 문제점 4: Hugging Face 인증 및 의존성 문제
*   **어려움**: `pyannote/speaker-diarization-3.1` 모델은 Hugging Face에서 사용자 약관 동의 및 인증 토큰이 필수적으로 요구됩니다. 토큰이 없거나 네트워크 환경 등에 의해 다운로드에 실패하면 전체 STT 프로세스가 중단되는 문제가 있었습니다.
*   **해결 방법**: 환경변수 `HUGGINGFACE_TOKEN`을 통해 토큰을 주입받도록 구성하고, `try-except` 블록으로 로딩 실패를 감지했습니다. 실패할 경우 화자 분리 기능만 비활성화하고, Whisper 단독으로 STT(기본 화자로 처리)만 진행하도록 Fallback 메커니즘을 구현하여 전체 파이프라인의 안정성을 높였습니다.

### 문제점 5: 출력 버퍼링과 인코딩(UnicodeEncodeError) 문제
*   **어려움**: 화자 이름이나 자막에 이모지가 포함되거나, 프로세스 크래시가 발생할 때 파이썬 에러 로그가 버퍼링에 막혀 콘솔에 제대로 출력되지 않고 죽어버려 원인 파악 및 디버깅이 어려운 문제가 있었습니다.
*   **해결 방법**: `sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)`를 적용하여 이모지 출력 시의 인코딩 에러를 방지하고 출력 버퍼 없이 즉시 콘솔에 텍스트가 표시되게 했으며, 추가로 `faulthandler.enable()`을 활성화하여 C++ 단의 크래시가 났을 때도 파이썬의 콜스택을 잡아내도록 구성했습니다.

### 문제점 6: PyTorch 분산/통계 관련 경고 메시지 스팸
*   **어려움**: 화자 분리 파이프라인 실행 중 특정 오디오 구간에서 PyTorch 내부적으로 분산(variance)을 계산할 때 자유도(degrees of freedom)가 0 이하가 되어 발생하는 경고 메시지(`std(): degrees of freedom is <= 0`)가 콘솔을 도배하는 현상이 있었습니다.
*   **해결 방법**: `warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")`를 선언하여 해당 특정 정규식 패턴을 가진 경고(UserWarning)만 무시하도록 처리하여 작업 로그를 깔끔하게 유지했습니다.