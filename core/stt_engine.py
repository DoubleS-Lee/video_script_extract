import os
from faster_whisper import WhisperModel
from utils.logger import setup_logger
from utils.gpu_helper import is_cuda_available

logger = setup_logger("stt_engine")

class STTEngine:
    def __init__(self, model_size="base", device="auto", compute_type="default"):
        """
        faster-whisper 모델을 초기화합니다.
        device: "cuda", "cpu", 또는 "auto" (자동 감지)
        """
        if device == "auto":
            self.device = "cuda" if is_cuda_available() else "cpu"
        else:
            self.device = device
            
        # GPU 사용 시 float16, CPU 사용 시 int8 권장
        if compute_type == "default":
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type
            
        logger.info(f"STT 엔진 초기화 중: 모델={model_size}, 장치={self.device}, 연산={self.compute_type}")
        
        # 모델 로드 (단일 실행 파일 배포를 위해 모델 경로 제어 가능)
        self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)

    def transcribe(self, audio_path: str):
        """
        오디오 파일을 텍스트로 변환합니다. 타임스탬프와 문장 정보를 반환합니다.
        """
        logger.info(f"음성 인식 시작: {audio_path}")
        
        # word_timestamps=True 옵션을 통해 단어 단위 정밀 타임스탬프 획득 가능
        segments, info = self.model.transcribe(audio_path, beam_size=5, word_timestamps=True)
        
        result_segments = []
        for segment in segments:
            # 대본 제작을 위해 필요한 정보만 추출
            result_segments.append({
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip(),
                "words": [
                    {"word": w.word.strip(), "start": w.start, "end": w.end, "probability": w.probability} 
                    for w in (segment.words or [])
                ]
            })
            
        logger.info(f"음성 인식 완료: 총 {len(result_segments)}개의 구간 발견")
        return result_segments, info
