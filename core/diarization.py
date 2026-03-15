import os
import torch
from pyannote.audio import Pipeline
from huggingface_hub import login # 인증을 위한 직접 로그인 도구
from utils.logger import setup_logger
from utils.gpu_helper import is_cuda_available

logger = setup_logger("diarization")

class DiarizationEngine:
    def __init__(self, hf_token: str, device="auto"):
        """
        pyannote.audio 화자 분리 파이프라인을 초기화합니다.
        hf_token: Hugging Face API 토큰
        """
        self.hf_token = hf_token
        if device == "auto":
            self.device = torch.device("cuda" if is_cuda_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"화자 분리 엔진 초기화 중: 장치={self.device}")
        
        try:
            # 1. Hugging Face 라이브러리에 직접 로그인 (토큰 매개변수 충돌 방지)
            login(token=self.hf_token)
            
            # 2. 파이프라인 로드 (인증 정보는 위에서 처리했으므로 생략하거나 True 전달)
            # 최신 버전 호환성을 위해 매개변수 없이 시도
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )
            
            # 모델을 선택한 장치(GPU/CPU)로 이동
            if self.pipeline:
                self.pipeline.to(self.device)
            else:
                raise ValueError("파이프라인 로드 실패 (권한 확인 필요)")
        except Exception as e:
            logger.error(f"화자 분리 모델 로딩 실패: {str(e)}")
            # 만약 위 방식이 실패하면, 환경변수 주입 후 재시도
            try:
                os.environ["HF_TOKEN"] = self.hf_token
                self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
                if self.pipeline: self.pipeline.to(self.device)
            except:
                raise e

    def diarize(self, audio_path: str):
        """
        오디오 파일을 분석하여 화자별 타임스탬프 구간을 반환합니다.
        """
        logger.info(f"화자 분리 분석 시작: {audio_path}")
        
        # Diarization 수행
        diarization = self.pipeline(audio_path)
        
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker
            })
            
        logger.info(f"화자 분리 완료: 총 {len(speaker_segments)}개의 발화 구간 발견")
        return speaker_segments
