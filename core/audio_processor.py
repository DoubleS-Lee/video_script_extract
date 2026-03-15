import os
import subprocess
from utils.logger import setup_logger

logger = setup_logger("audio_processor")

def extract_audio(video_path: str, output_path: str = None) -> str:
    """
    영상 파일에서 STT 모델이 인식하기 가장 좋은 포맷(16kHz, Mono, 16-bit WAV)으로 오디오를 추출합니다.
    """
    if output_path is None:
        # 영상 파일명과 같은 이름으로 .wav 생성
        base, _ = os.path.splitext(video_path)
        output_path = f"{base}_audio.wav"

    # FFmpeg 명령어 구성: 16kHz, 1채널(Mono), pcm_s16le(16-bit PCM)
    command = [
        'ffmpeg', '-y', # -y: 덮어쓰기 허용
        '-i', video_path,
        '-vn',          # -vn: 비디오 제외
        '-ac', '1',     # -ac 1: 모노(1채널)
        '-ar', '16000', # -ar 16000: 16kHz 샘플링 레이트
        '-acodec', 'pcm_s16le',
        output_path
    ]

    try:
        logger.info(f"오디오 추출 시작: {video_path}")
        # subprocess.run을 사용하여 FFmpeg 실행
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"오디오 추출 성공: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg 실행 중 오류 발생: {e.stderr.decode()}")
        raise RuntimeError(f"FFmpeg 오디오 추출 실패: {e.stderr.decode()}")
    except FileNotFoundError:
        logger.error("FFmpeg를 찾을 수 없습니다. 시스템에 FFmpeg가 설치되어 있는지 확인하세요.")
        raise FileNotFoundError("시스템에서 FFmpeg를 찾을 수 없습니다.")
