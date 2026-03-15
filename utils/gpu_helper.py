import os
import sys
import site
from utils.logger import setup_logger

logger = setup_logger("gpu_helper")

def setup_cuda_path():
    """
    Windows 환경에서 ctranslate2(faster-whisper)가 cuDNN, cuBLAS DLL을 찾을 수 있도록 
    파이썬 패키지 내의 NVIDIA 라이브러리 경로를 시스템 PATH에 주입합니다.
    """
    if sys.platform != "win32":
        return

    # 파이썬 패키지 설치 경로(site-packages) 목록 확인
    site_packages = site.getsitepackages()
    
    # nvidia 관련 DLL이 포함된 bin 디렉토리 경로 패턴들
    search_paths = [
        "nvidia/cublas/bin",
        "nvidia/cudnn/bin",
        "nvidia/cuda_runtime/bin",
        "nvidia/cuda_nvrtc/bin"
    ]

    added_paths = []
    for sp in site_packages:
        for pattern in search_paths:
            full_path = os.path.join(sp, pattern)
            if os.path.exists(full_path):
                # PATH 환경변수 최상단에 추가
                if full_path not in os.environ["PATH"]:
                    os.environ["PATH"] = full_path + os.pathsep + os.environ["PATH"]
                    added_paths.append(full_path)

    if added_paths:
        logger.info(f"CUDA 관련 DLL 경로 주입 완료: {added_paths}")
    else:
        logger.warning("NVIDIA 라이브러리 경로를 자동으로 찾을 수 없습니다. (CPU 모드 사용 권장)")

def is_cuda_available() -> bool:
    """
    현재 시스템에서 CUDA를 사용할 수 있는지 확인합니다.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
