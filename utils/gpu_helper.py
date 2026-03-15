import os
import sys
import site
import torch
from utils.logger import setup_logger

logger = setup_logger("gpu_helper")

def setup_cuda_path():
    """Windows 환경에서 CUDA DLL 경로를 시스템 PATH에 주입합니다."""
    if sys.platform != "win32":
        return

    site_packages = site.getsitepackages()
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
                if full_path not in os.environ["PATH"]:
                    os.environ["PATH"] = full_path + os.pathsep + os.environ["PATH"]
                    added_paths.append(full_path)

    if added_paths:
        logger.info(f"CUDA DLL 경로 주입 완료: {added_paths}")

def get_available_device(requested_device="auto") -> str:
    """
    사용자가 요청한 장치를 확인하고, 불가능할 경우 최선의 대안(CPU)을 반환합니다.
    """
    cuda_available = torch.cuda.is_available()
    
    if requested_device == "cuda":
        if cuda_available:
            return "cuda"
        else:
            logger.warning("GPU(CUDA)가 요청되었지만, 현재 환경에서 사용할 수 없습니다. CPU로 전환합니다.")
            return "cpu"
    
    if requested_device == "cpu":
        return "cpu"
        
    # "auto"인 경우
    return "cuda" if cuda_available else "cpu"

def is_cuda_available() -> bool:
    return torch.cuda.is_available()
