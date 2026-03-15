import logging
import sys

def setup_logger(name: str) -> logging.Logger:
    """
    작업 진행 상황을 추적하기 위해 기본 로거를 설정합니다.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # 윈도우 한글 인코딩 문제를 방지하기 위해 utf-8 설정
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.propagate = False
    
    return logger
