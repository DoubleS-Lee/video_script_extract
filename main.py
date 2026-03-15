import os
import sys
import types
import warnings

# --- [Step 1] 전역 환경 설정 ---
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# --- [Step 2] NUMPY 패치 (가장 먼저 실행되어야 함) ---
try:
    import numpy as np
    # NumPy 2.0에서 삭제된 속성들을 수동으로 복구
    np.NaN = np.nan
    np.NAN = np.nan
    if not hasattr(np, "float"): np.float = float
    if not hasattr(np, "int"): np.int = int
except Exception:
    pass

# --- [Step 3] TORCH & TORCHAUDIO 패치 ---
try:
    import torch
    import torchaudio
    import soundfile as sf

    # 3.1 torch.load 보안 체크 강제 해제
    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        if 'weights_only' in kwargs:
            kwargs['weights_only'] = False
        else:
            args_list = list(args)
            if len(args_list) >= 4:
                args_list[3] = False
                return original_torch_load(*tuple(args_list), **kwargs)
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load

    # 3.2 AudioMetaData 클래스 및 info/load 패치 (soundfile 우회)
    try:
        from torchaudio import AudioMetaData
    except ImportError:
        try: from torchaudio.backend.common import AudioMetaData
        except ImportError:
            class AudioMetaData:
                def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample=16, encoding="PCM_S16"):
                    self.sample_rate, self.num_frames, self.num_channels = sample_rate, num_frames, num_channels
                    self.bits_per_sample, self.encoding = bits_per_sample, encoding

    def patched_info(uri, **kwargs):
        info = sf.info(uri)
        return AudioMetaData(info.samplerate, info.frames, info.channels, 16, "PCM_S")

    def patched_load(uri, frame_offset=0, num_frames=-1, **kwargs):
        data, sr = sf.read(uri, always_2d=True)
        tensor = torch.from_numpy(data).t()
        if frame_offset > 0 or num_frames > 0:
            start = frame_offset
            end = frame_offset + (num_frames if num_frames > 0 else 0)
            tensor = tensor[:, start:end] if num_frames > 0 else tensor[:, start:]
        return tensor.float(), sr

    torchaudio.info, torchaudio.load = patched_info, patched_load

    # 3.3 torchaudio 구형 속성 복구
    if not hasattr(torchaudio, "list_audio_backends"): torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "set_audio_backend"): torchaudio.set_audio_backend = lambda x: None
    if not hasattr(torchaudio, "get_audio_backend"): torchaudio.get_audio_backend = lambda: "soundfile"

    # 3.4 가상 모듈 경로 생성 (pyannote 호환용)
    if "torchaudio.backend" not in sys.modules:
        sys.modules["torchaudio.backend"] = types.ModuleType("torchaudio.backend")
    if "torchaudio.backend.common" not in sys.modules:
        mock_common = types.ModuleType("torchaudio.backend.common")
        mock_common.AudioMetaData = AudioMetaData
        sys.modules["torchaudio.backend.common"] = mock_common

except ImportError:
    pass

# --- [Step 4] HUGGING FACE 패치 ---
try:
    import huggingface_hub.file_download as hf_download
    original_hf_download = hf_download.hf_hub_download
    def patched_hf_hub_download(*args, **kwargs):
        if 'use_auth_token' in kwargs: kwargs['token'] = kwargs.pop('use_auth_token')
        return original_hf_download(*args, **kwargs)
    hf_download.hf_hub_download = patched_hf_hub_download
except Exception:
    pass

# --- [Step 5] 경고 메시지 무시 ---
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")
warnings.filterwarnings("ignore", category=UserWarning)

# --- [Step 6] 드디어 메인 앱 및 나머지 모듈 로드 ---
from ui.app_window import App
from utils.logger import setup_logger
logger = setup_logger("main")

def main():
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        if "logger" in globals():
            logger.error(f"치명적 오류 발생: {e}")
        else:
            print(f"치명적 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        input("\n엔터 키를 눌러 종료하세요...")

if __name__ == "__main__":
    main()
