import os
import sys
import types

# --- [최상단] 모든 라이브러리 임포트 전 패치 시작 ---

# 1. 환경 변수 설정
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 2. torchaudio 속성 강제 주입 (AttributeError 방지)
try:
    import torchaudio
    
    # torchaudio 객체에 직접 누락된 함수들을 정의
    # (이미 존재하는 경우 덮어쓰지 않음)
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda x: None
        
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"

    # 가상 backend 모듈 구조 생성
    if "torchaudio.backend" not in sys.modules:
        mock_backend = types.ModuleType("torchaudio.backend")
        sys.modules["torchaudio.backend"] = mock_backend
        torchaudio.backend = mock_backend
        
    if "torchaudio.backend.common" not in sys.modules:
        mock_common = types.ModuleType("torchaudio.backend.common")
        try:
            from torchaudio import AudioMetaData
            mock_common.AudioMetaData = AudioMetaData
        except ImportError:
            class MockAudioMetaData:
                def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample=16, encoding="PCM_S"):
                    self.sample_rate = sample_rate
                    self.num_frames = num_frames
                    self.num_channels = num_channels
                    self.bits_per_sample = bits_per_sample
                    self.encoding = encoding
            mock_common.AudioMetaData = MockAudioMetaData
        sys.modules["torchaudio.backend.common"] = mock_common
        torchaudio.backend.common = mock_common

    # torchaudio.load 및 info 패치 (soundfile 우회)
    import torch
    import soundfile as sf

    def patched_info(uri, format=None, buffer_size=4096, backend=None):
        info = sf.info(uri)
        # 위에서 정의한 AudioMetaData 클래스 참조
        meta_cls = getattr(torchaudio.backend.common, "AudioMetaData")
        return meta_cls(info.samplerate, info.frames, info.channels)

    def patched_load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None, buffer_size=4096, backend=None):
        data, sr = sf.read(uri, always_2d=True)
        tensor = torch.from_numpy(data).t()
        if frame_offset > 0 or num_frames > 0:
            start = frame_offset
            end = frame_offset + num_frames if num_frames > 0 else None
            tensor = tensor[:, start:end]
        return tensor.float(), sr

    torchaudio.info = patched_info
    torchaudio.load = patched_load

except ImportError:
    pass

# 3. torch.load 및 기타 패치
try:
    import torch
    original_torch_load = torch.load
    def patched_torch_load(f, *args, **kwargs):
        if 'weights_only' in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(f, *args, **kwargs)
    torch.load = patched_torch_load
except ImportError:
    pass

try:
    import huggingface_hub.file_download as hf_download
    original_hf_hub_download = hf_download.hf_hub_download
    def patched_hf_hub_download(*args, **kwargs):
        if 'use_auth_token' in kwargs:
            kwargs['token'] = kwargs.pop('use_auth_token')
        return original_hf_hub_download(*args, **kwargs)
    hf_download.hf_hub_download = patched_hf_hub_download
except: pass

try:
    import numpy as np
    if not hasattr(np, "NaN"):
        np.NaN = np.nan
    if not hasattr(np, "NAN"):
        np.NAN = np.nan
except: pass

# --- [패치 완료] 앱 실행 ---
from ui.app_window import App
from utils.logger import setup_logger
logger = setup_logger("main")

def main():
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        logger.error(f"치명적 오류: {e}")
        import traceback
        traceback.print_exc()
        input("\n엔터 키를 눌러 종료하세요...")

if __name__ == "__main__":
    main()
