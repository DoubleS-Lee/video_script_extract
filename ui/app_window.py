import os
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from config import load_config, save_config # 설정 관리 함수 가져오기

# tkinterdnd2 버전별 호환성을 위한 안전한 임포트
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES, DnDWrapper
except ImportError:
    try:
        from tkinterdnd2.TkinterDnD import TkinterDnD, DND_FILES, DnDWrapper
    except ImportError:
        TkinterDnD = None
        DND_FILES = "DND_FILES"
        DnDWrapper = object

from core.audio_processor import extract_audio
from core.stt_engine import STTEngine
from core.diarization import DiarizationEngine
from core.script_builder import merge_stt_diarization, save_as_json
from utils.gpu_helper import is_cuda_available, setup_cuda_path
from utils.logger import setup_logger

logger = setup_logger("gui")

class App(ctk.CTk, DnDWrapper):
    def __init__(self):
        super().__init__()
        
        # TkinterDnD 초기화
        self.dnd_available = False
        if TkinterDnD is not None:
            try:
                self.TkdndVersion = TkinterDnD.Tk(self)
                self.dnd_available = True
            except Exception as e:
                logger.warning(f"DnD 초기화 중 오류: {e}")

        # 기본 설정
        self.title("AI Video Script Pro")
        self.geometry("900x550")
        ctk.set_appearance_mode("dark")
        setup_cuda_path()

        # 저장된 설정 불러오기
        self.config_data = load_config()

        # 레이아웃 구성
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="AI Script Pro", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # HF Token
        self.hf_label = ctk.CTkLabel(self.sidebar_frame, text="Hugging Face Token:", anchor="w")
        self.hf_label.grid(row=1, column=0, padx=20, pady=(20, 0))
        self.hf_token_entry = ctk.CTkEntry(self.sidebar_frame, placeholder_text="Enter HF Token...", show="*", width=210)
        self.hf_token_entry.grid(row=2, column=0, padx=20, pady=(5, 10))
        
        # 저장된 토큰이 있으면 입력창에 미리 넣기
        saved_token = self.config_data.get("hf_token", "")
        if saved_token:
            self.hf_token_entry.insert(0, saved_token)

        # Device Selection
        self.device_label = ctk.CTkLabel(self.sidebar_frame, text="Computing Device:", anchor="w")
        self.device_label.grid(row=3, column=0, padx=20, pady=(10, 0))
        
        # 저장된 장치 설정 불러오기 (없으면 기본값)
        saved_device = self.config_data.get("device", "cuda" if is_cuda_available() else "cpu")
        self.device_var = ctk.StringVar(value=saved_device)
        self.device_option = ctk.CTkOptionMenu(self.sidebar_frame, values=["cuda", "cpu"], variable=self.device_var)
        self.device_option.grid(row=4, column=0, padx=20, pady=(5, 10))

        # --- 추가된 설정 저장 버튼 ---
        self.save_settings_button = ctk.CTkButton(self.sidebar_frame, text="설정 저장", 
                                                 fg_color="#3498db", hover_color="#2980b9",
                                                 command=self.save_settings_action)
        self.save_settings_button.grid(row=5, column=0, padx=20, pady=20)

        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=6, column=0, padx=20, pady=(20, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"], command=self.change_appearance_mode)
        self.appearance_mode_optionemenu.grid(row=7, column=0, padx=20, pady=(5, 20))
        self.appearance_mode_optionemenu.set("Dark")

        # --- Main Frame (이전과 동일) ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=15)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        drop_text = "이곳에 영상 파일을 끌어다 놓으세요\n(또는 클릭하여 선택)" if self.dnd_available else "클릭하여 영상을 선택하세요"
        self.drop_label = ctk.CTkLabel(self.main_frame, text=drop_text, 
                                       font=ctk.CTkFont(size=16), width=500, height=250, 
                                       fg_color=("gray85", "gray25"), corner_radius=10)
        self.drop_label.grid(row=0, column=0, padx=40, pady=(40, 20), sticky="nsew")
        
        if self.dnd_available:
            try:
                self.drop_label.drop_target_register(DND_FILES)
                self.drop_label.dnd_bind('<<Drop>>', self.handle_drop)
            except: pass
            
        self.drop_label.bind("<Button-1>", lambda e: self.select_file())

        self.status_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.status_frame.grid(row=1, column=0, padx=40, pady=20, sticky="ew")
        self.status_frame.grid_columnconfigure(0, weight=1)

        self.file_path_label = ctk.CTkLabel(self.status_frame, text="선택된 파일: 없음", text_color="gray")
        self.file_path_label.grid(row=0, column=0, pady=5)

        self.progress_bar = ctk.CTkProgressBar(self.status_frame)
        self.progress_bar.grid(row=1, column=0, pady=10, sticky="ew")
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(self.status_frame, text="준비 완료", font=ctk.CTkFont(weight="bold"))
        self.status_label.grid(row=2, column=0, pady=5)

        self.convert_button = ctk.CTkButton(self.main_frame, text="대사집 변환 시작", 
                                           font=ctk.CTkFont(size=15, weight="bold"),
                                           height=45, fg_color="#2ecc71", hover_color="#27ae60",
                                           command=self.start_conversion_thread)
        self.convert_button.grid(row=2, column=0, padx=40, pady=(0, 40), sticky="ew")

        self.selected_file_path = ""

    def save_settings_action(self):
        """현재 입력된 토큰과 장치 설정을 config.json에 저장합니다."""
        hf_token = self.hf_token_entry.get().strip()
        device = self.device_var.get()
        
        new_config = {
            "hf_token": hf_token,
            "device": device
        }
        
        if save_config(new_config):
            messagebox.showinfo("성공", "설정이 안전하게 저장되었습니다.")
        else:
            messagebox.showerror("오류", "설정 저장 중 문제가 발생했습니다.")

    def change_appearance_mode(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def handle_drop(self, event):
        file_path = event.data.strip('{}')
        if file_path.lower().endswith(('.mp4', '.mov', '.mkv', '.avi')):
            self.set_selected_file(file_path)
        else:
            messagebox.showerror("오류", "지원하지 않는 파일 형식입니다.")

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mov *.mkv *.avi")])
        if file_path:
            self.set_selected_file(file_path)

    def set_selected_file(self, file_path):
        self.selected_file_path = file_path
        self.file_path_label.configure(text=f"선택된 파일: {os.path.basename(file_path)}", text_color="white")
        self.status_label.configure(text="변환 대기 중")

    def update_status(self, text, progress):
        self.status_label.configure(text=text)
        self.progress_bar.set(progress)
        self.update_idletasks()

    def start_conversion_thread(self):
        if not self.selected_file_path:
            messagebox.showwarning("경고", "파일을 먼저 선택해주세요.")
            return
        
        token = self.hf_token_entry.get().strip()
        if not token:
            messagebox.showwarning("경고", "Hugging Face 토큰을 입력해주세요.")
            return

        self.convert_button.configure(state="disabled")
        threading.Thread(target=self.run_conversion, args=(token,), daemon=True).start()

    def run_conversion(self, token):
        try:
            video_path = self.selected_file_path
            device = self.device_var.get()
            output_json = video_path.rsplit('.', 1)[0] + "_script.json"
            
            # 1단계: 오디오 추출
            self.update_status("1/4: 오디오 추출 중 (FFmpeg)...", 0.1)
            audio_path = extract_audio(video_path)
            
            # 2단계: 화자 분리
            self.update_status("2/4: 화자 분리 중 (Diarization)...", 0.3)
            diarizer = DiarizationEngine(hf_token=token, device=device)
            dia_segments = diarizer.diarize(audio_path)
            
            # 3단계: 음성 인식
            self.update_status("3/4: 음성 인식 중 (STT)...", 0.6)
            stt_engine = STTEngine(device=device)
            stt_segments, _ = stt_engine.transcribe(audio_path)
            
            # 4단계: 병합 및 저장
            self.update_status("4/4: 데이터 병합 및 저장 중...", 0.9)
            final_script = merge_stt_diarization(stt_segments, dia_segments)
            save_as_json(final_script, os.path.basename(video_path), output_json)
            
            self.update_status("변환 성공!", 1.0)
            messagebox.showinfo("완료", f"변환이 완료되었습니다!\n{output_json}")
            
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
        except Exception as e:
            logger.error(f"변환 오류: {str(e)}")
            messagebox.showerror("오류", f"변환 중 오류가 발생했습니다:\n{str(e)}")
            self.update_status("오류 발생", 0)
        finally:
            self.convert_button.configure(state="normal")
