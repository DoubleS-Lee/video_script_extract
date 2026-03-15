import os
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES

from config import load_config, save_config
from core.audio_processor import extract_audio
from core.stt_engine import STTEngine
from core.diarization import DiarizationEngine
from core.script_builder import merge_stt_diarization, save_as_json
from utils.gpu_helper import get_available_device, setup_cuda_path
from utils.logger import setup_logger

logger = setup_logger("gui")

# TkinterDnD.Tk를 기본으로 사용하여 금지 아이콘 문제 해결
class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        
        # 1. 기본 설정 및 테마 적용
        self.title("AI Video Script Pro (GPU Accelerated)")
        self.geometry("850x500")
        
        # customtkinter의 테마를 수동으로 입히기
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # 배경색 강제 설정 (ctk 느낌 유지)
        self.configure(bg="#242424")
        
        setup_cuda_path()
        self.config_data = load_config()

        # 2. 드래그 앤 드롭 등록 (창 전체)
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.handle_drop)

        # 3. 레이아웃 구성
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="AI Script Pro", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.hf_label = ctk.CTkLabel(self.sidebar_frame, text="Hugging Face Token:", anchor="w")
        self.hf_label.grid(row=1, column=0, padx=20, pady=(20, 0))
        self.hf_token_entry = ctk.CTkEntry(self.sidebar_frame, placeholder_text="Enter HF Token...", show="*", width=180)
        self.hf_token_entry.grid(row=2, column=0, padx=20, pady=(5, 10))
        
        saved_token = self.config_data.get("hf_token", "")
        if saved_token:
            self.hf_token_entry.insert(0, saved_token)

        self.device_info_label = ctk.CTkLabel(self.sidebar_frame, text="Device: GPU (CUDA)", text_color="#2ecc71")
        self.device_info_label.grid(row=3, column=0, padx=20, pady=(10, 0))

        self.save_settings_button = ctk.CTkButton(self.sidebar_frame, text="설정 저장", 
                                                 fg_color="#3498db", hover_color="#2980b9",
                                                 command=self.save_settings_action)
        self.save_settings_button.grid(row=4, column=0, padx=20, pady=20)

        # --- Main Frame ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=15)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Drop Zone (여기에 파일을 던지면 handle_drop 실행)
        drop_text = "이곳에 영상 파일을 끌어다 놓으세요\n\n(또는 클릭하여 파일 선택)"
        self.drop_label = ctk.CTkLabel(self.main_frame, text=drop_text, 
                                       font=ctk.CTkFont(size=16), width=500, height=220, 
                                       fg_color=("gray80", "gray20"), corner_radius=15)
        self.drop_label.grid(row=0, column=0, padx=40, pady=(40, 20), sticky="nsew")
        
        # 라벨 클릭 시 파일 선택창 열기
        self.drop_label.bind("<Button-1>", lambda e: self.select_file())

        # 상태 및 진행바
        self.status_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.status_frame.grid(row=1, column=0, padx=40, pady=10, sticky="ew")
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
        hf_token = self.hf_token_entry.get().strip()
        if save_config({"hf_token": hf_token, "device": "cuda"}):
            messagebox.showinfo("성공", "설정이 저장되었습니다.")

    def handle_drop(self, event):
        # 윈도우 탐색기에서 드롭된 경로에서 특수문자 제거
        file_path = event.data.strip('{} \n\t')
        if file_path.lower().endswith(('.mp4', '.mov', '.mkv', '.avi')):
            self.set_selected_file(file_path)
        else:
            messagebox.showerror("오류", "지원하지 않는 파일 형식입니다.")

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mov *.mkv *.avi")])
        if file_path:
            self.set_selected_file(file_path)

    def set_selected_file(self, file_path):
        if os.path.exists(file_path):
            self.selected_file_path = file_path
            self.file_path_label.configure(text=f"선택된 파일: {os.path.basename(file_path)}", text_color="white")
            self.status_label.configure(text="변환 대기 중")

    def update_status(self, text, progress):
        self.status_label.configure(text=text)
        self.progress_bar.set(progress)
        self.update_idletasks()

    def start_conversion_thread(self):
        if not self.selected_file_path:
            messagebox.showwarning("경고", "파일을 선택하거나 끌어다 놓으세요.")
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
            device = get_available_device("cuda")
            output_json = video_path.rsplit('.', 1)[0] + "_script.json"
            
            self.update_status("1/4: 오디오 추출 중...", 0.1)
            audio_path = extract_audio(video_path)
            
            self.update_status(f"2/4: 화자 분리 중 ({device.upper()})...", 0.3)
            diarizer = DiarizationEngine(hf_token=token, device=device)
            dia_segments = diarizer.diarize(audio_path)
            
            self.update_status(f"3/4: 음성 인식 중 ({device.upper()})...", 0.6)
            stt_engine = STTEngine(device=device)
            stt_segments, _ = stt_engine.transcribe(audio_path)
            
            self.update_status("4/4: 데이터 병합 및 저장 중...", 0.9)
            final_script = merge_stt_diarization(stt_segments, dia_segments)
            save_as_json(final_script, os.path.basename(video_path), output_json)
            
            self.update_status("변환 성공!", 1.0)
            messagebox.showinfo("완료", f"변환이 완료되었습니다!\n\n저장 위치: {output_json}")
            
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
        except Exception as e:
            logger.error(f"변환 오류: {str(e)}")
            messagebox.showerror("오류", f"변환 중 오류가 발생했습니다:\n{str(e)}")
            self.update_status("오류 발생", 0)
        finally:
            self.convert_button.configure(state="normal")
