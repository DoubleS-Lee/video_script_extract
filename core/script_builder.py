import json
from utils.logger import setup_logger

logger = setup_logger("script_builder")

def merge_stt_diarization(stt_segments, diarization_segments):
    """
    STT 텍스트 구간과 Diarization 화자 구간을 병합합니다.
    구간 겹침(Intersection)이 가장 큰 화자를 해당 텍스트의 화자로 지정합니다.
    """
    logger.info("데이터 병합(STT + Diarization) 시작")
    
    final_script = []
    
    for idx, stt_seg in enumerate(stt_segments):
        stt_start = stt_seg["start"]
        stt_end = stt_seg["end"]
        
        # 각 화자 구간과의 겹치는 시간 계산
        intersections = []
        for dia_seg in diarization_segments:
            dia_start = dia_seg["start"]
            dia_end = dia_seg["end"]
            
            # 겹치는 구간 계산 (max(start) ~ min(end))
            inter_start = max(stt_start, dia_start)
            inter_end = min(stt_end, dia_end)
            
            overlap = max(0, inter_end - inter_start)
            if overlap > 0:
                intersections.append((overlap, dia_seg["speaker"]))
        
        # 겹치는 시간이 가장 많은 화자 선택
        if intersections:
            best_speaker = max(intersections, key=lambda x: x[0])[1]
        else:
            best_speaker = "UNKNOWN"
            
        final_script.append({
            "index": idx + 1,
            "speaker": best_speaker,
            "start_time": format_timestamp(stt_start),
            "end_time": format_timestamp(stt_end),
            "text": stt_seg["text"],
            "raw_start": stt_start,
            "raw_end": stt_end
        })
        
    logger.info(f"데이터 병합 완료: 총 {len(final_script)}개의 대사 생성")
    return final_script

def format_timestamp(seconds: float) -> str:
    """초 단위를 HH:MM:SS.mmm 포맷으로 변환"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{msecs:03d}"

def save_as_json(script_data, video_filename, output_path):
    """최종 대사집을 JSON 형식으로 저장"""
    final_data = {
        "video_info": {
            "filename": video_filename,
            "total_segments": len(script_data)
        },
        "script": script_data
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"최종 JSON 대사집 저장 성공: {output_path}")
    return output_path
