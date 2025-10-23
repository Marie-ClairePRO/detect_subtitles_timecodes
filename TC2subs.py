import subprocess
import cv2
from pathlib import Path
import re
import os

def parse_srt_file(srt_path):
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newlines to separate subtitle blocks
    blocks = re.split(r'\n\s*\n', content.strip())
    
    subtitles = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        try:
            index = int(lines[0])
            timecode_line = lines[1]
            text = '\n'.join(lines[2:])
            
            # Parse timecodes: HH:MM:SS,mmm --> HH:MM:SS,mmm
            match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', timecode_line)
            if match:
                start_time = match.group(1)
                end_time = match.group(2)
                
                subtitles.append({
                    'index': index,
                    'start': start_time,
                    'end': end_time,
                    'text': text
                })
        except (ValueError, IndexError):
            continue
    
    return subtitles


def timecode_to_seconds(timecode):
    # Replace comma with dot for milliseconds
    timecode = timecode.replace(',', '.')
    parts = timecode.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_timecode(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def extract_middle_frame(video_path, start_time, end_time, output_path):
    start_sec = timecode_to_seconds(start_time)
    end_sec = timecode_to_seconds(end_time)
    middle_sec = (start_sec + end_sec) / 2

    print(middle_sec)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set position to middle frame
    frame_number = int(middle_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(output_path, frame)
        return True
    return False


def extract_subtitle_with_ollama(frame_path, model="qwen2.5vl"):
    prompt = (
        "Read the subtitle of this image and return only the content of it: "
        f"{os.path.abspath(frame_path)} "
        "Do only return exactly the content of it, do not double words, "
        "and skip a line with hyphens if detected dialogue. "
        "If you think there is nothing written, return empty response, "
        "not one word, no text, no symbol."
    )
    
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        text = result.stdout.strip()
        return text if text else None
    
    except subprocess.TimeoutExpired:
        print(f"Timeout processing {frame_path}")
        return None
    except Exception as e:
        print(f"Error processing {frame_path}: {e}")
        return None


def improve_subtitles_with_ollama(video_path, input_srt, output_srt, 
                                   temp_frames_dir="temp_frames", 
                                   model="qwen2.5vl",
                                   keep_frames=False):
    # Create temp directory
    temp_dir = Path(temp_frames_dir)
    temp_dir.mkdir(exist_ok=True)
    
    # Parse existing subtitles
    print(f"Reading subtitles from: {input_srt}")
    subtitles = parse_srt_file(input_srt)
    print(f"Found {len(subtitles)} subtitle entries\n")
    
    improved_subtitles = []
    
    for i, sub in enumerate(subtitles, 1):
        print(f"[{i}/{len(subtitles)}] Processing subtitle {sub['index']}")
        print(f"  Time: {sub['start']} --> {sub['end']}")
        print(f"  Original: {sub['text'][:60]}...")
        
        # Extract middle frame
        frame_path = temp_dir / f"sub_{sub['index']:04d}.png"
        success = extract_middle_frame(
            video_path, 
            sub['start'], 
            sub['end'], 
            str(frame_path)
        )
        
        if not success:
            print(f"Failed to extract frame, keeping original text")
            improved_subtitles.append(sub)
            continue
        
        # Extract text with Ollama
        new_text = extract_subtitle_with_ollama(str(frame_path), model)
        
        if new_text:
            print(f"---- Improved: {new_text[:60]}...")
            sub['text'] = new_text
        else:
            print(f" !! No text detected, keeping original")
        
        improved_subtitles.append(sub)
        
        # Clean up frame if not keeping
        if not keep_frames:
            frame_path.unlink()
        
        print()
    
    # Write improved SRT file
    write_srt_file(improved_subtitles, output_srt)
    
    print(f"{'='*60}")
    print(f"---- Improved subtitles saved to: {output_srt}")
    print(f"Total entries: {len(improved_subtitles)}")
    if keep_frames:
        print(f"  Frames saved in: {temp_frames_dir}")
    print(f"{'='*60}")


def write_srt_file(subtitles, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for sub in subtitles:
            f.write(f"{sub['index']}\n")
            f.write(f"{sub['start']} --> {sub['end']}\n")
            f.write(f"{sub['text']}\n\n")


if __name__ == "__main__":
    video_path = "Quincy_ep2_deinterlaced.mp4"           
    input_srt = "detected_subtitles.srt"    
    output_srt = "QUINCY_EP2_MGCPB0022516-BIS_EXP_01.srt"   
    
    improve_subtitles_with_ollama(
        video_path=video_path,
        input_srt=input_srt,
        output_srt=output_srt,
        model="qwen2.5vl",
        keep_frames=False 
    )