import subprocess
import cv2
from pathlib import Path
import re
import os

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

from ollama import chat
from ollama import ChatResponse

import requests
import base64

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


def extract_middle_frame(video_path, start_time, end_time, output_path, reduce_size=False):
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
        if reduce_size:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, frame)
        return True
    return False


def extract_subtitle_with_ollama(frame_path, prompt, model="qwen2.5vl", method="python"): 
    assert method in ["subprocess", "python", "requests"], f"method {method} not implemented yet"   
    try:
        if method == "subprocess":
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
        
            output = result.stdout
            if model in ["qwen3-vl:8b"]:
                output = output.split("...done thinking.")[-1]
            text = output.strip()

        elif method == "python":
            response: ChatResponse = chat(model=model, messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [frame_path],
                },
                ])
            text = response.message.content

        elif method == "requests":
            with open(frame_path, "rb") as f:
                img_64 = base64.b64encode(f.read()).decode()
            payload = {
                "model" : model,
                "messages" : [{
                    "role" : "user",
                    "content" : prompt,
                    "images" : [img_64]
                }],
                "stream" : False
            }
            try:
                resp = requests.post(
                    "http://localhost:11434/api/chat",
                    jeson=payload,
                    timeout=60
                )
                resp.raise_for_status()

            except requests.RequestException as e:
                print(e)

            text = resp.json()["message"]["content"]

        return text if text else None
    
    except subprocess.TimeoutExpired:
        print(f"Timeout processing {frame_path}")
        return None
    except Exception as e:
        print(f"Error processing {frame_path}: {e}")
        return None


def make_prompt_with_path(prompt= None, frame=None):
    frame_str = "FRAME_PATH" if frame is None else os.path.abspath(frame)
    if prompt is None: 
        prompt = f"""You are a subtitle OCR engine.
                Extract only the FRENCH subtitle text of this image.
                {frame_str}
                Rules:
                - Output only the subtitle
                - No explanation
                - No markdown
                - No quotes
                - No extra words
                - If missing, output <NO_SUBTITLE>

                Text:"""
    else:
        prompt = f"""{prompt} : {frame_str}"""
    return prompt

def simple_prompt():
    return "Read the french caption of this image. Output only the subtitle, without markdown, nor explanation, nor quotes. If missing, output <NO_SUBTITLE>"

def improve_subtitles_with_ollama(video_path, input_srt, output_srt, 
                                   temp_frames_dir="temp_frames", 
                                   model="qwen2.5vl",
                                   prompt=None,
                                   restart_at=None,
                                   keep_frames=False,
                                   save_live = False,
                                   method = "python"):
    # Create temp directory
    temp_dir = Path(temp_frames_dir)
    temp_dir.mkdir(exist_ok=True)
    
    # Parse existing subtitles
    print(f"Reading subtitles from: {input_srt}")
    subtitles = parse_srt_file(input_srt)
    print(f"Found {len(subtitles)} subtitle entries\n")
    
    improved_subtitles = []
    if method == "subprocess":
        prompt = make_prompt_with_path(prompt)
    elif prompt is None:
        prompt = simple_prompt()
    print(prompt)

    for i, sub in enumerate(subtitles, 1):
        if restart_at is not None and timecode_to_seconds(sub["end"]) < timecode_to_seconds(restart_at):
            continue
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
        if method == "subprocess":
            prompt = make_prompt_with_path(prompt, frame_path)
        new_text = extract_subtitle_with_ollama(str(frame_path), prompt, model, method=method)
        
        if new_text:
            print(f"---- Improved: {new_text}")
            sub['text'] = new_text
        else:
            print("ollama failed to respond, removing TC")
            continue

        if sub["text"] != "<NO_SUBTITLE>":
            improved_subtitles.append(sub)
        else:
            print("empty text, removing TC")

        # Clean up frame if not keeping
        if not keep_frames:
            frame_path.unlink()
        
        print()
        if save_live and (i+1)%10 == 0:
            write_srt_file(improved_subtitles, output_srt)
    
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, type=str, help="video to read subs from")
    parser.add_argument("--subs_path", required=True, type=str, help="subs with TCs to extract")
    parser.add_argument("--subs_output", type=str, default=None, help="output path to save found subs")
    parser.add_argument("--model", default="qwen3-vl:8b", help="model to read caption, please use qwen vl")
    parser.add_argument("--prompt", type=str, default=None, help="your prompt")
    parser.add_argument("--restart_at", type=str, help="restart from srt file, format hh:mm:ss")
    parser.add_argument("--save_live", action="store_true", help="save srt file regularly")
    parser.add_argument("--method", type=str, default="python", help="method to call ollama : python (default) for python ollama library, requests, or subprocess")

    args = parser.parse_args()

    if args.subs_output is None:
        args.subs_output = args.video_path.split(".")[0] + ".srt"
    
    assert not args.save_live or args.subs_path != args.subs_output, "Don't save subs if it erases original ones"

    improve_subtitles_with_ollama(
        video_path=args.video_path,
        input_srt=args.subs_path,
        output_srt=args.subs_output,
        model=args.model,
        prompt = args.prompt,
        restart_at = args.restart_at,
        keep_frames=False,
        save_live = args.save_live,
        method = args.method
    )

if __name__ == "__main__":
    main()
