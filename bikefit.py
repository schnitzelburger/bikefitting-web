import cv2
import os
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def get_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def interpolate_nones(arr):
    for i in range(len(arr)):
        if arr[i] is None:
            left, right = None, None
            for j in range(i-1, -1, -1):
                if arr[j] is not None: left = arr[j]; break
            for j in range(i+1, len(arr)):
                if arr[j] is not None: right = arr[j]; break
            if left is not None and right is not None: arr[i] = left + (right - left) * 0.5
            elif left is not None: arr[i] = left
            elif right is not None: arr[i] = right
            else: arr[i] = 0.0
    return arr

def analysiere_video(video_pfad, dateiname, status_speicher):
    status_speicher[dateiname] = "Lade Modell"
    model = YOLO("yolo11x-pose.pt")
    cap = cv2.VideoCapture(video_pfad)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale = width / 1080.0

    frames_data = []
    left_confs, right_confs = [], []

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if count % 10 == 0:
            status_speicher[dateiname] = f"Lese Positionen Bild {count} von {frame_count}"
        count += 1
        
        results = model(frame, verbose=False)
        if len(results) == 0 or len(results[0].keypoints) == 0:
            frames_data.append(None)
            continue
            
        kpts = results[0].keypoints
        confs = kpts.conf[0].cpu().numpy()
        xy = kpts.xy[0].cpu().numpy()
        
        frames_data.append((xy, confs))
        left_confs.append(np.mean([confs[5], confs[7], confs[9], confs[11], confs[13], confs[15]]))
        right_confs.append(np.mean([confs[6], confs[8], confs[10], confs[12], confs[14], confs[16]]))

    cap.release()

    status_speicher[dateiname] = "Werte Daten aus"
    side_is_left = np.mean(left_confs) > np.mean(right_confs)
    idx = [5,7,9,11,13,15] if side_is_left else [6,8,10,12,14,16]

    flexions, torsos, elbows = [], [], []
    
    for data in frames_data:
        if data is None:
            flexions.append(None); torsos.append(None); elbows.append(None)
            continue
            
        xy, confs = data
        sh, el, wr, hi, kn, an = [xy[i] for i in idx]
        
        flexions.append(180.0 - get_angle(hi, kn, an))
        torsos.append(abs(np.degrees(np.arctan2(hi[1] - sh[1], hi[0] - sh[0]))))
        elbows.append(180.0 - get_angle(sh, el, wr))

    flexions = interpolate_nones(flexions)
    torsos = interpolate_nones(torsos)
    elbows = interpolate_nones(elbows)

    status_speicher[dateiname] = "Zeichne Overlay"
    cap = cv2.VideoCapture(video_pfad)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_out = f"static/uploads/temp_{dateiname}"
    out = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))

    def color_knee(val): return (50, 200, 50, 255) if 27.5 <= val <= 42.5 else (220, 50, 50, 255)
    def color_torso(val): return (50, 200, 50, 255) if 40 <= val <= 50 else (220, 200, 50, 255)
    def color_elbow(val): return (50, 200, 50, 255) if 15 <= val <= 30 else (220, 200, 50, 255)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx % 10 == 0:
            status_speicher[dateiname] = f"Zeichne Video Bild {frame_idx} von {frame_count}"
            
        overlay = Image.new('RGBA', (width, height), (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        
        if frames_data[frame_idx] is not None:
            xy, _ = frames_data[frame_idx]
            sh, el, wr, hi, kn, an = [xy[i] for i in idx]
            lines = [(hi, kn, color_knee(flexions[frame_idx])), (kn, an, color_knee(flexions[frame_idx])), (sh, hi, color_torso(torsos[frame_idx])), (sh, el, color_elbow(elbows[frame_idx])), (el, wr, color_elbow(elbows[frame_idx]))]
            for p1, p2, col in lines:
                draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=col, width=int(12*scale))
            
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
        combined = Image.alpha_composite(frame_pil, overlay)
        res_frame = cv2.cvtColor(np.array(combined), cv2.COLOR_RGBA2BGR)
        out.write(res_frame)
        frame_idx += 1

    out.release()
    cap.release()

    status_speicher[dateiname] = "Kodiere fuer Web"
    final_out = f"static/uploads/out_{dateiname}"
    subprocess.run(["ffmpeg", "-y", "-i", temp_out, "-c:v", "libx264", "-pix_fmt", "yuv420p", final_out], capture_output=True)
    if os.path.exists(temp_out): os.remove(temp_out)
