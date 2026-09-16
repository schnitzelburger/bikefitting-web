import cv2
import os
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def berechne_winkel(p1, p2, p3):
    wert = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
    return abs(wert) % 360

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

def analysiere_video(video_pfad, dateiname, status_speicher, ergebnis_speicher, a_thigh, b_calf):
    target_flexion = 35.0

    gamma_rad = math.radians(180.0 - target_flexion)
    c_val = math.sqrt(a_thigh**2 + b_calf**2 - 2 * a_thigh * b_calf * math.cos(gamma_rad))
    dc_dgamma_rad = (a_thigh * b_calf * math.sin(gamma_rad)) / c_val
    mm_per_degree = dc_dgamma_rad * (math.pi / 180.0)

    status_speicher[dateiname] = "Lade Modell und suche Koerperpunkte..."
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

    status_speicher[dateiname] = "Berechne Winkel und Totpunkte..."
    side_is_left = np.mean(left_confs) > np.mean(right_confs)
    idx = [5,7,9,11,13,15] if side_is_left else [6,8,10,12,14,16]

    flexions, torsos, elbows, shoulders, hips = [], [], [], [], []
    
    for data in frames_data:
        if data is None:
            for arr in [flexions, torsos, elbows, shoulders, hips]: arr.append(None)
            continue
            
        xy, confs = data
        sh, el, wr, hi, kn, an = [xy[i] for i in idx]
        
        flexions.append(180.0 - get_angle(hi, kn, an))
        torsos.append(abs(np.degrees(np.arctan2(hi[1] - sh[1], hi[0] - sh[0]))))
        elbows.append(180.0 - get_angle(sh, el, wr))
        shoulders.append(get_angle(hi, sh, el))
        hips.append(get_angle(sh, hi, kn))

    flexions = interpolate_nones(flexions)
    torsos = interpolate_nones(torsos)
    elbows = interpolate_nones(elbows)
    shoulders = interpolate_nones(shoulders)
    hips = interpolate_nones(hips)

    start_f = int(fps)
    end_f = frame_count - int(fps)
    
    minima = []
    for i in range(start_f + 5, end_f - 5):
        if flexions[i] == min(flexions[i-5:i+6]):
            if not minima or (i - minima[-1] > fps / 3):
                minima.append(i)

    cycle_lengths = np.diff(minima)
    med_cycle = np.median(cycle_lengths) if len(cycle_lengths) > 0 else fps
    window = max(3, int(med_cycle / 4))
    if window % 2 == 0: window += 1
    hw = window // 2

    ut_frames, ut_values = [], []
    for m in minima:
        w_start, w_end = max(start_f, m - hw), min(end_f, m + hw + 1)
        x = np.arange(w_start, w_end)
        y = flexions[w_start:w_end]
        if len(x) > 2:
            poly = np.polyfit(x, y, 2)
            if poly[0] > 0:
                vx = -poly[1] / (2 * poly[0])
                vy = poly[0]*vx**2 + poly[1]*vx + poly[2]
                ut_frames.append(int(vx))
                ut_values.append(vy)

    med_ut_flexion = np.median(ut_values) if ut_values else 0
    med_torso = np.median(torsos[start_f:end_f]) if len(torsos) > end_f else 0
    med_elbow = np.median(elbows[start_f:end_f]) if len(elbows) > end_f else 0
    
    diff = med_ut_flexion - target_flexion
    adjustment_mm = diff * mm_per_degree

    status_speicher[dateiname] = "Erstelle Video..."
    cap = cv2.VideoCapture(video_pfad)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_out = f"static/uploads/temp_{dateiname}"
    out = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))

    def get_font(size):
        try: return ImageFont.truetype("DejaVuSans.ttf", int(size))
        except: return ImageFont.load_default()

    font_big = get_font(60 * scale)
    font_med = get_font(35 * scale)
    font_sml = get_font(20 * scale)

    def color_knee(val): return (50, 200, 50, 255) if 27.5 <= val <= 42.5 else (220, 50, 50, 255)
    def text_knee(val): return "Passt" if 27.5 <= val <= 42.5 else ("Zu tief" if val > 42.5 else "Zu hoch")
    def color_torso(val): return (50, 200, 50, 255) if 40 <= val <= 50 else (220, 200, 50, 255)
    def color_elbow(val): return (50, 200, 50, 255) if 15 <= val <= 30 else (220, 200, 50, 255)

    max_knie = float("-inf")
    bester_frame = None
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        overlay = Image.new('RGBA', (width, height), (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        
        draw.rectangle([0, 0, width, 100 * scale], fill=(20, 20, 20, 220))
        draw.text((20 * scale, 10 * scale), "BIKEFIT", font=font_big, fill=(255,255,255,255))
        
        if frames_data[frame_idx] is not None:
            xy, _ = frames_data[frame_idx]
            sh, el, wr, hi, kn, an = [xy[i] for i in idx]
            schulter = (float(sh[0]), float(sh[1]))
            ellbogen_punkt = (float(el[0]), float(el[1]))
            hand = (float(wr[0]), float(wr[1]))
            huefte = (float(hi[0]), float(hi[1]))
            knie_punkt = (float(kn[0]), float(kn[1]))
            fuss = (float(an[0]), float(an[1]))

            rumpf = berechne_winkel(schulter, huefte, (huefte[0], huefte[1] - 100))
            ellbogen = berechne_winkel(schulter, ellbogen_punkt, hand)
            knie = berechne_winkel(huefte, knie_punkt, fuss)

            cv2.putText(frame, f"Rumpf: {int(rumpf)}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Ellbogen: {int(ellbogen)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if knie > max_knie:
                max_knie = knie
                bester_frame = frame.copy()

            pts = [sh, el, wr, hi, kn, an]
            
            lines = [
                (hi, kn, color_knee(flexions[frame_idx])),
                (kn, an, color_knee(flexions[frame_idx])),
                (sh, hi, color_torso(torsos[frame_idx])),
                (sh, el, color_elbow(elbows[frame_idx])),
                (el, wr, color_elbow(elbows[frame_idx]))
            ]
            for p1, p2, col in lines:
                draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=(0,0,0,255), width=int(12*scale))
                draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=col, width=int(6*scale))
                
            for p in pts:
                draw.ellipse([p[0]-8*scale, p[1]-8*scale, p[0]+8*scale, p[1]+8*scale], fill=(255,255,255,255), outline=(0,0,0,255), width=int(2*scale))
                
            draw.text((kn[0] + 30*scale, kn[1] - 20*scale), f"{flexions[frame_idx]:.1f}°", font=font_med, fill=(255,255,255,255))
        
        tx = width - 350 * scale
        ty = 150 * scale
        
        def draw_tile(y_pos, name, val, band_str, col, eval_str):
            draw.rectangle([tx, y_pos, width - 20*scale, y_pos + 120*scale], fill=(20,20,20,200), outline=col, width=int(3*scale))
            draw.text((tx + 15*scale, y_pos + 10*scale), name, font=font_sml, fill=(200,200,200,255))
            draw.text((width - 120*scale, y_pos + 10*scale), eval_str, font=font_sml, fill=col)
            draw.text((tx + 15*scale, y_pos + 40*scale), f"{val:.1f}°", font=font_med, fill=(255,255,255,255))
            draw.text((tx + 15*scale, y_pos + 90*scale), band_str, font=font_sml, fill=(150,150,150,255))
            
        draw_tile(ty, "Kniewinkel", med_ut_flexion, "Soll 30 bis 40°", color_knee(med_ut_flexion), text_knee(med_ut_flexion))
        draw_tile(ty + 140*scale, "Rumpf", med_torso, "Soll 40 bis 50°", color_torso(med_torso), "")
        draw_tile(ty + 280*scale, "Ellbogen", med_elbow, "Soll 15 bis 30°", color_elbow(med_elbow), "")
        
        gh = 200 * scale
        gy = height - gh - 60*scale
        draw.rectangle([0, gy, width, gy + gh], fill=(20, 20, 20, 220))
        def map_y(val): return gy + gh - ((np.clip(val, 10, 70) - 10) / 60.0) * gh
        draw.rectangle([0, map_y(42.5), width, map_y(27.5)], fill=(50, 200, 50, 50))
        
        for i in range(1, frame_count):
            if i < len(flexions):
                draw.line([((i-1)/frame_count*width, map_y(flexions[i-1])), (i/frame_count*width, map_y(flexions[i]))], fill=(255,255,255,200), width=int(2*scale))
        
        cx = frame_idx / frame_count * width
        draw.line([(cx, gy), (cx, gy + gh)], fill=(255,255,255,255), width=int(2*scale))
        
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
        combined = Image.alpha_composite(frame_pil, overlay)
        res_frame = cv2.cvtColor(np.array(combined), cv2.COLOR_RGBA2BGR)
        out.write(res_frame)
        frame_idx += 1

    out.release()
    cap.release()

    status_speicher[dateiname] = "Kodiere fuer Browser..."
    final_out = f"static/uploads/out_{dateiname}"
    subprocess.run(["ffmpeg", "-y", "-i", temp_out, "-c:v", "libx264", "-pix_fmt", "yuv420p", final_out], capture_output=True)
    if os.path.exists(temp_out): os.remove(temp_out)

    if bester_frame is not None:
        bild_pfad = final_out.replace(".mp4", ".jpg").replace("out_", "bild_")
        cv2.imwrite(bild_pfad, bester_frame)

    if 27.5 <= med_ut_flexion <= 42.5:
        befund = f"Deine Sattelhoehe liegt mit {med_ut_flexion:.1f} Grad Beugung im perfekten Bereich."
    else:
        richtung = "nach oben" if med_ut_flexion > 42.5 else "nach unten"
        befund = f"Dein Kniewinkel betraegt {med_ut_flexion:.1f} Grad.\nDu solltest deinen Sattel um {abs(adjustment_mm):.0f} mm {richtung} verstellen."
    
    befund += f"\n\nHaltung Rumpf liegt bei {med_torso:.1f} Grad, Ellbogen bei {med_elbow:.1f} Grad."
    ergebnis_speicher[dateiname] = befund
