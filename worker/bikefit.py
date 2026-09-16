import cv2
import os
import math
import subprocess
import time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

TARGET_FLEXION = 35.0
MIN_CONFIDENCE = 0.5

def waehle_geraet():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def berechne_huefthoehe(kurbel_mm, oberschenkel_mm, unterschenkel_mm, beugung_grad):
    gamma = math.radians(180.0 - beugung_grad)
    beinlaenge = math.sqrt(
        oberschenkel_mm ** 2 + unterschenkel_mm ** 2
        - 2 * oberschenkel_mm * unterschenkel_mm * math.cos(gamma)
    )
    return beinlaenge + kurbel_mm

def parabel_scheitelpunkt(values, center, window_width):
    half_window = max(1, window_width // 2)
    start = max(0, center - half_window)
    end = min(len(values), center + half_window + 1)
    x = np.arange(start, end, dtype=float)
    y = np.asarray(values[start:end], dtype=float)
    if len(x) < 3:
        return float(center), float(values[center])
    coefficients = np.polyfit(x, y, 2)
    if coefficients[0] <= 0:
        return float(center), float(values[center])
    vertex = float(np.clip(-coefficients[1] / (2 * coefficients[0]), start, end - 1))
    return vertex, float(np.polyval(coefficients, vertex))

def ermittle_ut_werte(flexions, ankle_heights, fps, frame_count):
    start = int(fps)
    end = frame_count - int(fps)
    smoothed = np.convolve(ankle_heights, np.ones(5) / 5, mode="same")
    ankle_minima = []
    minimum_distance = max(2, int(fps / 3))
    for index in range(start + 1, end - 1):
        if smoothed[index] <= smoothed[index - 1] and smoothed[index] < smoothed[index + 1]:
            if not ankle_minima or index - ankle_minima[-1] >= minimum_distance:
                ankle_minima.append(index)
    if len(ankle_minima) < 2:
        raise ValueError("Keine ausreichende Knoechelschwingung fuer Trittzyklen erkannt.")
    cycle_width = max(3, int(round(float(np.median(np.diff(ankle_minima))) / 4.0)))
    frames, values = [], []
    for cycle_start, cycle_end in zip(ankle_minima[:-1], ankle_minima[1:]):
        local = cycle_start + int(np.argmin(flexions[cycle_start:cycle_end + 1]))
        vertex, value = parabel_scheitelpunkt(flexions, local, cycle_width)
        frames.append(vertex)
        values.append(value)
    return frames, values, cycle_width

def test_messkette():
    huefthoehe = berechne_huefthoehe(172.5, 430.0, 440.0, TARGET_FLEXION)
    assert huefthoehe > 172.5
    rng = np.random.default_rng(42)
    errors = []
    for phase in np.linspace(0, 2 * math.pi, 8, endpoint=False):
        frames = np.arange(720, dtype=float)
        flexion = TARGET_FLEXION + 0.08 * rng.normal(size=len(frames))
        ankle = np.sin(2 * math.pi * frames / 60 + phase)
        _, values, _ = ermittle_ut_werte(flexion, ankle, 60.0, len(frames))
        errors.extend(abs(value - TARGET_FLEXION) for value in values)
    assert float(np.mean(errors)) < 0.5

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

def analysiere_video(video_pfad, dateiname, status_speicher, ergebnis_speicher, a_thigh, b_calf, abbruch_signale, fortschritt_speicher):
    job_name = os.path.basename(dateiname)
    if job_name.startswith("out_"):
        job_name = job_name[4:]
    target_flexion = TARGET_FLEXION

    gamma_rad = math.radians(180.0 - target_flexion)
    c_val = math.sqrt(a_thigh**2 + b_calf**2 - 2 * a_thigh * b_calf * math.cos(gamma_rad))
    dc_dgamma_rad = (a_thigh * b_calf * math.sin(gamma_rad)) / c_val
    mm_per_degree = dc_dgamma_rad * (math.pi / 180.0)

    status_speicher[job_name] = "Pruefe Video und lade Modell..."
    device = waehle_geraet()
    model = YOLO("yolo11x-pose.pt")
    model.to(device)
    cap = cv2.VideoCapture(video_pfad)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale = width / 1080.0
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise ValueError("Video konnte nicht gelesen werden.")
    # temporary fix for videos with less than 10 seconds duration
    #if frame_count / fps <= 10.0:
    #    cap.release()
    #    raise ValueError("Das Video muss laenger als zehn Sekunden sein.")
    fortschritt_speicher[job_name] = {
        "frames_analysiert": 0,
        "gesamt_frames": frame_count,
        "prozent": 0.0,
    }

    frames_data = []
    left_confs, right_confs = [], []
    letzter_log = time.monotonic()
    print(f"Analyse {dateiname}: 0.0% (0/{frame_count} Frames)", flush=True)

    count = 0
    while True:
        if abbruch_signale.get(job_name, False):
            cap.release()
            status_speicher[job_name] = "Abgebrochen"
            return False

        ret, frame = cap.read()
        if not ret: break
        
        count += 1
        prozent = count / frame_count * 100 if frame_count else 0.0
        fortschritt_speicher[job_name] = {
            "frames_analysiert": count,
            "gesamt_frames": frame_count,
            "prozent": prozent,
        }
        status_speicher[job_name] = f"Lese Positionen Bild {count} von {frame_count} ({prozent:.2f}%)"

        if time.monotonic() - letzter_log >= 3:
            print(f"Analyse {dateiname}: {prozent:.2f}% ({count}/{frame_count} Frames)", flush=True)
            letzter_log = time.monotonic()
        
        results = model(frame, verbose=False, device=device)
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

    status_speicher[job_name] = "Berechne Winkel und Totpunkte..."
    if not left_confs or not right_confs:
        raise ValueError("Keine Pose mit ausreichender Sichtbarkeit erkannt.")
    side_is_left = np.mean(left_confs) > np.mean(right_confs)
    idx = [5,7,9,11,13,15] if side_is_left else [6,8,10,12,14,16]

    flexions, torsos, elbows, shoulders, hips = [], [], [], [], []
    ankle_heights = []
    profile_ratios = []
    visibility = []
    
    for data in frames_data:
        if data is None:
            for arr in [flexions, torsos, elbows, shoulders, hips, ankle_heights]: arr.append(None)
            continue
            
        xy, confs = data
        sh, el, wr, hi, kn, an = [xy[i] for i in idx]
        other_shoulder = xy[6 if side_is_left else 5]
        shoulder_hip_distance = np.linalg.norm(sh - hi)
        if shoulder_hip_distance > 0:
            profile_ratios.append(abs(float(sh[0] - other_shoulder[0])) / shoulder_hip_distance)
        visibility.append(float(np.mean(confs[[idx[3], idx[4], idx[5]]])))
        ankle_heights.append(float(an[1]))
        
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
    ankle_heights = interpolate_nones(ankle_heights)

    if not profile_ratios or float(np.mean(profile_ratios)) >= 0.25:
        raise ValueError("Kein ausreichendes Seitenprofil erkannt (Schulterabstand ist zu gross).")
    if not visibility or float(np.mean(visibility)) <= MIN_CONFIDENCE:
        raise ValueError("Huefte, Knie und Knoechel sind nicht ausreichend sichtbar.")

    start_f = int(fps)
    end_f = frame_count - int(fps)
    
    ut_frames, ut_values, window = ermittle_ut_werte(
        flexions, ankle_heights, fps, frame_count
    )

    med_ut_flexion = np.median(ut_values) if ut_values else 0
    med_torso = np.median(torsos[start_f:end_f]) if len(torsos) > end_f else 0
    med_elbow = np.median(elbows[start_f:end_f]) if len(elbows) > end_f else 0
    med_shoulder = np.median(shoulders[start_f:end_f]) if len(shoulders) > end_f else 0
    med_hip = np.median(hips[start_f:end_f]) if len(hips) > end_f else 0
    
    diff = med_ut_flexion - target_flexion
    adjustment_mm = diff * mm_per_degree

    status_speicher[job_name] = "Erstelle Video..."
    cap = cv2.VideoCapture(video_pfad)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_out = f"{dateiname}.tmp.mp4"
    out = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))

    def get_font(size, numeric=False):
        candidates = [
            "/System/Library/Fonts/SFNS.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf" if numeric else "",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return ImageFont.truetype(path, int(size))
        return ImageFont.load_default()

    font_big = get_font(60 * scale)
    font_med = get_font(35 * scale)
    font_num = get_font(30 * scale, numeric=True)
    font_sml = get_font(20 * scale)

    def color_knee(val):
        if 30 <= val <= 40: return (50, 200, 50, 255)
        if val < 28 or val > 42: return (220, 50, 50, 255)
        return (220, 180, 50, 255)
    def text_knee(val):
        if 30 <= val <= 40: return "Passt"
        if val > 42: return "Sattel zu tief"
        if val < 28: return "Sattel zu hoch"
        return "Grenzbereich"
    def color_torso(val): return (50, 200, 50, 255) if 40 <= val <= 50 else (220, 200, 50, 255)
    def color_elbow(val): return (50, 200, 50, 255) if 15 <= val <= 30 else (220, 200, 50, 255)

    max_knie = float("-inf")
    bester_frame = None
    frame_idx = 0
    while True:
        if abbruch_signale.get(job_name, False):
            cap.release()
            out.release()
            if os.path.exists(temp_out): os.remove(temp_out)
            status_speicher[job_name] = "Abgebrochen"
            return False

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
            draw.text((10 * scale, 10 * scale), f"Rumpf: {int(rumpf)}", font=font_med, fill=(50, 220, 50, 255))
            draw.text((10 * scale, 55 * scale), f"Ellbogen: {int(ellbogen)}", font=font_med, fill=(50, 220, 50, 255))
            radius = 65 * scale
            draw.arc([kn[0] - radius, kn[1] - radius, kn[0] + radius, kn[1] + radius], 205, 330, fill=color_knee(flexions[frame_idx]), width=max(2, int(5 * scale)))
            draw.text((kn[0] + 30 * scale, kn[1] - 20 * scale), f"{flexions[frame_idx]:.1f} live", font=font_num, fill=(255,255,255,255))

            left = min(p[0] for p in pts) - 20 * scale
            top = min(p[1] for p in pts) - 20 * scale
            right = max(p[0] for p in pts) + 20 * scale
            bottom = max(p[1] for p in pts) + 20 * scale
            corner = 45 * scale
            for x, y, dx, dy in [(left, top, 1, 1), (right, top, -1, 1), (left, bottom, 1, -1), (right, bottom, -1, -1)]:
                draw.line([x, y, x + dx * corner, y], fill=(255,255,255,255), width=max(1, int(4 * scale)))
                draw.line([x, y, x, y + dy * corner], fill=(255,255,255,255), width=max(1, int(4 * scale)))
        
        tx = width - 350 * scale
        ty = 150 * scale
        
        def draw_tile(y_pos, name, val, band_str, col, eval_str):
            draw.rectangle([tx, y_pos, width - 20*scale, y_pos + 120*scale], fill=(20,20,20,200), outline=col, width=int(3*scale))
            draw.text((tx + 15*scale, y_pos + 10*scale), name, font=font_sml, fill=(200,200,200,255))
            draw.text((width - 120*scale, y_pos + 10*scale), eval_str, font=font_sml, fill=col)
            draw.text((tx + 15*scale, y_pos + 40*scale), f"{val:.1f}°", font=font_med, fill=(255,255,255,255))
            draw.text((tx + 15*scale, y_pos + 90*scale), band_str, font=font_sml, fill=(150,150,150,255))
            
        draw_tile(ty, "Kniewinkel UT", med_ut_flexion, "Soll 30 bis 40 Grad", color_knee(med_ut_flexion), text_knee(med_ut_flexion))
        draw_tile(ty + 140*scale, "Rumpf", med_torso, "Erfahrungswert: 40 bis 50 Grad", color_torso(med_torso), "Erfahrungswert")
        draw_tile(ty + 280*scale, "Ellbogen", med_elbow, "Erfahrungswert: 15 bis 30 Grad", color_elbow(med_elbow), "Erfahrungswert")
        draw_tile(ty + 420*scale, "Schulterwinkel", med_shoulder, "Nur Messwert", (210,210,210,255), "")
        draw_tile(ty + 560*scale, "Hueftwinkel", med_hip, "Nur Messwert", (210,210,210,255), "")
        
        gh = 200 * scale
        gy = height - gh - 60*scale
        draw.rectangle([0, gy, width, gy + gh], fill=(20, 20, 20, 220))
        def map_y(val): return gy + gh - ((np.clip(val, 10, 70) - 10) / 60.0) * gh
        draw.rectangle([0, map_y(40), width, map_y(30)], fill=(50, 200, 50, 50))
        
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

    status_speicher[job_name] = "Kodiere fuer Browser..."
    final_out = dateiname
    subprocess.run(["ffmpeg", "-y", "-i", temp_out, "-c:v", "libx264", "-pix_fmt", "yuv420p", final_out], capture_output=True)
    if os.path.exists(temp_out): os.remove(temp_out)

    if bester_frame is not None:
        bild_name = os.path.splitext(job_name)[0]
        bild_pfad = os.path.join(os.path.dirname(final_out), f"bild_{bild_name}.jpg")
        cv2.imwrite(bild_pfad, bester_frame)

    if 30 <= med_ut_flexion <= 40:
        befund = f"Deine Sattelhoehe liegt mit {med_ut_flexion:.1f} Grad Beugung im perfekten Bereich."
    else:
        richtung = "nach oben" if med_ut_flexion > 42 else "nach unten"
        befund = f"Dein Kniewinkel betraegt {med_ut_flexion:.1f} Grad.\nDu solltest deinen Sattel um {abs(adjustment_mm):.0f} mm {richtung} verstellen."
    
    befund += f"\n\nHaltung Rumpf liegt bei {med_torso:.1f} Grad, Ellbogen bei {med_elbow:.1f} Grad."
    befund += f"\nSchulterwinkel: {med_shoulder:.1f} Grad, Hueftwinkel: {med_hip:.1f} Grad (nur Messwerte)."
    ergebnis_speicher[job_name] = befund
    return True
