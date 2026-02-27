"""
Sprint Analyzer — Modal Backend
"""

import json
import math
import os
import uuid
from pathlib import Path

import modal

app = modal.App("sprint-analyzer")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "ultralytics>=8.4.0",
        "numpy",
        "fastapi",
        "python-multipart",
        "uvicorn",
        "slowapi",
    )
    .pip_install("opencv-python-headless")
)

volume = modal.Volume.from_name("sprint-analyzer-videos", create_if_missing=True)
VOLUME_PATH = Path("/videos")

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

COL_LEFT  = (50,  230, 100)
COL_RIGHT = (50,  100, 255)
COL_MID   = (255, 220,  50)

LEFT_KPT  = {1, 3, 5, 7, 9,  11, 13, 15}
RIGHT_KPT = {2, 4, 6, 8, 10, 12, 14, 16}

BONE_COLORS = [
    COL_MID, COL_MID, COL_MID, COL_MID,
    COL_LEFT, COL_LEFT,
    COL_RIGHT, COL_RIGHT,
    COL_MID,
    COL_LEFT, COL_RIGHT, COL_MID,
    COL_LEFT, COL_LEFT,
    COL_RIGHT, COL_RIGHT,
]

ANGLE_DEFS = [
    (6,  12, 14, "R Hip"),
    (12, 14, 16, "R Knee"),
    (6,   8, 10, "R Elbow"),
    (5,  11, 13, "L Hip"),
    (11, 13, 15, "L Knee"),
    (5,   7,  9, "L Elbow"),
]

# COCO-17 keypoint indices stored per-frame for ankle-velocity phase detection.
#   In COCO-17 (confirmed by ANGLE_DEFS):
#     11=L hip  12=R hip  13=L knee  14=R knee  15=L ankle  16=R ankle
#
#   The frontend uses ankle y-position (image coords: high y = low = ground)
#   to detect contact frames (local maxima) and velocity zero-crossings
#   (touch-down / toe-off), replacing the old fixed PHASE_OFFSETS approach.
KPT_RECORD = [
    (11, "lhip"),
    (12, "rhip"),
    (13, "lknee"),
    (14, "rknee"),
    (15, "lankle"),
    (16, "rankle"),
]


def angle_at_vertex(a, v, b):
    import numpy as np
    va, vb = a - v, b - v
    n1, n2 = np.linalg.norm(va), np.linalg.norm(vb)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    return math.degrees(math.acos(float(np.clip(va.dot(vb) / (n1 * n2), -1, 1))))


def trunk_angle(kpts):
    import numpy as np
    mid_sh = (kpts[5] + kpts[6]) / 2
    mid_hp = (kpts[11] + kpts[12]) / 2
    vec    = mid_sh - mid_hp
    n      = np.linalg.norm(vec)
    if n < 1e-6:
        return None
    return math.degrees(math.acos(float(np.clip(-vec[1] / n, -1, 1))))


class KalmanKeypoint:
    def __init__(self):
        import cv2
        import numpy as np
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0], [0, 1, 0, 1],
            [0, 0, 1, 0], [0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self.initialised = False

    def update(self, x, y):
        import numpy as np
        m = np.array([[x],[y]], dtype=np.float32)
        if not self.initialised:
            self.kf.statePost = np.array([[x],[y],[0],[0]], dtype=np.float32)
            self.initialised  = True
        self.kf.predict()
        s = self.kf.correct(m).flatten()
        return float(s[0]), float(s[1])

    def predict_only(self):
        s = self.kf.predict().flatten()
        return float(s[0]), float(s[1])


class KalmanSkeleton:
    def __init__(self):
        self.filters = [KalmanKeypoint() for _ in range(17)]

    def update(self, kpts, confs, min_conf=0.3):
        import numpy as np
        smoothed = kpts.copy()
        for i, (x, y) in enumerate(kpts):
            if confs[i] >= min_conf:
                sx, sy = self.filters[i].update(x, y)
            else:
                sx, sy = self.filters[i].predict_only()
            smoothed[i] = [sx, sy]
        return smoothed


class EMA:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.state = {}

    def update(self, d):
        out = {}
        for k, v in d.items():
            if v is None:
                out[k] = self.state.get(k)
                continue
            prev           = self.state.get(k, v)
            self.state[k]  = (1 - self.alpha) * prev + self.alpha * v
            out[k]         = self.state[k]
        return out


def draw_skeleton(frame, kpts, confs, min_conf=0.3):
    import cv2
    for idx, (i, j) in enumerate(SKELETON):
        if confs[i] < min_conf or confs[j] < min_conf:
            continue
        cv2.line(frame,
                 (int(kpts[i,0]), int(kpts[i,1])),
                 (int(kpts[j,0]), int(kpts[j,1])),
                 BONE_COLORS[idx], 3, cv2.LINE_AA)
    for i, (x, y) in enumerate(kpts):
        if confs[i] < min_conf:
            continue
        col = COL_LEFT if i in LEFT_KPT else (COL_RIGHT if i in RIGHT_KPT else COL_MID)
        cv2.circle(frame, (int(x), int(y)), 6, col,        -1, cv2.LINE_AA)
        cv2.circle(frame, (int(x), int(y)), 7, (20,20,20),  1, cv2.LINE_AA)


def draw_hud(frame, angles):
    import cv2
    x, y     = 15, 45
    line_h   = 26
    pad      = 10
    panel_w  = 220
    panel_h  = (len(angles) + 1) * line_h + 2 * pad
    overlay  = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+panel_w, y+panel_h), (15,15,15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, "JOINT ANGLES", (x+pad, y+pad+13),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (255,220,50), 1, cv2.LINE_AA)
    for row, (label, val) in enumerate(angles.items()):
        yy    = y + pad + (row+1)*line_h + 6
        color = (120,120,120) if val is None else (50,230,100) if val < 90 else (50,200,255)
        text  = f"{label:<12} {val:>5.1f}\u00b0" if val is not None else f"{label:<12}  ---"
        cv2.putText(frame, text, (x+pad, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def draw_title(frame, progress_pct):
    import cv2
    cv2.putText(frame, "teebs",
                (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                (255,220,50), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{progress_pct:.0f}%",
                (frame.shape[1]-70, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200,200,200), 1, cv2.LINE_AA)


@app.function(
    image=gpu_image,
    gpu="A10G",
    timeout=600,
    volumes={str(VOLUME_PATH): volume},
    secrets=[modal.Secret.from_name("sprint-analyzer-secrets")],
    memory=4096,
)
def process_video(job_id: str):
    import cv2
    import numpy as np
    from ultralytics import YOLO

    volume.reload()

    job_dir       = VOLUME_PATH / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path    = job_dir / "input.mp4"
    output_path   = job_dir / "output.mp4"
    status_path   = job_dir / "status.txt"
    analysis_path = job_dir / "analysis.json"

    frame_data = []

    def write_status(msg: str):
        status_path.write_text(msg)
        volume.commit()

    write_status("loading_model")
    model = YOLO("yolo26x-pose.pt")

    cap    = cv2.VideoCapture(str(input_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height),
    )

    kalman      = KalmanSkeleton()
    smoother    = EMA(alpha=0.15)
    last_angles = {}
    frame_idx   = 0

    write_status("processing:0")

    results = model.track(
        source=str(input_path),
        tracker="bytetrack.yaml",
        imgsz=1280,
        conf=0.35,
        iou=0.3,
        persist=True,
        stream=True,
        verbose=False,
        device=0,
        half=True,
    )

    cap2 = cv2.VideoCapture(str(input_path))

    for result in results:
        ret, frame = cap2.read()
        if not ret:
            break

        best_kpts  = None
        best_confs = None
        best_score = -1.0

        if result.keypoints is not None and len(result.keypoints) > 0:
            for person in result.keypoints:
                kp    = person.data[0].cpu().numpy()
                score = float(kp[:, 2].mean())
                if score > best_score:
                    best_score = score
                    best_kpts  = kp[:, :2]
                    best_confs = kp[:, 2]

        if best_kpts is not None:
            best_kpts = kalman.update(best_kpts, best_confs, 0.2)
            draw_skeleton(frame, best_kpts, best_confs, 0.2)

            raw = {}
            for (a, v, b, label) in ANGLE_DEFS:
                if best_confs[a] >= 0.2 and best_confs[v] >= 0.2 and best_confs[b] >= 0.2:
                    raw[label] = angle_at_vertex(best_kpts[a], best_kpts[v], best_kpts[b])
                else:
                    raw[label] = None

            if all(best_confs[i] >= 0.2 for i in [5, 6, 11, 12]):
                raw["Trunk"] = trunk_angle(best_kpts)
                mid_sh = ((best_kpts[5] + best_kpts[6]) / 2).astype(int)
                mid_hp = ((best_kpts[11] + best_kpts[12]) / 2).astype(int)
                cv2.line(frame, tuple(mid_sh), tuple(mid_hp), (255,220,50), 2, cv2.LINE_AA)
            else:
                raw["Trunk"] = None

            last_angles = smoother.update(raw)

            # Record ankle/knee/hip positions for frontend phase detection.
            # ankle y in image coords: high y = low on screen = near ground.
            # Local maxima in ankle-y → ground contact frames (Touch Down).
            # Velocity zero-crossings → Toe Off boundaries.
            kpt_record = {}
            for kpt_idx, kpt_name in KPT_RECORD:
                if best_confs[kpt_idx] >= 0.2:
                    kpt_record[kpt_name] = [
                        round(float(best_kpts[kpt_idx][0]), 1),
                        round(float(best_kpts[kpt_idx][1]), 1),
                    ]
                else:
                    kpt_record[kpt_name] = None

            frame_data.append({
                "frame": frame_idx,
                "time":  round(frame_idx / fps, 3),
                "angles": {k: round(v, 1) if v is not None else None
                           for k, v in last_angles.items()},
                "kpts": kpt_record,
            })

        draw_hud(frame, last_angles)
        pct = 100.0 * frame_idx / max(total, 1)
        draw_title(frame, pct)
        writer.write(frame)

        frame_idx += 1
        if frame_idx % 60 == 0:
            write_status(f"processing:{pct:.1f}")

    cap2.release()
    writer.release()

    import subprocess
    web_path = job_dir / "output_web.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(output_path),
         "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(web_path)],
        check=True, capture_output=True,
    )
    web_path.replace(output_path)

    angle_names = ["R Hip", "L Hip", "R Knee", "L Knee", "R Elbow", "L Elbow", "Trunk"]
    summary = {}
    for name in angle_names:
        vals = [f["angles"].get(name) for f in frame_data
                if f["angles"].get(name) is not None]
        if vals:
            import numpy as _np
            arr = _np.array(vals)
            summary[name] = {
                "mean": round(float(arr.mean()), 1),
                "min":  round(float(arr.min()), 1),
                "max":  round(float(arr.max()), 1),
                "std":  round(float(arr.std()), 1),
            }
        else:
            summary[name] = None

    analysis = {
        "job_id":       job_id,
        "fps":          round(fps, 2),
        "total_frames": total,
        "width":        width,
        "height":       height,
        "duration_s":   round(total / max(fps, 1), 2),
        "summary":      summary,
        "frames":       frame_data,
    }
    analysis_path.write_text(json.dumps(analysis))
    volume.commit()
    write_status("done")


@app.function(
    image=gpu_image,
    volumes={str(VOLUME_PATH): volume},
    memory=512,
    secrets=[modal.Secret.from_name("sprint-analyzer-secrets")],
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import hashlib
    import hmac
    import time
    from fastapi import FastAPI, File, HTTPException, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address, default_limits=["200/hour"])
    api = FastAPI(title="Sprint Analyzer API")
    api.state.limiter = limiter
    api.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    allowed_origin = os.environ.get("ALLOWED_ORIGIN", "*")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[allowed_origin],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    API_SECRET = os.environ.get("API_SECRET")

    def check_api_key(request: Request):
        if API_SECRET:
            key = request.headers.get("X-API-Key", "")
            if key != API_SECRET:
                raise HTTPException(status_code=403, detail="Invalid or missing API key.")

    def check_upload_token(request: Request):
        if not API_SECRET:
            return
        token = request.headers.get("X-Upload-Token", "")
        try:
            payload, sig = token.rsplit(".", 1)
        except ValueError:
            raise HTTPException(status_code=403, detail="Invalid upload token.")
        expected = hmac.new(
            API_SECRET.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, sig):
            raise HTTPException(status_code=403, detail="Invalid upload token.")
        try:
            expires = int(payload)
        except ValueError:
            raise HTTPException(status_code=403, detail="Invalid upload token.")
        if time.time() > expires:
            raise HTTPException(status_code=403, detail="Upload token expired.")

    @api.post("/analyze")
    @limiter.limit("5/hour")
    async def analyze(request: Request, file: UploadFile = File(...)):
        check_upload_token(request)
        if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            raise HTTPException(400, "Please upload an MP4, MOV, AVI or MKV file.")
        contents = await file.read()
        if len(contents) > 500 * 1024 * 1024:
            raise HTTPException(400, "File too large. Max 500 MB.")
        job_id  = str(uuid.uuid4())
        job_dir = VOLUME_PATH / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "input.mp4").write_bytes(contents)
        (job_dir / "status.txt").write_text("queued")
        volume.commit()
        process_video.spawn(job_id)
        return JSONResponse({"job_id": job_id})

    @api.get("/status/{job_id}")
    async def status(job_id: str, request: Request):
        check_api_key(request)
        volume.reload()
        status_file = VOLUME_PATH / job_id / "status.txt"
        if not status_file.exists():
            raise HTTPException(404, "Job not found.")
        raw = status_file.read_text().strip()
        if raw == "done":
            return {"status": "done", "progress": 100}
        elif raw.startswith("processing:"):
            pct = float(raw.split(":")[1])
            return {"status": "processing", "progress": round(pct, 1)}
        elif raw == "loading_model":
            return {"status": "loading_model", "progress": 0}
        else:
            return {"status": "queued", "progress": 0}

    @api.get("/download/{job_id}")
    async def download(job_id: str, request: Request):
        check_api_key(request)
        volume.reload()
        output = VOLUME_PATH / job_id / "output.mp4"
        if not output.exists():
            raise HTTPException(404, "Output not ready yet.")
        return FileResponse(
            str(output),
            media_type="video/mp4",
            filename=f"sprint_analyzed_{job_id[:8]}.mp4",
        )

    @api.get("/analysis/{job_id}")
    async def get_analysis(job_id: str, request: Request):
        check_api_key(request)
        volume.reload()
        path = VOLUME_PATH / job_id / "analysis.json"
        if not path.exists():
            raise HTTPException(404, "Analysis not ready yet.")
        return JSONResponse(json.loads(path.read_text()))

    return api