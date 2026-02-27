"""
Sprint Analyzer — Modal Backend (RTMPose Edition)

Key changes from YOLO version:
  - Two-stage pipeline: RTMDet-x (detector) + RTMPose-x (pose estimator)
  - Person selection by largest bounding box area, not confidence score
  - Only detected frames are recorded in frame_data / summary stats
  - Summary windowed to middle 60% of detected frames (steady-state sprint)
  - Improved trunk angle using partial visibility fallback
  - `detected` flag per frame so frontend can render gaps correctly
  - `coverage` field in analysis JSON (% frames with valid detection)
  - All frontend JSON fields preserved — zero frontend changes needed
"""

import json
import math
import os
import uuid
from pathlib import Path

import modal

app = modal.App("sprint-analyzer")

# ── Docker image ──────────────────────────────────────────────────────────────
# Pin mmcv to the exact wheel that matches torch 2.1 + cu118.
# Version mismatches in the mm* ecosystem cause silent failures, so pin everything.

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "git", "wget")
    .pip_install(
        "torch==2.0.1",
        "torchvision==0.15.2",
        index_url="https://download.pytorch.org/whl/cu118",
    )
    # openmim is the official OpenMMLab installer — it detects the installed
    # torch/cuda version at install time and picks the correct pre-built
    # mmcv wheel automatically, bypassing the unreliable CDN index entirely.
    .pip_install("openmim")
    .run_commands(
        "mim install mmengine==0.10.3",
        "mim install 'mmcv>=2.0.0,<2.2.0'",
        "mim install mmdet==3.2.0",
        "mim install mmpose==1.3.1",
    )
    .pip_install(
        "numpy",
        "fastapi",
        "python-multipart",
        "uvicorn",
        "slowapi",
        "opencv-python-headless",
    )
    .run_commands(
        "mkdir -p /opt/rtm/weights",
        # RTMDet-x — best general-purpose person detector
        "wget -q -O /opt/rtm/weights/rtmdet_x.pth "
        "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
        "rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth",
        # RTMPose-x — best accuracy for single-person pose on COCO
        "wget -q -O /opt/rtm/weights/rtmpose_x.pth "
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
        "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.pth",
    )
)

volume = modal.Volume.from_name("sprint-analyzer-videos", create_if_missing=True)
VOLUME_PATH = Path("/videos")


# ── Skeleton / angle config (COCO-17, unchanged from YOLO version) ────────────

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

KPT_RECORD = [
    (11, "lhip"),
    (12, "rhip"),
    (13, "lknee"),
    (14, "rknee"),
    (15, "lankle"),
    (16, "rankle"),
]

ANGLE_NAMES = ["R Hip", "L Hip", "R Knee", "L Knee", "R Elbow", "L Elbow", "Trunk"]

# RTMPose confidence scores are well-calibrated at 0.3
MIN_CONF = 0.3


# ── Math helpers ──────────────────────────────────────────────────────────────

def angle_at_vertex(a, v, b):
    """Interior angle in degrees at vertex v between rays v->a and v->b."""
    import numpy as np
    va, vb = a - v, b - v
    n1, n2 = np.linalg.norm(va), np.linalg.norm(vb)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    return math.degrees(math.acos(float(np.clip(va.dot(vb) / (n1 * n2), -1.0, 1.0))))


def trunk_angle(kpts, confs):
    """
    Forward lean of the trunk relative to vertical (degrees).
    Uses whatever shoulders/hips are visible — handles partial occlusion.
    0 = upright, positive = leaning forward.
    """
    import numpy as np
    sh_ok = [i for i in [5, 6]   if confs[i] >= MIN_CONF]
    hp_ok = [i for i in [11, 12] if confs[i] >= MIN_CONF]
    if not sh_ok or not hp_ok:
        return None
    mid_sh = np.mean([kpts[i] for i in sh_ok],  axis=0)
    mid_hp = np.mean([kpts[i] for i in hp_ok], axis=0)
    vec  = mid_sh - mid_hp   # points UP when athlete is upright
    norm = math.hypot(float(vec[0]), float(vec[1]))
    if norm < 1e-6:
        return None
    # Image Y increases downward so "true up" = [0, -1]
    # cos θ = vec · [0,-1] / |vec| = -vec_y / |vec|
    return math.degrees(math.acos(float(
        __import__('numpy').clip(-vec[1] / norm, -1.0, 1.0)
    )))


# ── Kalman filter ─────────────────────────────────────────────────────────────

class KalmanKeypoint:
    """Constant-velocity 2-D Kalman filter for one keypoint."""

    def __init__(self):
        import cv2
        import numpy as np
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self.kf          = kf
        self.initialised = False

    def update(self, x: float, y: float):
        import numpy as np
        if not self.initialised:
            self.kf.statePost = np.array([[x],[y],[0],[0]], np.float32)
            self.initialised  = True
            return x, y
        self.kf.predict()
        s = self.kf.correct(np.array([[x],[y]], np.float32)).flatten()
        return float(s[0]), float(s[1])

    def predict_only(self):
        if not self.initialised:
            return None, None
        s = self.kf.predict().flatten()
        return float(s[0]), float(s[1])


class KalmanSkeleton:
    def __init__(self):
        self.filters = [KalmanKeypoint() for _ in range(17)]

    def update(self, kpts, confs):
        smoothed = kpts.copy()
        for i, (x, y) in enumerate(kpts):
            if confs[i] >= MIN_CONF:
                sx, sy = self.filters[i].update(x, y)
            else:
                px, py = self.filters[i].predict_only()
                sx, sy = (px, py) if px is not None else (x, y)
            smoothed[i] = [sx, sy]
        return smoothed

    def predict_all(self):
        """Advance all filters without a measurement (undetected frame)."""
        for f in self.filters:
            f.predict_only()


# ── Angle smoother ────────────────────────────────────────────────────────────

class AngleSmoother:
    """EMA that only updates on real measurements; returns last known on None."""

    def __init__(self, alpha: float = 0.18):
        self.alpha = alpha
        self.state: dict = {}

    def update(self, measurements: dict) -> dict:
        out = {}
        for k, v in measurements.items():
            if v is None:
                out[k] = self.state.get(k)
            else:
                prev          = self.state.get(k, v)
                smoothed      = (1 - self.alpha) * prev + self.alpha * v
                self.state[k] = smoothed
                out[k]        = smoothed
        return out


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_skeleton(frame, kpts, confs):
    import cv2
    for idx, (i, j) in enumerate(SKELETON):
        if confs[i] < MIN_CONF or confs[j] < MIN_CONF:
            continue
        cv2.line(frame,
                 (int(kpts[i, 0]), int(kpts[i, 1])),
                 (int(kpts[j, 0]), int(kpts[j, 1])),
                 BONE_COLORS[idx], 3, cv2.LINE_AA)
    for i, (x, y) in enumerate(kpts):
        if confs[i] < MIN_CONF:
            continue
        col = COL_LEFT if i in LEFT_KPT else (COL_RIGHT if i in RIGHT_KPT else COL_MID)
        cv2.circle(frame, (int(x), int(y)), 6, col,          -1, cv2.LINE_AA)
        cv2.circle(frame, (int(x), int(y)), 7, (20, 20, 20),  1, cv2.LINE_AA)


def draw_trunk_line(frame, kpts, confs):
    import cv2
    import numpy as np
    sh_ok = [i for i in [5, 6]   if confs[i] >= MIN_CONF]
    hp_ok = [i for i in [11, 12] if confs[i] >= MIN_CONF]
    if not sh_ok or not hp_ok:
        return
    mid_sh = np.mean([kpts[i] for i in sh_ok],  axis=0).astype(int)
    mid_hp = np.mean([kpts[i] for i in hp_ok], axis=0).astype(int)
    cv2.line(frame, tuple(mid_sh), tuple(mid_hp), (255, 220, 50), 2, cv2.LINE_AA)


def draw_hud(frame, angles: dict, detected: bool, coverage_pct: float):
    import cv2
    x, y    = 15, 45
    line_h  = 26
    pad     = 10
    panel_w = 240
    panel_h = (len(angles) + 2) * line_h + 2 * pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "JOINT ANGLES", (x + pad, y + pad + 13),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 220, 50), 1, cv2.LINE_AA)

    for row, (label, val) in enumerate(angles.items()):
        yy = y + pad + (row + 1) * line_h + 6
        if not detected or val is None:
            color, text = (80, 80, 80), f"{label:<12}  ---"
        else:
            color = (50, 230, 100) if val < 90 else (50, 200, 255)
            text  = f"{label:<12} {val:>5.1f}\u00b0"
        cv2.putText(frame, text, (x + pad, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    # Coverage bar at bottom of HUD
    bar_y  = y + panel_h - line_h
    bar_x0 = x + pad
    bar_w  = panel_w - 2 * pad
    cv2.rectangle(frame, (bar_x0, bar_y), (bar_x0 + bar_w, bar_y + 6), (40, 40, 40), -1)
    fill  = int(bar_w * coverage_pct / 100.0)
    bcol  = (50, 230, 100) if coverage_pct >= 70 else (50, 200, 255) if coverage_pct >= 40 else (80, 80, 200)
    cv2.rectangle(frame, (bar_x0, bar_y), (bar_x0 + fill, bar_y + 6), bcol, -1)
    cv2.putText(frame, f"cov {coverage_pct:.0f}%", (bar_x0, bar_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)


def draw_title(frame, progress_pct: float):
    import cv2
    cv2.putText(frame, "Stridelab",
                (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 220, 50), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{progress_pct:.0f}%",
                (frame.shape[1] - 70, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)


# ── RTMPose model loading + inference ─────────────────────────────────────────

def build_inferencer():
    """
    Build RTMDet-x + RTMPose-x inferencer.
    Called once inside the Modal function after GPU is available.

    Config names are resolved by MMPose from its internal registry —
    no config file path needed.
    """
    from mmpose.apis import MMPoseInferencer

    return MMPoseInferencer(
        # RTMPose-x at 384x288 — best accuracy in the RTMPose family
        pose2d='rtmpose-x_8xb256-700e_body8-halpe26-384x288',
        pose2d_weights='/opt/rtm/weights/rtmpose_x.pth',

        # RTMDet-x — best single-stage detector for person bounding boxes
        det_model='rtmdet-x_8xb32-300e_coco',
        det_weights='/opt/rtm/weights/rtmdet_x.pth',

        # Only detect persons (COCO class 0)
        det_cat_ids=[0],

        device='cuda',
        show=False,
        draw_heatmap=False,
    )


def infer_frame(inferencer, frame_bgr):
    """
    Run RTMDet + RTMPose on one BGR frame.

    Returns (kpts, confs, bbox) for the most prominent person, or
    (None, None, None) if no person found.

    Person selection: largest bounding-box area.
    This handles videos with background spectators or multiple athletes
    by reliably targeting the closest/largest person in frame.
    """
    import numpy as np

    result      = next(inferencer(frame_bgr, return_datasamples=False))
    predictions = result.get('predictions', [[]])[0]

    if not predictions:
        return None, None, None

    best_kpts  = None
    best_confs = None
    best_bbox  = None
    best_area  = -1.0

    for pred in predictions:
        bbox = pred.get('bbox', [[0, 0, 1, 1]])[0]   # [x1, y1, x2, y2, score]
        kp   = pred.get('keypoints', [])
        kp_s = pred.get('keypoint_scores', [])

        if len(kp) < 17 or len(kp_s) < 17:
            continue

        x1, y1, x2, y2 = bbox[:4]
        area = float((x2 - x1) * (y2 - y1))

        if area > best_area:
            best_area  = area
            best_kpts  = np.array(kp,   dtype=np.float32)[:17]
            best_confs = np.array(kp_s, dtype=np.float32)[:17]
            best_bbox  = [x1, y1, x2, y2]

    return best_kpts, best_confs, best_bbox


# ── Main processing function ──────────────────────────────────────────────────

@app.function(
    image=gpu_image,
    gpu="A10G",
    timeout=600,
    volumes={str(VOLUME_PATH): volume},
    secrets=[modal.Secret.from_name("sprint-analyzer-secrets")],
    memory=8192,   # RTMPose + RTMDet need more headroom than single-stage YOLO
)
def process_video(job_id: str):
    import cv2
    import numpy as np

    volume.reload()

    job_dir       = VOLUME_PATH / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path    = job_dir / "input.mp4"
    output_path   = job_dir / "output.mp4"
    status_path   = job_dir / "status.txt"
    analysis_path = job_dir / "analysis.json"

    def write_status(msg: str):
        status_path.write_text(msg)
        volume.commit()

    # ── Load models ───────────────────────────────────────────────────────
    write_status("loading_model")
    inferencer = build_inferencer()

    # ── Video metadata ────────────────────────────────────────────────────
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

    kalman          = KalmanSkeleton()
    smoother        = AngleSmoother(alpha=0.18)
    display_angles  = {k: None for k in ANGLE_NAMES}
    frame_data: list[dict] = []
    detected_count  = 0
    frame_idx       = 0

    write_status("processing:0")

    cap2 = cv2.VideoCapture(str(input_path))

    while True:
        ret, frame = cap2.read()
        if not ret:
            break

        # ── RTMPose inference ──────────────────────────────────────────────
        kpts_raw, confs, bbox = infer_frame(inferencer, frame)
        detected = kpts_raw is not None

        if detected:
            detected_count += 1

            # Smooth keypoint positions through the Kalman filter
            kpts = kalman.update(kpts_raw, confs)

            # Draw annotated skeleton onto the frame
            draw_skeleton(frame, kpts, confs)
            draw_trunk_line(frame, kpts, confs)

            # Compute joint angles — None where keypoints are low confidence
            raw: dict = {}
            for (a, v, b, label) in ANGLE_DEFS:
                if confs[a] >= MIN_CONF and confs[v] >= MIN_CONF and confs[b] >= MIN_CONF:
                    raw[label] = angle_at_vertex(kpts[a], kpts[v], kpts[b])
                else:
                    raw[label] = None
            raw["Trunk"] = trunk_angle(kpts, confs)

            # EMA smoothing — only updates state on real measurements
            display_angles = smoother.update(raw)

            # Record keypoints for frontend ankle-velocity phase detection
            kpt_record = {}
            for kpt_idx, kpt_name in KPT_RECORD:
                if confs[kpt_idx] >= MIN_CONF:
                    kpt_record[kpt_name] = [
                        round(float(kpts[kpt_idx][0]), 1),
                        round(float(kpts[kpt_idx][1]), 1),
                    ]
                else:
                    kpt_record[kpt_name] = None

            # Only append to frame_data on real detections.
            # This means summary stats are computed only over frames where a
            # person was actually found — no undetected frames polluting averages.
            frame_data.append({
                "frame":    frame_idx,
                "time":     round(frame_idx / fps, 3),
                "detected": True,
                "angles": {
                    k: round(v, 1) if v is not None else None
                    for k, v in display_angles.items()
                },
                "kpts": kpt_record,
            })

        else:
            # Advance Kalman filters without a measurement so they stay
            # ready to resume tracking cleanly after a brief occlusion.
            kalman.predict_all()
            # Do NOT append to frame_data — keeps all stats clean.

        coverage_pct = 100.0 * detected_count / max(frame_idx + 1, 1)
        draw_hud(frame, display_angles, detected, coverage_pct)
        pct = 100.0 * frame_idx / max(total, 1)
        draw_title(frame, pct)
        writer.write(frame)

        frame_idx += 1
        if frame_idx % 60 == 0:
            write_status(f"processing:{pct:.1f}")

    cap2.release()
    writer.release()

    # ── Re-encode for web ─────────────────────────────────────────────────
    import subprocess
    web_path = job_dir / "output_web.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(output_path),
         "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(web_path)],
        check=True, capture_output=True,
    )
    web_path.replace(output_path)

    # ── Summary statistics ────────────────────────────────────────────────
    # Window to the middle 60% of detected frames to exclude warm-up /
    # deceleration phases and give a cleaner steady-state sprint measurement.
    n = len(frame_data)
    if n >= 10:
        lo, hi     = int(n * 0.20), int(n * 0.80)
        win_frames = frame_data[lo:hi]
    else:
        win_frames = frame_data

    summary = {}
    for name in ANGLE_NAMES:
        vals = [
            f["angles"].get(name)
            for f in win_frames
            if f["angles"].get(name) is not None
        ]
        if vals:
            arr = np.array(vals, dtype=np.float32)
            summary[name] = {
                "mean": round(float(arr.mean()), 1),
                "min":  round(float(arr.min()),  1),
                "max":  round(float(arr.max()),  1),
                "std":  round(float(arr.std()),  1),
            }
        else:
            summary[name] = None

    analysis = {
        "job_id":          job_id,
        "fps":             round(fps, 2),
        "total_frames":    total,
        "detected_frames": detected_count,
        "coverage":        round(100.0 * detected_count / max(total, 1), 1),
        "width":           width,
        "height":          height,
        "duration_s":      round(total / max(fps, 1), 2),
        "summary":         summary,
        "frames":          frame_data,   # only detected frames
    }

    analysis_path.write_text(json.dumps(analysis))
    volume.commit()
    write_status("done")


# ── FastAPI web layer ─────────────────────────────────────────────────────────

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
    api     = FastAPI(title="Sprint Analyzer API")
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
            if request.headers.get("X-API-Key", "") != API_SECRET:
                raise HTTPException(403, "Invalid or missing API key.")

    def check_upload_token(request: Request):
        if not API_SECRET:
            return
        token = request.headers.get("X-Upload-Token", "")
        try:
            payload, sig = token.rsplit(".", 1)
        except ValueError:
            raise HTTPException(403, "Invalid upload token.")
        expected = hmac.new(
            API_SECRET.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, sig):
            raise HTTPException(403, "Invalid upload token.")
        try:
            expires = int(payload)
        except ValueError:
            raise HTTPException(403, "Invalid upload token.")
        if time.time() > expires:
            raise HTTPException(403, "Upload token expired.")

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
            return {"status": "done",          "progress": 100}
        elif raw.startswith("processing:"):
            return {"status": "processing",    "progress": round(float(raw.split(":")[1]), 1)}
        elif raw == "loading_model":
            return {"status": "loading_model", "progress": 0}
        else:
            return {"status": "queued",        "progress": 0}

    @api.get("/download/{job_id}")
    async def download(job_id: str, request: Request):
        check_api_key(request)
        volume.reload()
        output = VOLUME_PATH / job_id / "output.mp4"
        if not output.exists():
            raise HTTPException(404, "Output not ready yet.")
        return FileResponse(
            str(output), media_type="video/mp4",
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