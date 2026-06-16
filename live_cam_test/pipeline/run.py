"""
Live multi-camera detect + track + ReID pipeline.

Usage:
    cd live_cam_test
    .venv/bin/python pipeline/run.py [--config pipeline/config.yaml]

View streams in browser:  http://<server-ip>:8080/cam_a   http://<server-ip>:8080/cam_b
Press Ctrl+C to quit.
To test a different combo: edit config.yaml, re-run.
"""
import sys, pathlib, argparse, threading, time, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, yaml, numpy as np, torch
from PIL import Image

# Force RTSP over TCP — more reliable than UDP for cameras on LAN
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "ByteTrack"))
sys.path.insert(0, str(ROOT / "TransReID"))

from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from yolox.data.data_augment import preproc
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils.visualize import plot_tracking
from yolox.tracking_utils.timer import Timer

# ── config ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(ROOT / "pipeline/config.yaml"))
args = parser.parse_args()

with open(args.config) as f:
    C = yaml.safe_load(f)

DET   = C["detector"]
TRK   = C["tracker"]
REID  = C["reid"]
CAMS  = C["cameras"]
OUT   = C["output"]

# ── BYTETracker needs an args-like object ────────────────────────────────────
class _TrkArgs:
    track_thresh       = TRK["track_thresh"]
    track_buffer       = TRK["track_buffer"]
    match_thresh       = TRK["match_thresh"]
    mot20              = False
    aspect_ratio_thresh = TRK["aspect_ratio_thresh"]
    min_box_area       = TRK["min_box_area"]

# ── load YOLOX ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[detector] device: {device}")

exp = get_exp(None, DET["model"])
exp.num_classes = 1  # ByteTrack weights are person-only
exp.test_conf  = DET["conf_thresh"]
exp.nmsthre    = DET["nms_thresh"]
exp.test_size  = (DET["input_size"], DET["input_size"])

yolox = exp.get_model()
ckpt  = torch.load(str(ROOT / DET["weights"]), map_location="cpu")
yolox.load_state_dict(ckpt["model"])
yolox.to(device).eval()
yolox = fuse_model(yolox)
print(f"[detector] YOLOX loaded from {DET['weights']}")

# ── load ReID ─────────────────────────────────────────────────────────────────
reid_model = None
if REID["enabled"]:
    from embed import TransReIDEmbedder
    reid_model = TransReIDEmbedder(str(ROOT / REID["weights"]), device=str(device))

# ── camera reader (background thread — keeps only the latest frame) ───────────
class CamReader:
    def __init__(self, cam_id, source):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = False
        threading.Thread(target=self._read, daemon=True).start()
        print(f"[camera] '{cam_id}' opened: {source}")

    def _read(self):
        while not self._stop:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def get(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._stop = True
        self.cap.release()

# ── per-frame detection ───────────────────────────────────────────────────────
RGB_MEAN = (0.485, 0.456, 0.406)
STD      = (0.229, 0.224, 0.225)

def detect(frame):
    img, ratio = preproc(frame, exp.test_size, RGB_MEAN, STD)
    t = torch.from_numpy(img).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = yolox(t)
        out = postprocess(out, exp.num_classes, exp.test_conf, exp.nmsthre)
    return out, ratio

# ── crop person regions for each active track ─────────────────────────────────
def crop_tracks(frame, tracks):
    h, w = frame.shape[:2]
    result = []
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.tlbr)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            result.append((t.track_id, Image.fromarray(crop)))
    return result

# ── cosine similarity cross-camera match ──────────────────────────────────────
def match_cameras(gallery, query, thresh):
    """Returns dict: query_track_id -> (gallery_track_id, score)"""
    if not gallery or not query:
        return {}
    g_ids = list(gallery.keys());  G = np.stack([gallery[i] for i in g_ids])
    q_ids = list(query.keys());    Q = np.stack([query[i]   for i in q_ids])
    sim = Q @ G.T  # (Nq, Ng)
    out = {}
    for qi, qid in enumerate(q_ids):
        best = int(np.argmax(sim[qi]))
        score = float(sim[qi, best])
        if score >= thresh:
            out[qid] = (g_ids[best], score)
    return out

# ── MJPEG server (no X display needed — view in browser) ─────────────────────
_latest_frames = {}   # cam_id -> latest JPEG bytes
_frame_locks   = {}

class _MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass  # silence access log
    def do_GET(self):
        cam_id = self.path.lstrip("/")
        if cam_id not in _latest_frames:
            self.send_error(404, f"Camera '{cam_id}' not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with _frame_locks[cam_id]:
                    jpg = _latest_frames[cam_id]
                if jpg is not None:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            pass

def _push_frame(cam_id, vis_frame):
    _, jpg = cv2.imencode(".jpg", vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    with _frame_locks[cam_id]:
        _latest_frames[cam_id] = jpg.tobytes()

def _start_mjpeg_server(port=8080):
    server = ThreadingHTTPServer(("0.0.0.0", port), _MJPEGHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[mjpeg] http://0.0.0.0:{port}/<cam_id>  — open in browser")

# ── setup ─────────────────────────────────────────────────────────────────────
# init MJPEG broadcaster
for c in CAMS:
    _latest_frames[c["id"]] = None
    _frame_locks[c["id"]]   = threading.Lock()
_start_mjpeg_server(OUT.get("mjpeg_port", 8080))

readers  = [CamReader(c["id"], c["source"]) for c in CAMS]
trackers = [BYTETracker(_TrkArgs(), frame_rate=25) for _ in CAMS]
galleries = [{} for _ in CAMS]   # cam_idx -> {track_id: embedding}
timers   = [Timer() for _ in CAMS]

cam_ids = [c["id"] for c in CAMS]
gal_idx = cam_ids.index(REID["gallery_cam"]) if reid_model else 0
qry_idx = cam_ids.index(REID["query_cam"])   if reid_model else 1

time.sleep(1.0)  # let camera threads buffer a frame
print("[pipeline] running — Ctrl+C to quit")

# ── main loop ─────────────────────────────────────────────────────────────────
frame_id = 0
try:
    while True:
        frame_id += 1

        frames = [r.get() for r in readers]
        if any(f is None for f in frames):
            time.sleep(0.01)
            continue

        all_tracks = []
        for i, (frame, tracker, timer) in enumerate(zip(frames, trackers, timers)):
            timer.tic()
            outputs, ratio = detect(frame)
            tracks = []
            if outputs[0] is not None:
                dets = outputs[0].clone()
                dets[:, :4] /= ratio
                tracks = tracker.update(dets, [frame.shape[0], frame.shape[1]], exp.test_size)

            if reid_model and tracks:
                crops = crop_tracks(frame, tracks)
                if crops:
                    tids, pils = zip(*crops)
                    embs = reid_model.embed(list(pils))
                    for tid, emb in zip(tids, embs):
                        galleries[i][tid] = emb

            timer.toc()
            all_tracks.append(tracks)

        # cross-camera match
        matches = {}
        if reid_model and len(readers) >= 2:
            matches = match_cameras(galleries[gal_idx], galleries[qry_idx], REID["similarity_thresh"])

        # push annotated frames to MJPEG server
        for i, (frame, tracks) in enumerate(zip(frames, all_tracks)):
            tlwhs = [t.tlwh for t in tracks]
            tids  = [t.track_id for t in tracks]
            fps   = 1.0 / max(1e-5, timers[i].average_time)
            vis   = plot_tracking(frame, tlwhs, tids, frame_id=frame_id, fps=fps)

            if i == qry_idx and matches:
                for t in tracks:
                    if t.track_id in matches:
                        match_id, score = matches[t.track_id]
                        label = f"{cam_ids[gal_idx]}:{match_id} ({score:.2f})"
                        x = int(t.tlwh[0])
                        y = max(15, int(t.tlwh[1]) - 5)
                        cv2.putText(vis, label, (x, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            _push_frame(cam_ids[i], vis)

            if frame_id % 100 == 0:
                print(f"[{cam_ids[i]}] frame {frame_id}  fps {fps:.1f}  tracks {len(tracks)}")

except KeyboardInterrupt:
    print("\n[pipeline] stopped")

for r in readers:
    r.stop()
