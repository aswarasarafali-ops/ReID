# ReID Test Pipeline

A simple live pipeline for testing person detection, tracking, and Re-Identification (ReID) across multiple cameras. Swap detector, tracker, or ReID model by editing one config file.

**Stack:** YOLOX (detect) · ByteTrack (track) · TransReID (ReID) · MJPEG HTTP server (view in browser)

---

## What it does

```
Camera A (RTSP) ──┐
                  ├─► YOLOX detect ──► ByteTrack track ──► TransReID embed ──► cross-cam match ──► browser
Camera B (RTSP) ──┘
```

- Each camera gets its own track IDs within that camera
- TransReID embeds each tracked person crop into a feature vector
- Camera B tracks are matched against Camera A's gallery by cosine similarity
- Matched tracks show a yellow label: `cam_a:3 (0.82)` — which cam_a identity, and confidence

---

## Prerequisites

- Python 3.12
- CUDA-capable GPU (tested on RTX 2070, 8 GB)
- `pip install gdown` available for weight downloads

---

## Setup

### 1. Clone this repo

```bash
git clone https://github.com/aswarasarafali-ops/ReID.git
cd ReID
```

### 2. Clone ByteTrack and TransReID

```bash
cd live_cam_test
git clone https://github.com/ifzhang/ByteTrack.git
git clone https://github.com/damo-cv/TransReID.git
```

### 3. Create virtual environment and install dependencies

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install opencv-python pyyaml numpy pillow loguru lap \
    pycocotools cython_bbox scikit-image thop tabulate filterpy yacs
```

### 4. Patch TransReID for PyTorch 2.x

`torch._six` was removed in PyTorch 2.x. Apply the fix:

```bash
sed -i 's/from torch._six import container_abcs/import collections.abc as container_abcs/' \
    TransReID/model/backbones/vit_pytorch.py
```

### 5. Download weights

```bash
# ByteTrack (YOLOX-S trained on MOT17, person-only, 69 MB)
mkdir -p ByteTrack/weights
gdown 1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj -O ByteTrack/weights/bytetrack_s_mot17.pth

# TransReID* ViT-Base trained on MSMT17 (400 MB)
mkdir -p TransReID/weights
gdown 1x6Na97ycxS0t2Dn_0iRKWe1U5ccIqASK -O TransReID/weights/transreid_msmt17.pth
```

---

## Configuration

Edit `pipeline/config.yaml` before running:

```yaml
cameras:
  - id: cam_a
    source: rtsp://user:pass@ip:port/stream
  - id: cam_b
    source: rtsp://user:pass@ip2:port/stream

detector:
  model: yolox-s          # yolox-nano / yolox-tiny / yolox-s / yolox-m / yolox-l / yolox-x
  weights: ByteTrack/weights/bytetrack_s_mot17.pth
  conf_thresh: 0.5
  nms_thresh: 0.45
  input_size: 608

tracker:
  track_thresh: 0.5
  track_buffer: 30
  match_thresh: 0.8
  min_box_area: 100
  aspect_ratio_thresh: 1.6

reid:
  enabled: true
  weights: TransReID/weights/transreid_msmt17.pth
  similarity_thresh: 0.75   # cosine similarity threshold for cross-cam match
  gallery_cam: cam_a
  query_cam: cam_b

output:
  mjpeg_port: 8080
```

### Swapping components

| To change | Edit in config.yaml |
|---|---|
| Detector size | `detector.model: yolox-m` (nano/tiny/s/m/l/x) |
| Detector weights | `detector.weights:` path |
| Tracker sensitivity | `tracker.track_thresh`, `tracker.match_thresh` |
| ReID model | `reid.weights:` path |
| Camera source | `cameras[].source` |
| Match confidence | `reid.similarity_thresh` |

---

## Run

```bash
cd live_cam_test
.venv/bin/python pipeline/run.py
```

View streams in a browser (no display required):

```
http://<server-ip>:8080/cam_a
http://<server-ip>:8080/cam_b
```

Press **Ctrl+C** to stop.

---

## Troubleshooting

**`453 Not Enough Bandwidth`** — Camera's main stream is already in use by another client. Try the substream:
```yaml
source: rtsp://user:pass@ip:554/Streaming/Channels/102   # 102 = substream
```

**`401 Unauthorized` in logs** — This is normal. It's the RTSP digest-auth challenge and is automatically resolved. The stream will still connect.

**No frames / black browser window** — Camera failed to connect. Check the RTSP URL with:
```bash
.venv/bin/python setup/test_rtsp.py rtsp://user:pass@ip:port/stream
```

**CUDA out of memory** — Switch to a smaller detector (`yolox-nano`) or reduce `input_size` to `416`.

---

## Project structure

```
live_cam_test/
├── pipeline/
│   ├── config.yaml      # ← edit this to change detector / tracker / reid / cameras
│   ├── run.py           # main pipeline script
│   └── embed.py         # TransReID inference wrapper
├── ByteTrack/           # cloned from github.com/ifzhang/ByteTrack
├── TransReID/           # cloned from github.com/damo-cv/TransReID
├── setup/
│   ├── test_rtsp.py     # verify camera connectivity
│   └── download_weights.py
└── notes/
    └── camera_layout.md
```
