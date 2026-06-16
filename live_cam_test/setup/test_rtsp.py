"""
Verify RTSP camera connectivity and print basic stream info.
Usage:
  python setup/test_rtsp.py rtsp://user:pass@ip:port/stream [rtsp://...]
"""
import sys, time
import cv2

def test_stream(url: str, label: str = ""):
    tag = f"[{label}] " if label else ""
    print(f"{tag}Connecting to {url.split('@')[-1]} ...")  # hide credentials in print
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"{tag}FAILED — could not open stream")
        return False

    # Read a few frames to confirm stable connection
    frames_ok = 0
    t0 = time.time()
    for _ in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            frames_ok += 1

    elapsed = time.time() - t0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if frames_ok == 0:
        print(f"{tag}FAILED — stream opened but no frames decoded")
        return False

    print(f"{tag}OK — {w}x{h} @ {fps:.1f}fps | {frames_ok}/30 frames in {elapsed:.1f}s")
    return True

if __name__ == "__main__":
    urls = sys.argv[1:]
    if not urls:
        print("Usage: python setup/test_rtsp.py <rtsp_url_A> [rtsp_url_B ...]")
        sys.exit(1)

    labels = [f"CAM_{chr(65+i)}" for i in range(len(urls))]
    results = [test_stream(u, l) for u, l in zip(urls, labels)]

    print()
    if all(results):
        print("All cameras OK. Ready to proceed.")
    else:
        print("Some cameras failed. Check URLs, credentials, and network.")
        sys.exit(1)
