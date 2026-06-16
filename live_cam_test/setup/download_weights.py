"""
Download TransReID pretrained weights.
Run from live_cam_test/ with the venv active:
  python setup/download_weights.py
"""
import subprocess, sys, pathlib

WEIGHTS_DIR = pathlib.Path(__file__).parent.parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

# TransReID*(ViT) trained on MSMT17 — JPM + SIE + stride=12
# 67.8 mAP / 85.3 R1 on MSMT17 (best zero-shot generalization)
MSMT17_MODEL_ID = "1x6Na97ycxS0t2Dn_0iRKWe1U5ccIqASK"
MSMT17_OUT     = WEIGHTS_DIR / "transreid_msmt17.pth"

def download(file_id, out_path):
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return
    print(f"[download] {out_path.name} ...")
    subprocess.run(
        [sys.executable, "-m", "gdown", file_id, "-O", str(out_path)],
        check=True,
    )
    print(f"[done] saved to {out_path}")

if __name__ == "__main__":
    try:
        import gdown
    except ImportError:
        print("installing gdown...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)

    download(MSMT17_MODEL_ID, MSMT17_OUT)
    print("\nWeights ready:", WEIGHTS_DIR)
