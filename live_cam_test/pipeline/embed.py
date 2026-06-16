"""
TransReID inference wrapper.
Loads a trained model and converts person crops to embeddings.

Usage:
  from pipeline.embed import TransReIDEmbedder
  model = TransReIDEmbedder("weights/transreid_msmt17.pth")
  embeddings = model.embed(list_of_pil_images)  # (N, 3840) numpy array
"""
import sys, pathlib
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# TransReID repo must be on the path
REPO = pathlib.Path(__file__).parent.parent / "TransReID"
sys.path.insert(0, str(REPO))

from config import cfg as _default_cfg
from model.make_model import make_model

# MSMT17 dataset constants (must match the weights)
MSMT17_NUM_CLASSES = 1041
MSMT17_NUM_CAMERAS = 15

# Normalization used by TransReID (NOT ImageNet mean/std)
_MEAN = [0.5, 0.5, 0.5]
_STD  = [0.5, 0.5, 0.5]

_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


class TransReIDEmbedder:
    def __init__(
        self,
        weights_path: str,
        config_file: str = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[TransReIDEmbedder] using device: {self.device}")

        # Load config — use MSMT17 TransReID stride config as base
        cfg = _default_cfg.clone()
        default_config = str(REPO / "configs/MSMT17/vit_transreid_stride.yml")
        if config_file:
            cfg.merge_from_file(config_file)
        else:
            cfg.merge_from_file(default_config)
        # Skip loading the ImageNet backbone — full TransReID weights cover everything
        cfg.merge_from_list(["MODEL.PRETRAIN_CHOICE", "self"])
        cfg.freeze()

        # Build model
        self.model = make_model(
            cfg,
            num_class=MSMT17_NUM_CLASSES,
            camera_num=MSMT17_NUM_CAMERAS,
            view_num=0,
        )

        # Load trained weights (handles DDP 'module.' prefix automatically)
        self.model.load_param(weights_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"[TransReIDEmbedder] loaded weights from {weights_path}")

    @torch.no_grad()
    def embed(self, images: list, batch_size: int = 64) -> np.ndarray:
        """
        Args:
            images: list of PIL.Image (person crops, any size — resized internally)
        Returns:
            embeddings: np.ndarray of shape (N, D), L2-normalized
        """
        if not images:
            return np.empty((0, 0), dtype=np.float32)

        all_embs = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            tensors = torch.stack([_TRANSFORM(img) for img in batch_imgs]).to(self.device)
            n = tensors.size(0)
            # cam_label=0 for all: SIE tokens are irrelevant for new cameras
            cam_labels = torch.zeros(n, dtype=torch.long).to(self.device)

            feats = self.model(tensors, cam_label=cam_labels)
            feats = F.normalize(feats, p=2, dim=1)
            all_embs.append(feats.cpu().numpy())

        return np.concatenate(all_embs, axis=0)

    def embed_crops_dir(self, crops_dir: str) -> tuple[np.ndarray, list[str]]:
        """
        Load all crops from a directory tree and embed them.
        Returns (embeddings, paths).
        """
        import os
        from tqdm import tqdm

        image_exts = {".jpg", ".jpeg", ".png"}
        paths = []
        for root, _, files in os.walk(crops_dir):
            for f in sorted(files):
                if pathlib.Path(f).suffix.lower() in image_exts:
                    paths.append(os.path.join(root, f))

        print(f"[embed] found {len(paths)} crops in {crops_dir}")
        images = []
        for p in tqdm(paths, desc="loading crops"):
            images.append(Image.open(p).convert("RGB"))

        embeddings = self.embed(images)
        return embeddings, paths


if __name__ == "__main__":
    # Quick smoke test: embed a single random crop
    import sys
    weights = sys.argv[1] if len(sys.argv) > 1 else "weights/transreid_msmt17.pth"
    model = TransReIDEmbedder(weights)

    dummy = Image.fromarray(np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8))
    emb = model.embed([dummy])
    print(f"Embedding shape: {emb.shape}")           # expect (1, 3840)
    print(f"Embedding norm:  {np.linalg.norm(emb):.4f}")  # expect ~1.0
    print("Smoke test passed.")
