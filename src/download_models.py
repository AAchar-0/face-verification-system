"""Download the two OpenCV face models into models/ and print their size and SHA-256.

Run it from the project root with the virtual environment active:
    python src/download_models.py

Models (both from the OpenCV Zoo):
    YuNet  - face detector,    MIT licence
    SFace  - face recogniser,  Apache 2.0 licence
"""
import hashlib                      # hashlib computes checksums (a fingerprint of a file)
import urllib.error
import urllib.request               # urllib downloads files: no extra package needed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "face_detection_yunet_2026may.onnx":
        f"{BASE}/face_detection_yunet/face_detection_yunet_2026may.onnx",
    "face_recognition_sface_2021dec.onnx":
        f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

# These files are stored with Git LFS. If the download goes wrong you can end up with a tiny
# text "pointer" file (about 130 bytes) instead of the model, so we reject anything this small.
MIN_BYTES = 100_000


def sha256_of(path: Path) -> str:
    """Fingerprint of a file: if even one byte changes, the fingerprint changes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:                       # "rb" = read raw bytes
        for block in iter(lambda: handle.read(1024 * 1024), b""):   # 1 MB at a time
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    for name, url in MODELS.items():
        target = MODELS_DIR / name
        if target.exists() and target.stat().st_size >= MIN_BYTES:
            print(f"Already downloaded: {name}")
        else:
            print(f"Downloading {name} ...")
            temporary = target.with_suffix(".part")        # download to a temp name first
            try:
                urllib.request.urlretrieve(url, temporary)  # follows redirects automatically
            except urllib.error.URLError as error:
                temporary.unlink(missing_ok=True)           # never leave a half-written file behind
                raise SystemExit(f"Download failed ({error.reason}). Check your internet connection and try again.")
            if temporary.stat().st_size < MIN_BYTES:
                temporary.unlink()
                raise RuntimeError(f"{name} downloaded too small: probably a Git LFS pointer, not the model")
            temporary.replace(target)                      # rename only once the download is complete

        size_mb = target.stat().st_size / (1024 * 1024)
        print(f"   size: {size_mb:.2f} MB")
        print(f"   sha256: {sha256_of(target)}")

    print(f"\nModels are in {MODELS_DIR}")
    print("Keep the sha256 values: they let you confirm later that the files were not altered.")


if __name__ == "__main__":
    main()
