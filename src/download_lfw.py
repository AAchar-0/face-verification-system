"""Download the Labeled Faces in the Wild (LFW) benchmark for LOCAL evaluation only.

Run it from the project root with the virtual environment active:
    python src/download_lfw.py

Ethics: LFW is web-collected photos of public figures. It is used here only to measure
accuracy on your own computer. Never commit it, share it or publish its images.
It is saved under data/, which .gitignore keeps out of Git.

The original host (vis-www.cs.umass.edu) no longer resolves for some networks, so this uses the
mirror that the scikit-learn library uses. Each file is checked against the SHA-256 checksum
recorded in scikit-learn's source code, so a damaged or altered download is rejected.
"""
import hashlib                      # computes the SHA-256 fingerprint of a file
import tarfile                      # reads .tgz archives (a compressed bundle of files)
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# file name -> (web address, expected SHA-256 fingerprint)
FILES = {
    "lfw.tgz": (
        "https://ndownloader.figshare.com/files/5976018",
        "055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0",
    ),
    "pairs.txt": (
        "https://ndownloader.figshare.com/files/5976006",
        "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592",
    ),
}


def sha256_of(path: Path) -> str:
    """Fingerprint of a file: if even one byte changes, the fingerprint changes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):    # 1 MB at a time
            digest.update(block)
    return digest.hexdigest()


def show_progress(block_number: int, block_size: int, total_size: int) -> None:
    """Print progress about every 10 percent (urllib calls this after each block)."""
    if total_size <= 0:
        return
    percent = min(100, block_number * block_size * 100 // total_size)
    if percent % 10 == 0 and percent != show_progress.last:
        show_progress.last = percent
        print(f"   {percent}%", flush=True)


show_progress.last = -1                                              # remembers the last percent printed


def fetch(name: str, url: str, expected: str) -> Path:
    """Download one file (unless already present and correct) and verify its checksum."""
    target = RAW_DIR / name
    if target.exists() and sha256_of(target) == expected:
        print(f"Already downloaded and verified: {name}")
        return target

    print(f"Downloading {name} ...")
    temporary = target.with_suffix(target.suffix + ".part")           # download to a temporary name first
    show_progress.last = -1
    try:
        urllib.request.urlretrieve(url, temporary, show_progress)
    except urllib.error.URLError as error:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Download failed ({error.reason}). Check your internet connection and try again.")

    actual = sha256_of(temporary)
    if actual != expected:
        temporary.unlink()                                            # never keep a file that fails the check
        raise SystemExit(f"Checksum mismatch for {name}.\n  expected {expected}\n  got      {actual}\n"
                         "The file was deleted. Try again, and tell me if it keeps happening.")
    temporary.replace(target)
    print(f"   checksum OK ({target.stat().st_size / (1024 * 1024):.0f} MB)")
    return target


def extract(archive: Path) -> None:
    """Unpack the images into data/raw/lfw/<person>/<person>_0001.jpg."""
    destination = RAW_DIR / "lfw"
    if destination.exists() and any(destination.iterdir()):
        print("Images already extracted.")
        return
    print("Extracting images (this takes a minute) ...")
    with tarfile.open(archive, "r:gz") as bundle:
        # filter="data" blocks unsafe paths inside the archive (a standard safety measure).
        bundle.extractall(RAW_DIR, filter="data")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {name: fetch(name, url, checksum) for name, (url, checksum) in FILES.items()}
    extract(paths["lfw.tgz"])

    images = list((RAW_DIR / "lfw").glob("*/*.jpg"))
    people = {image.parent.name for image in images}
    print(f"\nImages: {len(images):,} | People: {len(people):,}  (expected 13,233 images of 5,749 people)")
    print(f"Saved in {RAW_DIR}. This folder is ignored by Git: never commit or share it.")


if __name__ == "__main__":
    main()
