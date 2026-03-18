import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download, hf_hub_download
import h5py


def parse_args():
    parser = argparse.ArgumentParser(description="Download LIBERO datasets from Hugging Face Hub")
    parser.add_argument(
        "--download-dir",
        type=str,
        default=os.path.expanduser("~/.libero/datasets"),
        help="Directory to store downloaded datasets (default: ~/.libero/datasets)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        choices=["all", "libero_goal", "libero_spatial", "libero_object", "libero_100"],
        default="all",
        help="Which LIBERO subset(s) to download",
    )
    return parser.parse_args()


def get_dataset_patterns(dataset_name):
    """Return the file patterns for a given LIBERO subset on HF Hub."""
    mapping = {
        "libero_goal": ["libero_goal/**"],
        "libero_spatial": ["libero_spatial/**"],
        "libero_object": ["libero_object/**"],
        "libero_100": ["libero_100/**"],
    }
    if dataset_name == "all":
        patterns = []
        for v in mapping.values():
            patterns.extend(v)
        return patterns
    else:
        return mapping[dataset_name]


def check_hdf5_files(dataset_dir: Path):
    """Basic integrity check: ensure .hdf5 files are readable."""
    hdf5_files = list(dataset_dir.rglob("*.hdf5"))
    if not hdf5_files:
        print("⚠️  No .hdf5 files found. Data may be incomplete.")
        return False

    for f in hdf5_files[:3]:  # check first 3 files as sample
        try:
            with h5py.File(f, "r") as _:
                pass
        except Exception as e:
            print(f"❌ Corrupted or unreadable HDF5 file: {f} ({e})")
            return False
    print(f"✅ Verified {len(hdf5_files)} HDF5 files (sample-checked).")
    return True


def main():
    args = parse_args()
    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"📥 Datasets will be downloaded to: {download_dir}")
    print(f"📋 Requested dataset(s): {args.datasets}")

    # Determine which patterns to download
    patterns = get_dataset_patterns(args.datasets)
    print(f"🔍 Downloading files matching: {patterns}")

    # Download from Hugging Face Hub
    try:
        snapshot_download(
            repo_id="physical-intelligence/libero",
            repo_type="dataset",
            allow_patterns=patterns,
            local_dir=download_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=8,
        )
        print("✅ Download completed successfully.")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return

    # Basic integrity check
    print("🔍 Verifying dataset integrity...")
    if check_hdf5_files(download_dir):
        print("🎉 LIBERO dataset is ready for use!")
    else:
        print("⚠️  Warning: Dataset may be incomplete or corrupted.")


if __name__ == "__main__":
    main()