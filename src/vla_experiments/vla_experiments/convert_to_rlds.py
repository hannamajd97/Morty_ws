"""Prepare an RLDS/TFDS dataset for OpenVLA from raw Morty episodes.

OpenVLA consumes RLDS built with kpertsch/rlds_dataset_builder.  This command
copies the ready-made builder in rlds_builder/morty_ur3e/ next to your raw
data and prints the exact `tfds build` invocation.  The builder reads the raw
episodes directly (no intermediate .npy files).

    convert_to_rlds --raw ~/morty_ws/experiments --out ~/rlds/morty_ur3e

then, inside the rlds_dataset_builder conda env (tensorflow + tensorflow_datasets):

    cd ~/rlds/morty_ur3e && MORTY_VLA_RAW=<raw dir> tfds build --overwrite
"""
import argparse
import shutil
import sys
from pathlib import Path

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # running outside ROS
    get_package_share_directory = None


def builder_src() -> Path:
    here = Path(__file__).resolve().parent.parent / "rlds_builder" / "morty_ur3e"
    if here.exists():
        return here
    if get_package_share_directory:
        return Path(get_package_share_directory("vla_experiments")) / "rlds_builder" / "morty_ur3e"
    raise FileNotFoundError("rlds_builder/morty_ur3e not found")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True, help="directory that will contain the tfds builder")
    a = ap.parse_args(argv)
    raw = Path(a.raw).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for f in builder_src().iterdir():
        shutil.copy(f, out / f.name)
    print(f"builder copied to {out}\n")
    print("Build (in the rlds_dataset_builder env, see README):")
    print(f"  cd {out} && MORTY_VLA_RAW={raw} tfds build --overwrite")
    print("Result lands in ~/tensorflow_datasets/morty_ur3e/1.0.0/ ; point OpenVLA's --data_root_dir at ~/tensorflow_datasets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
