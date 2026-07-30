"""Generate synthetic medical-image fixtures for container contract testing.

None of the checks in this repository require real patient data. They require
files with the right shape, the right header, and -- for the case that matters
most -- the wrong bytes.

Produces:
    valid.mha       a small volume with non-trivial spacing, origin and
                    direction, so geometry-copy bugs are detectable
    absent.mha      an all-background volume (the empty-prediction path)
    degenerate.mha  an all-zeros volume with a near-denormal maximum, mimicking
                    a real degenerate case observed in challenge validation data
    corrupt.mha     a truncated header; SimpleITK raises on read

Run:
    python -m preflight.make_fixtures --out fixtures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

SHAPE = (32, 40, 28)  # z, y, x -- small enough to be instant, large enough to be real
SPACING = (1.0, 1.0, 1.0)
ORIGIN = (-133.512, 121.364, -137.328)
DIRECTION = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)  # note the -1


def _write(array: np.ndarray, path: Path) -> None:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(SPACING)
    image.SetOrigin(ORIGIN)
    image.SetDirection(DIRECTION)
    sitk.WriteImage(image, str(path), useCompression=True)


def make_valid(path: Path, seed: int = 0) -> None:
    """A volume with a plausible intensity distribution and one bright blob."""
    rng = np.random.default_rng(seed)
    array = rng.normal(loc=300.0, scale=40.0, size=SHAPE).astype(np.float32)
    z, y, x = SHAPE
    array[z // 2 - 3 : z // 2 + 3, y // 2 - 4 : y // 2 + 4, x // 2 - 3 : x // 2 + 3] = 900.0
    _write(array, path)


def make_absent(path: Path, seed: int = 1) -> None:
    """Background only -- exercises the empty-prediction path."""
    rng = np.random.default_rng(seed)
    array = rng.normal(loc=300.0, scale=40.0, size=SHAPE).astype(np.float32)
    _write(array, path)


def make_degenerate(path: Path) -> None:
    """All zeros with a near-denormal maximum.

    Mirrors a real case found in a challenge validation set: a file that is
    structurally valid, reads without error, and contains no anatomy at all.
    Any pipeline that assumes non-empty input will divide by zero here.
    """
    array = np.zeros(SHAPE, dtype=np.float32)
    array[0, 0, 0] = np.float32(9e-41)
    _write(array, path)


def make_corrupt(path: Path, n_bytes: int = 220) -> None:
    """A truncated MetaImage header. SimpleITK.ReadImage raises on this."""
    header = (
        b"ObjectType = Image\n"
        b"NDims = 3\n"
        b"BinaryData = True\n"
        b"BinaryDataByteOrderMSB = False\n"
        b"CompressedData = True\n"
        b"TransformMatrix = 1 0 0 0 -1 0 0 0 1\n"
        b"Offset = -133.512 121.364 -137.328\n"
        b"ElementSpacing = 1 1 1\n"
        b"DimSize = 32 40 28\n"
        b"ElementType = MET_FLOAT\n"
        b"ElementDataFile = LOCAL\n"
    )
    path.write_bytes(header[:n_bytes])


def make_all(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "valid": out_dir / "valid.mha",
        "absent": out_dir / "absent.mha",
        "degenerate": out_dir / "degenerate.mha",
        "corrupt": out_dir / "corrupt.mha",
    }
    make_valid(paths["valid"])
    make_absent(paths["absent"])
    make_degenerate(paths["degenerate"])
    make_corrupt(paths["corrupt"])
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("fixtures"))
    args = parser.parse_args()

    paths = make_all(args.out)
    for name, path in paths.items():
        print(f"{name:<11} {path}  ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
