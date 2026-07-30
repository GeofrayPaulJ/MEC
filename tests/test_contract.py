"""Contract tests for a challenge submission container.

Two layers:

1. Tests that always run. They verify the fixtures, the geometry audit and the
   device probe. No Docker, no GPU, no weights, no data agreement.

2. Tests that run only when CONTAINER_CMD is set. These invoke a real container
   against each fixture and assert the contract that matters: exit 0 and a
   well-formed output, on every input including the corrupt one.

       export CONTAINER_CMD="docker run --rm --network none \
           -v {input}:/input/images/t1-brain-mri/ \
           -v {output}:/output/ my-algorithm"
       pytest -v

   {input} and {output} are substituted with host directories.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from preflight import make_fixtures

FIXTURE_NAMES = ["valid", "absent", "degenerate", "corrupt"]
CONTAINER_CMD = os.environ.get("CONTAINER_CMD")
TIMEOUT_S = int(os.environ.get("CONTAINER_TIMEOUT_S", "600"))


@pytest.fixture(scope="session")
def fixtures(tmp_path_factory) -> dict[str, Path]:
    return make_fixtures.make_all(tmp_path_factory.mktemp("fixtures"))


# --------------------------------------------------------------------------
# Layer 1: always runs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["valid", "absent", "degenerate"])
def test_readable_fixtures_read(fixtures, name):
    """The three valid fixtures must load without error."""
    image = sitk.ReadImage(str(fixtures[name]))
    assert image.GetSize() == (28, 40, 32)


def test_corrupt_fixture_actually_raises(fixtures):
    """The corrupt fixture is only useful if it genuinely fails to read."""
    with pytest.raises(RuntimeError):
        sitk.ReadImage(str(fixtures["corrupt"]))


def test_fixtures_carry_non_identity_geometry(fixtures):
    """A geometry-copy bug is invisible against identity direction cosines.

    The stock example in many starter kits builds output with
    GetImageFromArray() and never copies the input geometry. If your fixture
    has identity spacing, origin and direction, that bug passes every test.
    """
    image = sitk.ReadImage(str(fixtures["valid"]))
    assert image.GetDirection() != (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    assert image.GetOrigin() != (0.0, 0.0, 0.0)


def test_degenerate_fixture_is_degenerate(fixtures):
    """Guards against a pipeline that assumes non-empty input."""
    array = sitk.GetArrayFromImage(sitk.ReadImage(str(fixtures["degenerate"])))
    assert float(array.max()) < 1e-30


def test_device_probe_agrees_with_itself():
    """The probe must not claim CUDA it cannot use."""
    torch = pytest.importorskip("torch")
    from preflight.check_device import report

    info = report()
    if info["device_selected"] == "cuda":
        # If we selected CUDA, a kernel must launch.
        _ = (torch.randn(4, 4, device="cuda") @ torch.randn(4, 4, device="cuda")).sum()
    if info["lying"]:
        pytest.skip(
            "CUDA reports available but kernels fail -- probe correctly fell back. "
            "Any timing on this machine is a CPU timing."
        )


# --------------------------------------------------------------------------
# Layer 2: runs only against a real container
# --------------------------------------------------------------------------

requires_container = pytest.mark.skipif(
    CONTAINER_CMD is None, reason="CONTAINER_CMD not set"
)


def _run_container(input_dir: Path, output_dir: Path) -> subprocess.CompletedProcess:
    cmd = CONTAINER_CMD.format(input=input_dir.as_posix(), output=output_dir.as_posix())
    return subprocess.run(
        shlex.split(cmd),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )


@requires_container
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_container_exits_zero_on_every_input(fixtures, tmp_path, name):
    """The contract. Exit 0 on all four inputs -- including the corrupt one.

    A crash is not a low score. On a scoring board it is the worst possible
    rank for that case, and on some platforms it fails the entire run.
    """
    input_dir = tmp_path / f"in_{name}"
    output_dir = tmp_path / f"out_{name}"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "case.mha").write_bytes(fixtures[name].read_bytes())

    result = _run_container(input_dir, output_dir)

    assert result.returncode == 0, (
        f"container exited {result.returncode} on '{name}' input\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    produced = list(output_dir.rglob("*"))
    assert [p for p in produced if p.is_file()], (
        f"container exited 0 on '{name}' but wrote no output file. "
        "Exit code alone is not the contract."
    )


@requires_container
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_container_output_is_well_formed(fixtures, tmp_path, name):
    """Dump the artefact, then parse it. In that order."""
    input_dir = tmp_path / f"in2_{name}"
    output_dir = tmp_path / f"out2_{name}"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "case.mha").write_bytes(fixtures[name].read_bytes())

    _run_container(input_dir, output_dir)

    for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
        raw = path.read_bytes()
        print(f"{path.name}: {len(raw)} bytes, head={raw[:32].hex(' ')}")

        if path.suffix == ".json":
            json.loads(raw)  # raises on malformed output
        elif path.suffix in {".mha", ".mhd", ".nii", ".gz"}:
            mask = sitk.ReadImage(str(path))
            array = sitk.GetArrayFromImage(mask)
            assert set(np.unique(array)).issubset({0, 1}), (
                f"mask for '{name}' contains values outside {{0, 1}}: "
                f"{np.unique(array)[:10]}"
            )
            if name != "corrupt":
                source = sitk.ReadImage(str(input_dir / "case.mha"))
                assert mask.GetSpacing() == pytest.approx(source.GetSpacing())
                assert mask.GetOrigin() == pytest.approx(source.GetOrigin())
                assert mask.GetDirection() == pytest.approx(source.GetDirection()), (
                    "output geometry does not match input -- the container built "
                    "the mask from an array and never copied the header."
                )
