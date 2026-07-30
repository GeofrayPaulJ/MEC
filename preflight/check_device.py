"""Device selection that survives a GPU too new for the installed torch build.

`torch.cuda.is_available()` returns True when the driver is present and the
device enumerates -- even when no kernel can actually launch on it. That happens
whenever the deployment target and the development machine sit on opposite sides
of a CUDA compute-capability boundary, which is the normal condition when you
pin a base image for a datacentre GPU and develop on a consumer card.

The only reliable test is to launch a kernel.

Run standalone:
    python -m preflight.check_device        # prints the verdict
    python -m preflight.check_device --strict   # exit 1 if CUDA lies
"""

from __future__ import annotations

import argparse
import sys


def select_device(verbose: bool = True):
    """Return a torch.device that has been proven to execute a real operation.

    Falls back to CPU if CUDA reports availability but a kernel launch fails.
    """
    import torch

    if torch.cuda.is_available():
        try:
            a = torch.randn(8, 8, device="cuda")
            b = torch.randn(8, 8, device="cuda")
            _ = (a @ b).sum().item()
            if verbose:
                print(f"CUDA usable: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        except Exception as exc:  # noqa: BLE001 - we want everything
            if verbose:
                print(f"CUDA present but unusable ({exc}); falling back to CPU.")
    elif verbose:
        print("CUDA not available; using CPU.")
    return torch.device("cpu")


def report() -> dict:
    """Collect everything worth printing at the top of a container run."""
    import torch

    claimed = torch.cuda.is_available()
    device = select_device(verbose=False)
    actual = device.type == "cuda"

    info = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_claimed_available": claimed,
        "cuda_actually_usable": actual,
        "device_selected": str(device),
        "device_name": torch.cuda.get_device_name(0) if claimed else None,
        "capability": (
            "sm_%d%d" % torch.cuda.get_device_capability(0) if claimed else None
        ),
        "lying": claimed and not actual,
    }
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if CUDA reports available but no kernel can launch",
    )
    args = parser.parse_args()

    try:
        info = report()
    except ImportError:
        print("torch is not installed; nothing to probe.")
        return 0

    width = max(len(k) for k in info)
    for key, value in info.items():
        print(f"{key:<{width}} : {value}")

    if info["lying"]:
        print(
            "\nWARNING: torch.cuda.is_available() is True but kernels fail.\n"
            "         Any timing measured in this environment is a CPU timing.\n"
            f"         Device is {info['capability']}; this torch build does not "
            "support it."
        )
        return 1 if args.strict else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
