# challenge-preflight

Preflight checks for medical imaging challenge submission containers.

Companion repository to **[Your GPU Is Lying To You](https://youtu.be/8Epz6I85QDY)** — MICCAI Educational Challenge 2026.

Every check here is derived from a specific failure encountered while building and
submitting containers to the AIMS-TBI challenge (MICCAI 2026). None of them require
real patient data, a GPU, model weights, or a data use agreement.

---

## What it checks

| Check | The failure it catches |
|---|---|
| `check_device.py` | `torch.cuda.is_available()` returns `True` on a GPU too new for the installed torch build. Every kernel fails, the run silently falls back to CPU, and every timing you measure afterwards is meaningless. |
| `make_fixtures.py` | Produces four synthetic `.mha` volumes: valid, absent, degenerate (all-zeros, near-denormal max), and corrupt (220-byte truncated header). |
| `test_contract.py` | Asserts your container exits 0 and writes a well-formed output on **all four** inputs, including the corrupt one. Also audits output geometry against input, and mask values against `{0, 1}`. |

The geometry check exists because starter-kit examples frequently build the output
mask with `GetImageFromArray()` and never copy the input header. That bug is
**invisible** if your test volume has identity spacing, origin and direction — so
the fixtures here deliberately do not.

---

## Install and run

```bash
git clone https://github.com/<you>/challenge-preflight
cd challenge-preflight
pip install -r requirements.txt

python -m preflight.make_fixtures --out fixtures/
python -m preflight.check_device
pytest -v tests/
```

Sixty seconds, no downloads, no account.

Without `CONTAINER_CMD` set, the container tests skip and the fixture, geometry and
device tests run. That is the CI configuration.

## Testing a real container

```bash
export CONTAINER_CMD="docker run --rm --network none \
    -v {input}:/input/images/t1-brain-mri/ \
    -v {output}:/output/ my-algorithm"

pytest -v tests/
```

`{input}` and `{output}` are substituted with temporary host directories, one per
fixture. Adjust the mount paths to your challenge's I/O contract.

`--network none` is not optional. Inference containers are commonly run with the
network severed; anything your code fetches at runtime must already be inside the
image. Test that assumption before the platform tests it for you.

**Note:** the build is not offline even though the runtime is. `pip install` at
build time needs an index. Those two facts get conflated constantly.

---

## The device probe, standalone

```
$ python -m preflight.check_device
torch                   : 2.5.1+cu124
cuda_build              : 12.4
cuda_claimed_available  : True
cuda_actually_usable    : False
device_selected         : cpu
device_name             : [GPU model redacted]
capability              : sm_120
lying                   : True

WARNING: torch.cuda.is_available() is True but kernels fail.
         Any timing measured in this environment is a CPU timing.
         Device is sm_120; this torch build does not support it.
```

Use `--strict` to exit non-zero, which is what you want in a CI job that gates a
timing benchmark.

---

## Why synthetic data

Challenge data, histopathology and clinical MRI all arrive under agreements that
forbid redistribution. Tutorials depending on that data are unrunnable by their
readers, and the usual workaround — gesturing at a public dataset that resembles
the private one — imports a second set of assumptions to cover the first.

Real data is required to measure whether your model is *good*. It is not required
to verify that your container is *honest*. These are separate problems. This
repository addresses only the second, which is the one that can be shared.

---

## Licence

MIT.
