"""p2b environment verification.

Runs a real functional smoke test of the V1 geometry stack plus the CUDA
stack -- it does not just print version numbers. Exit code 0 means every
check passed.

Usage:
    conda run --no-capture-output -n p2b python envs/verify_env.py

Console messages are ASCII on purpose: on Windows cmd.exe the default
code page mangles non-ASCII output and makes failures unreadable.
"""
import sys
import traceback

RESULTS = {}


def check(name, fn):
    try:
        RESULTS[name] = ("OK", fn())
    except Exception as exc:  # noqa: BLE001 - report anything, never crash
        RESULTS[name] = ("FAIL", "{}: {}".format(type(exc).__name__, str(exc)[:140]))


def _numeric():
    import matplotlib
    import numpy
    import pandas
    import scipy
    import sklearn
    return "numpy {} / scipy {} / sklearn {} / pandas {} / mpl {}".format(
        numpy.__version__, scipy.__version__, sklearn.__version__,
        pandas.__version__, matplotlib.__version__,
    )


def _numba():
    import numpy as np
    from numba import njit

    @njit(cache=True)
    def sq(x):
        s = 0.0
        for i in range(x.shape[0]):
            s += x[i] * x[i]
        return s

    return "numba JIT ok, sum={:.0f}".format(sq(np.arange(1000, dtype=np.float64)))


def _open3d():
    import numpy as np
    import open3d as o3d

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(
        np.random.default_rng(0).normal(size=(2000, 3)))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
    down = pc.voxel_down_sample(0.5)
    return "open3d {}, 2000 pts -> normals + voxel -> {} pts".format(
        o3d.__version__, len(down.points))


def _plane_fit():
    """The V1 core operation: noise robustness of a planar fit.

    eigh returns eigenvalues in ASCENDING order, so the surface normal is
    the eigenvector of the SMALLEST eigenvalue (index 0). Using index 2
    here is the classic off-by-index bug that reports ~90 deg error.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    pts = np.column_stack([
        rng.uniform(-50, 50, 3000),
        rng.uniform(-50, 50, 3000),
        rng.normal(0, 0.5, 3000),      # 0.5 mm noise
    ])
    cov = np.cov((pts - pts.mean(0)).T)
    _, vecs = np.linalg.eigh(cov)
    n = vecs[:, 0]
    n = n / np.linalg.norm(n)
    err = np.degrees(np.arccos(min(1.0, abs(n[2]))))
    assert err < 1.0, "normal error too large: {:.3f} deg".format(err)
    return "0.5mm-noise plane fit, normal error = {:.4f} deg".format(err)


def _cadquery():
    import cadquery as cq

    part = (cq.Workplane("XY").box(20, 20, 20)
            .faces(">Z").workplane().hole(6)
            .edges("|Z").fillet(1.0))
    solid = part.val()
    nv = len(part.vertices().vals())
    ne = len(part.edges().vals())
    nf = len(part.faces().vals())
    assert solid.isValid(), "OCCT reports the solid as invalid"
    assert nv - ne + nf == 2, "Euler characteristic != 2 (not a closed solid)"
    return "cadquery {}, valid, V={} E={} F={} chi={} vol={:.1f} mm3".format(
        cq.__version__, nv, ne, nf, nv - ne + nf, solid.Volume())


def _step_roundtrip():
    import os
    import tempfile

    import cadquery as cq

    path = os.path.join(tempfile.gettempdir(), "p2b_verify.step")
    cq.exporters.export(
        cq.Workplane("XY").box(10, 10, 10).faces(">Z").workplane().hole(3),
        path,
    )
    back = cq.importers.importStep(path)
    assert back.val().isValid(), "re-imported STEP is invalid"
    return "STEP write+read {}, {} bytes".format(
        "ok", os.path.getsize(path))


def _torch():
    import torch
    return "torch {} (cuda build {}, cudnn {})".format(
        torch.__version__, torch.version.cuda, torch.backends.cudnn.version())


def _torch_gpu():
    import torch

    if not torch.cuda.is_available():
        return "no CUDA device visible (CPU-only build or no driver)"
    cap = torch.cuda.get_device_capability(0)
    arch = torch.cuda.get_arch_list()
    if "sm_{}{}".format(cap[0], cap[1]) not in arch:
        return ("WARNING: device sm_{}{} not in arch_list {} -- kernels will "
                "not run".format(cap[0], cap[1], arch))
    x = torch.randn(3000, 3000, device="cuda")
    total = float((x @ x).sum())
    w = torch.randn(512, 512, device="cuda", requires_grad=True)
    w.pow(2).mean().backward()
    assert w.grad is not None, "backward() produced no gradient"
    return "{} sm_{}{} | matmul={:.0f} | backward ok".format(
        torch.cuda.get_device_name(0), cap[0], cap[1], total)


def _torchvision():
    import torchvision
    from torchvision import transforms

    transforms.Resize(8)
    return "torchvision {}, transforms ok".format(torchvision.__version__)


def main():
    print("python {} @ {}".format(sys.version.split()[0], sys.executable))
    print("=" * 70)
    for name, fn in [
        ("numeric", _numeric),
        ("numba", _numba),
        ("open3d", _open3d),
        ("plane-fit", _plane_fit),
        ("cadquery/occt", _cadquery),
        ("step-roundtrip", _step_roundtrip),
        ("torch", _torch),
        ("torch-gpu", _torch_gpu),
        ("torchvision", _torchvision),
    ]:
        check(name, fn)

    print()
    for name, (status, detail) in RESULTS.items():
        print("[{:<4}] {:<15} {}".format(status, name, detail))

    failed = [n for n, (s, _) in RESULTS.items() if s != "OK"]
    print("\n" + "=" * 70)
    if failed:
        print("FAILED: {}".format(", ".join(failed)))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
