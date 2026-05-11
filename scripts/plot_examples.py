#!/usr/bin/env python3
import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
mpl_config_dir = Path(tempfile.gettempdir()) / "tinysdp-matplotlib"
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(mpl_config_dir / "cache"))

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "Matplotlib is required for plotting. Install it with "
        "`python -m pip install matplotlib`, then rerun this script."
    ) from exc


U_SHAPE_DISKS = [
    (2.5, 0.0, 0.8),
    (2.5, 1.2, 0.8),
    (2.5, -1.2, 0.8),
    (3.8, 1.2, 0.8),
    (3.8, -1.2, 0.8),
    (5.0, 1.2, 0.8),
    (5.0, -1.2, 0.8),
]


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row, key, default=0.0):
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def set_equal_2d(ax, xs, ys, pad=0.6):
    if not xs or not ys:
        return
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    span = max(xmax - xmin, ymax - ymin, 1.0) + 2.0 * pad
    ax.set_xlim(cx - 0.5 * span, cx + 0.5 * span)
    ax.set_ylim(cy - 0.5 * span, cy + 0.5 * span)
    ax.set_aspect("equal", adjustable="box")


def set_equal_3d(ax, xs, ys, zs, pad=0.4):
    if not xs or not ys or not zs:
        return
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    span = max(xmax - xmin, ymax - ymin, zmax - zmin, 1.0) + 2.0 * pad
    ax.set_xlim(cx - 0.5 * span, cx + 0.5 * span)
    ax.set_ylim(cy - 0.5 * span, cy + 0.5 * span)
    ax.set_zlim(max(0.0, cz - 0.5 * span), cz + 0.5 * span)


def plot_ushape_tracking(path, out_dir):
    rows = read_csv(path)
    if not rows:
        return None

    xs = [as_float(r, "x1") for r in rows]
    ys = [as_float(r, "x2") for r in rows]
    signed = [as_float(r, "seg_signed_dist", as_float(r, "signed_dist")) for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 5.6), dpi=160)
    for i, (cx, cy, r) in enumerate(U_SHAPE_DISKS):
        disk = plt.Circle((cx, cy), r, color="#d95f02", alpha=0.22, ec="#8c2d04", lw=1.0)
        ax.add_patch(disk)
        ax.text(cx, cy, str(i), ha="center", va="center", fontsize=7, color="#8c2d04")

    ax.plot(xs, ys, color="#1f77b4", lw=2.2, marker="o", ms=3.5, label="closed-loop trajectory")
    ax.scatter([xs[0]], [ys[0]], s=70, color="#2ca02c", label="start", zorder=5)
    ax.scatter([0.0], [0.0], s=80, color="#111111", marker="*", label="goal", zorder=5)
    ax.set_title(f"U-shape closed-loop trajectory ({path.stem.replace('tinysdp_ushape_tracking_', '')})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    all_xs = xs + [d[0] - d[2] for d in U_SHAPE_DISKS] + [d[0] + d[2] for d in U_SHAPE_DISKS] + [0.0]
    all_ys = ys + [d[1] - d[2] for d in U_SHAPE_DISKS] + [d[1] + d[2] for d in U_SHAPE_DISKS] + [0.0]
    set_equal_2d(ax, all_xs, all_ys)

    min_sd = min(signed) if signed else math.nan
    ax.text(0.02, 0.02, f"min segment clearance: {min_sd:.3g}", transform=ax.transAxes,
            fontsize=9, bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85})

    out_path = out_dir / f"{path.stem}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def read_spheres_by_step(path):
    rows = read_csv(path)
    spheres = {}
    for row in rows:
        k = int(float(row["k"]))
        spheres.setdefault(k, []).append((
            as_float(row, "cx"),
            as_float(row, "cy"),
            as_float(row, "cz"),
            as_float(row, "r"),
        ))
    return spheres


def plot_3d_tracking(path, sphere_path, out_dir):
    rows = read_csv(path)
    if not rows:
        return None

    xs = [as_float(r, "x") for r in rows]
    ys = [as_float(r, "y") for r in rows]
    zs = [as_float(r, "z") for r in rows]
    signed = [as_float(r, "seg_signed_dist", as_float(r, "signed_dist")) for r in rows]
    spheres = read_spheres_by_step(sphere_path) if sphere_path and sphere_path.exists() else {}

    fig = plt.figure(figsize=(7.0, 5.8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, color="#1f77b4", lw=2.2, marker="o", ms=3.0, label="trajectory")
    ax.scatter([xs[0]], [ys[0]], [zs[0]], s=60, color="#2ca02c", label="start")
    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], s=70, color="#111111", marker="*", label="final")

    for k in sorted(spheres):
        alpha = 0.10 if k not in (0, max(spheres)) else 0.35
        for cx, cy, cz, r in spheres[k]:
            ax.scatter([cx], [cy], [cz], s=max(20.0, 220.0 * r), color="#d95f02", alpha=alpha)

    label = path.stem.replace("tinysdp_3d_", "").replace("_tracking", "").replace("_", " ")
    ax.set_title(f"3D {label} trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="best")
    set_equal_3d(ax, xs, ys, zs)

    min_sd = min(signed) if signed else math.nan
    ax.text2D(0.02, 0.02, f"min segment clearance: {min_sd:.3g}", transform=ax.transAxes,
              fontsize=9, bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85})

    out_path = out_dir / f"{path.stem}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_all(input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for path in sorted(input_dir.glob("tinysdp_ushape_tracking_*.csv")):
        out = plot_ushape_tracking(path, output_dir)
        if out:
            written.append(out)

    for path in sorted(input_dir.glob("tinysdp_3d_*_tracking.csv")):
        prefix = path.name.removesuffix("_tracking.csv")
        sphere_path = input_dir / f"{prefix}_spheres.csv"
        out = plot_3d_tracking(path, sphere_path, output_dir)
        if out:
            written.append(out)

    return written


def main():
    parser = argparse.ArgumentParser(description="Plot TinySDP example CSV outputs.")
    parser.add_argument("--input-dir", default="outputs", type=Path,
                        help="Directory containing TinySDP CSV outputs.")
    parser.add_argument("--output-dir", default=None, type=Path,
                        help="Directory for PNG plots. Defaults to <input-dir>/plots.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir if args.output_dir else input_dir / "plots"
    if not input_dir.exists():
        raise SystemExit(f"input directory does not exist: {input_dir}")

    written = plot_all(input_dir, output_dir)
    if not written:
        print(f"No supported TinySDP CSV files found in {input_dir}")
        return

    print("Wrote plots:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
