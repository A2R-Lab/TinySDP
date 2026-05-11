# TinySDP

Real-time semidefinite optimization for certifiable edge robotics.

TinySDP extends TinyMPC with small structured semidefinite relaxations, enabling
obstacle-aware MPC to run onboard resource-constrained robots such as the
Crazyflie 2.1 Brushless.

✅ Runs onboard at 25 Hz  
✅ Supports PSD-relaxed obstacle avoidance  
✅ Static-memory embedded implementation  
✅ Hardware demos on Crazyflie  
✅ Accepted to RSS 2026

![TinySDP Crazyflie demo](assets/tinysdp-demo.gif)

## Why TinySDP?

Robots running on small processors usually cannot afford generic semidefinite
programming. TinySDP keeps the optimization structure small and fixed, so
certifiable obstacle-aware control can run in real time without dynamic memory
allocation or desktop-class hardware.

This repository contains the reusable C++ solver in `include/solver` and
`src/solver`, plus three standalone TinySDP examples:

- `ushape_demo`: closed-loop 2D U-shape obstacle avoidance with rank-1 certificate logging
- `sweeping_gate_3d_demo`: 3D sweeping gate
- `rising_gate_3d_demo`: 3D rising gate

## Dependencies

- CMake 3.15 or newer
- A C++17 compiler
- Eigen 3
- Matplotlib for plotting example outputs

On macOS with Homebrew:

```sh
brew install eigen cmake
python -m pip install matplotlib
```

## Quick Start

```sh
cmake -S . -B build
cmake --build build
```

Run the demos:

```sh
./build/examples/ushape_demo
./build/examples/sweeping_gate_3d_demo
./build/examples/rising_gate_3d_demo
```

Example CSV outputs are written to `outputs/` by default. Set
`TINYSDP_OUTPUT_DIR=/path/to/output` to write them elsewhere.

The U-shape demo defaults to the inside-the-cul-de-sac start. Other starts are
available with:

```sh
./build/examples/ushape_demo --start edge_up
./build/examples/ushape_demo --start edge_down
./build/examples/ushape_demo --start outside_center
./build/examples/ushape_demo --start above
./build/examples/ushape_demo --start below
```

Plot any generated trajectory CSVs:

```sh
python scripts/plot_examples.py
```

The plotting script scans `outputs/` and writes trajectory PNGs to `outputs/plots/`.
Use `--input-dir` or `--output-dir` to override those paths.

## Project Layout

- `include/solver`: public solver headers
- `src/solver`: solver implementation
- `examples`: TinySDP-only demos
- `scripts`: plotting and utility scripts
- `assets`: README media

## Getting Help

Open a GitHub issue for bugs, build problems, or questions about the examples.

## Maintainers

TinySDP is maintained by the Accessible and Accelerated Robotics Lab (A²R Lab).
