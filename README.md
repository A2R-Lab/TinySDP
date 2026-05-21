# TinySDP (Robotics: Science and Systems 2026)

Real-time semidefinite optimization for certifiable edge robotics.

TinySDP extends TinyMPC with small structured semidefinite relaxations, enabling
obstacle-aware MPC to run onboard resource-constrained robots such as the
Crazyflie 2.1 Brushless.

✅ Runs onboard at 25 Hz  
✅ Supports PSD-relaxed obstacle avoidance  
✅ Static-memory embedded implementation  
✅ Hardware demos on Crazyflie  

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

## Benchmarks

TinySDP is evaluated on obstacle-avoidance problems designed to challenge local
and reactive methods: a static U-shaped cul-de-sac, a dynamic moving-gap
benchmark, dynamic 3D sphere scenarios, and Crazyflie hardware deployment.

Across the static U-shape benchmark, TinySDP reaches the goal from all tested
starts while zero-margin local baselines crash or get stuck. Tuned local
baselines require impractically large safety margins to recover safety.

**Start: Inside**

| Method | Path length | Goal distance | Safe |
| --- | ---: | ---: | :---: |
| TinySDP (ours) | 17.95 | 0.006 | ✓ |
| RPCBF | 26.03 | 0.091 | ✓ |
| TinyMPC-LIN (m=3.1m) | 18.38 | 1.400 | ✓ |
| TinyMPC-LIN (m=0m) | — | — | ✗ |
| TinyMPC-HOCBF (m=3m) | — | — | ✗ |
| TinyMPC-HOCBF (m=0m) | — | — | ✗ |

**Start: Outside Center**

| Method | Path length | Goal distance | Safe |
| --- | ---: | ---: | :---: |
| TinySDP (ours) | 10.15 | 0.021 | ✓ |
| RPCBF | 31.03 | 0.093 | ✓ |
| TinyMPC-LIN (m=3.1m) | 24.58 | 1.400 | ✓ |
| TinyMPC-LIN (m=0m) | — | — | ✗ |
| TinyMPC-HOCBF (m=3m) | — | — | ✗ |
| TinyMPC-HOCBF (m=0m) | — | — | ✗ |

**Start: Edge Up**

| Method | Path length | Goal distance | Safe |
| --- | ---: | ---: | :---: |
| TinySDP (ours) | 9.93 | 0.023 | ✓ |
| RPCBF | 36.81 | 0.132 | ✓ |
| TinyMPC-LIN (m=3.1m) | 18.58 | 1.400 | ✓ |
| TinyMPC-LIN (m=0m) | — | — | ✗ |
| TinyMPC-HOCBF (m=3m) | — | — | ✗ |
| TinyMPC-HOCBF (m=0m) | — | — | ✗ |

**Start: Edge Down**

| Method | Path length | Goal distance | Safe |
| --- | ---: | ---: | :---: |
| TinySDP (ours) | 9.93 | 0.023 | ✓ |
| RPCBF | 36.25 | 0.108 | ✓ |
| TinyMPC-LIN (m=3.1m) | 23.64 | 1.400 | ✓ |
| TinyMPC-LIN (m=0m) | — | — | ✗ |
| TinyMPC-HOCBF (m=3m) | — | — | ✗ |
| TinyMPC-HOCBF (m=0m) | — | — | ✗ |

In the dynamic moving-gap benchmark, TinySDP maintains safety with a final goal
distance of `0.018`, while zero-margin TinyMPC-LIN and TinyMPC-HOCBF collide.
The same PSD lifting also extends to 3D sphere avoidance, including sweeping
barrier and vertical gate scenarios.

| Method | Goal distance | Safe |
| --- | ---: | :---: |
| TinySDP (ours) | 0.018 | ✓ |
| RPCBF | 0.069 | ✓ |
| TinyMPC-LIN (m=1.5m) | 0.023 | ✓ |
| TinyMPC-LIN (m=0m) | — | ✗ |
| TinyMPC-HOCBF (m=3m) | 0.077 | ✓ |
| TinyMPC-HOCBF (m=0m) | — | ✗ |

On Crazyflie hardware, TinySDP runs onboard in the control loop and maintains
the rank-1 certificate online, keeping the lifted safety margin positive during
flight.

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

## Citation

If you use our code, please cite:

```bibtex
@inproceedings{mahajan2026tinysdp,
  title={TinySDP: Real Time Semidefinite Optimization for Certifiable and Agile Edge Robotics},
  author={Ishaan Mahajan and Jon Arrizabalaga and Andrea Grillo and Fausto Vega and James Anderson and Zachary Manchester and Brian Plancher},
  booktitle={Robotics Science and Systems (RSS)},
  address = {Sydney, Australia},
  month={July},
  year = {2026}
}
```

## Maintainers

TinySDP is maintained by the Accessible and Accelerated Robotics Lab (A²R Lab).
