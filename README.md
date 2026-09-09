# TinySDP (Robotics: Science and Systems 2026)

<p align="center">
  <a href="https://ishaanmahajan.com">Ishaan Mahajan</a><sup>1</sup>,
  <a href="https://jonarriza96.github.io/">Jon Arrizabalaga</a><sup>2,&#8225;</sup>,
  <a href="https://grilloandrea6.github.io/">Andrea Grillo</a><sup>3,4,&#8225;</sup>,
  <a href="https://www.linkedin.com/in/fausto-vega/">Fausto Vega</a><sup>2,&#8225;</sup>,
  <a href="https://www.columbia.edu/~ja3451/index.html">James Anderson</a><sup>1,&#8224;</sup>,
  <a href="https://www.linkedin.com/in/zacmanchester/">Zachary Manchester</a><sup>2,&#8224;</sup>,
  and <a href="https://brianplancher.com/">Brian Plancher</a><sup>3</sup>
</p>

<p align="center">
  <sup>1</sup>Columbia University &nbsp;&nbsp;
  <sup>2</sup>Massachusetts Institute of Technology &nbsp;&nbsp;
  <sup>3</sup>Dartmouth College &nbsp;&nbsp;
  <sup>4</sup>EPFL
  <br>
  <sup>&#8225;</sup>Equal contribution &nbsp;&nbsp; <sup>&#8224;</sup>Equal advising
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.13748"><img src="https://img.shields.io/badge/arXiv-2605.13748-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://a2r-lab.org/TinySDP/"><img src="https://img.shields.io/badge/Project-Website-4c71f0?logo=googlechrome&logoColor=white" alt="Project Website"></a>
  <a href="https://youtu.be/iYQyT-WK-X8?si=s_7gRyH7ZHOE82T_"><img src="https://img.shields.io/badge/YouTube-Video-ff0000?logo=youtube&logoColor=white" alt="YouTube Video"></a>
</p>

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

In the dynamic moving-gap benchmark, TinySDP maintains safety with a final goal
distance of `0.018`, while zero-margin TinyMPC-LIN and TinyMPC-HOCBF collide.

| Method | Goal distance | Safe |
| --- | ---: | :---: |
| TinySDP (ours) | 0.018 | ✓ |
| RPCBF | 0.069 | ✓ |
| TinyMPC-LIN (m=1.5m) | 0.023 | ✓ |
| TinyMPC-LIN (m=0m) | — | ✗ |
| TinyMPC-HOCBF (m=3m) | 0.077 | ✓ |
| TinyMPC-HOCBF (m=0m) | — | ✗ |

For more information, visit the [project page](https://a2r-lab.org/TinySDP/).

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

## Funding Acknowledgement

This work was supported by the National Science Foundation (under Awards
[2144634](https://www.nsf.gov/awardsearch/show-award/?AWD_ID=2144634),
[2231350](https://www.nsf.gov/awardsearch/show-award/?AWD_ID=2231350), and
[2411369](https://www.nsf.gov/awardsearch/show-award/?AWD_ID=2411369)) and by the
Center of AI Technology (CAIT) in collaboration with Amazon. Any opinions,
findings, conclusions, or recommendations expressed in this material are those
of the authors and do not necessarily reflect those of the funding
organizations.
