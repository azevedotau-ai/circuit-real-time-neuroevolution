# 🏎️ Neural Circuit — AI Learning to Drive

> Neuroevolution simulation where AI-controlled cars teach themselves to drive a circuit using a genetic algorithm, no backpropagation, no labelled data, just evolution.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📖 Overview

Neural Circuit is a real-time neuroevolution sandbox built entirely in Python. A population of cars, each controlled by a small feedforward neural network, starts with random weights and gradually evolves to complete laps around a custom circuit. There is no training dataset and no gradient descent — fitness is determined purely by how far each car gets, how fast it goes, and whether it stays on track.

The simulation comes with a **live dashboard** (Matplotlib), an **interactive track editor**, and a fully configurable **JSON-based config system**.

---

## ✨ Features

- **Neuroevolution** — Genetic algorithm with elitism, crossover, adaptive mutation rate, and stagnation detection
- **Real-time dashboard** — Live track view, learning curve, neural network visualisation, sensor radar, AI commands panel, and event log
- **Car sprites** — Top-down car shapes with windscreen, wheels, and glow effects for the best car
- **Ghost trail** — The leading car leaves a fading trail showing its recent path
- **Sensor system** — 7 distance sensors at ±90°/±45°/±22.5°/0° plus speed, feeding directly into the network
- **Interactive track editor** — Draw custom circuits with Catmull-Rom splines, set start/finish lines, adjust track width
- **Responsive UI** — Layout and font sizes adapt automatically to the screen resolution
- **Fully configurable** — All physics, rewards, penalties, and population parameters live in `config.json`

---

## 🧠 Architecture

```
Inputs (8)              Hidden (16)           Outputs (2)
────────────────        ──────────────        ──────────────
Sensor  −90°   ──┐                            Steer  [-1, +1]
Sensor  −45°   ──┤
Sensor −22.5°  ──┤   tanh activation    ──►   Throttle [0, 1]
Sensor    0°   ──┼──► 16 neurons        ──►   (≥0.5 = gas,
Sensor +22.5°  ──┤   sigmoid output           <0.5 = brake)
Sensor  +45°   ──┤
Sensor  +90°   ──┘
Speed (norm.)  ───
```

**Genetic operators:**
| Operator | Description |
|---|---|
| Elitism | Top N survivors copied unchanged |
| Crossover | Uniform gene-level mixing between two parents |
| Mutation | Per-gene Gaussian noise, rate adapts to stagnation |
| Random injection | Fresh random brains added each generation |

**Adaptive mutation schedule:**

| Stagnation | Rate | Mode       |
| ---------- | ---- | ---------- |
| < 10 gens  | 40%  | `explore`  |
| 10–30 gens | 28%  | `normal`   |
| 30+ gens   | 28%  | `shake`    |
| 30–60 gens | 38%  | `plateau`  |
| 60+ gens   | 50%  | `CRITICAL` |

---

## 🗂️ Project Structure

```
neural-network-circuit-game/
│
├── app.py                  # Entry point
├── config.json             # All simulation parameters
├── track.json              # Circuit definition (control points, start/finish)
│
├── src/
│   ├── track.py            # Circuit geometry, SDF collision, car physics
│   ├── simulation.py       # Genetic algorithm training loop
│   ├── neural_network.py   # Feedforward network + genetic operators
│   └── visualization.py   # Real-time Matplotlib dashboard
│
└── editor_track.py         # Interactive track editor
```

---

## 🚀 Getting Started

### Requirements

```
Python 3.11+
numpy
matplotlib
```

Install dependencies:

```bash
pip install numpy matplotlib
```

### Run the simulation

```bash
python app.py
```

### Run the track editor

```bash
python editor_track.py
```

---

## ⌨️ Controls

| Key        | Action                    |
| ---------- | ------------------------- |
| `Q`        | Quit simulation           |
| `SPACE`    | Toggle sensor rays on/off |
| `?` button | Show/hide help overlay    |

---

## ⚙️ Configuration

All parameters are in `config.json`. Key sections:

```jsonc
{
  "simulation": {
    "population": 80, // Number of cars per generation
    "target_laps": 10, // Laps required to win
    "top_survivors": 10, // Elite cars carried to next gen
    "new_random_per_generation": 10,
  },
  "cars": {
    "max_speed": 1.0,
    "speed_step": 0.025,
    "max_turn_angle": 12, // Degrees per frame
  },
  "rewards": {
    "lap_reward": 1000,
    "circuit_progress_weight": 15,
    "speed_weight": 8,
  },
  "penalties": {
    "collision_penalty": 800,
    "wrong_way_penalty": 12,
    "near_wall_penalty": 4,
  },
}
```

---

## 🗺️ Track Editor

The editor lets you design circuits interactively:

- **Left click** — Place a control point
- **Right click** — Remove the last point
- **Drag** — Move existing control points
- Adjust track width with the slider
- **Save** — Exports `track.json` ready for the simulation

The circuit is interpolated using **Catmull-Rom splines** for smooth curves from a small number of control points.

---

## 📊 How the Dashboard Works

| Panel                | Content                                                          |
| -------------------- | ---------------------------------------------------------------- |
| **Track**            | Live car positions, ghost trail, sensor rays, start/finish lines |
| **Best Alive**       | Lap progress gauge, score gauge, lap timings, speed bar          |
| **Sensor Radar**     | Polar plot of the leading car's 7 distance sensors               |
| **Population Stats** | Alive/dead bar, fitness history chart, mutation mode badge       |
| **Learning Curve**   | Best and average fitness per generation                          |
| **Simulation**       | Architecture info, training parameters, runtime stats            |
| **AI Commands**      | Current steering, throttle, and speed of the leading car         |
| **Events**           | Recent reward/penalty events for the leading car                 |
| **Neural Network**   | Live weight visualisation of the winning brain                   |

---

## 🏗️ How It Works — Step by Step

1. **Spawn** — 80 cars start at the beginning of the track with random neural network weights
2. **Sense** — Each car fires 7 distance sensors and reads its own speed (8 inputs total)
3. **Decide** — The network produces a steering angle and throttle value
4. **Move** — Physics updates position; the car dies if it goes off-track or stops making progress
5. **Score** — Fitness accumulates from lap completions, speed bonuses, and centreline progress
6. **Evolve** — When all cars are dead, the top performers breed the next generation
7. **Repeat** — Each generation the population gets slightly better at navigating the circuit

---

## 📁 Output

When a car completes the target number of laps, the simulation saves `results.json` with:

- Timestamp, generation number, lap record
- Winner car score
- Full neural network weights (W1, b1, W2, b2)
- Fitness history arrays
- Config snapshot

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

Built with [NumPy](https://numpy.org) and [Matplotlib](https://matplotlib.org).  
Circuit geometry uses Catmull-Rom spline interpolation and a pre-computed Signed Distance Field (SDF) for O(1) collision queries.
