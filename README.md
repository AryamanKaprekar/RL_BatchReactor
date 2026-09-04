# RL Batch Reactor Temperature Control

Reinforcement learning controller for regulating the temperature of a batch
reactor. A custom Gym environment simulates the reactor's polymerization
kinetics and thermal dynamics, and a dual-output Policy Mirror Descent (PMD)
controller with Temporal Difference (TD) learning learns to track a
time-varying temperature setpoint by manipulating coolant flow and heater
current.

Main notebook: [`Miso_PMD+TD.ipynb`](Miso_PMD+TD.ipynb)

## The environment (`BR3`)

`BR3` is a `gym.Env` wrapping an ODE model (`br`) of an exothermic batch
polymerization reactor, integrated with `scipy.integrate.odeint` at each
step.

**State (`x`)** — 4 physical variables tracked by the ODE, of which 2 are
exposed to the agent:

| Variable | Meaning |
|---|---|
| `Ii` | Initiator concentration |
| `M` | Monomer concentration |
| `Tr` | Reactor temperature (°C) — **observed, controlled variable** |
| `Tj` | Jacket (cooling/heating) temperature (°C) |

**Observation** given to the agent is `(Tr, setpoint)` — current reactor
temperature and the target temperature at that timestep, read from a
setpoint trajectory CSV (one value per second, 7200 steps = 2 hours of
simulated time).

**Action** — continuous, 2-dimensional:

| Action | Range | Meaning |
|---|---|---|
| `u[0]` coolant flow | 0.1 – 1.0 LPM | Flow rate through the cooling jacket |
| `u[1]` heater current | 4 – 20 mA | Heater drive signal (standard industrial 4-20 mA loop), internally scaled to an effective heating power via `((mA - 4) / 16)^2` |

**Reward** is asymmetric and shaped for precision tracking:
- Overshoot (`Tr` above setpoint) is penalized ~4x harder than undershoot,
  since overheating risks a runaway polymerization reaction.
- An error-derivative term penalizes oscillation/instability.
- Bonus reward is given for tracking within `0.3°C` and `0.8°C` bands.

An episode runs for the full 7200-step setpoint trajectory
(`Sample_Trajectory.csv`).

**Physics note:** the ODE (`br`) models reaction kinetics (initiator decay,
propagation), heat generation from polymerization, heat exchange between
reactor and jacket (`UA`), and coolant heat removal — parameterized by
kinetic constants (`Ad`, `Ed`, `Ap`, `Ep`, `deltaHp`, etc.) for a specific
polymer system.

## The controller (`DualPMD_Controller`)

A tabular Policy Mirror Descent controller with two independent action
heads — one for coolant flow, one for heater current — each learned with
TD-based Q-value estimates.

- **State discretization**: continuous tracking error (`setpoint - Tr`) is
  binned into 300 bins over a `±15°C` range.
- **Policy representation**: for each error bin, a softmax over discretized
  action values (`theta_coolant`, `theta_heater`), with temperature `tau`
  controlling exploration.
- **Mirror descent update**: policy logits (`theta`) are updated in the
  direction of learned Q-values, then clipped for numerical stability.
- **TD learning**: Q-tables (`Q_coolant`, `Q_heater`) are updated from an
  experience replay buffer (SARSA-style, using the actually-sampled next
  action rather than a max over actions).
- **Domain-informed priors**: the policy is initialized and nudged each
  update with a physical bias — when the reactor is too hot, favor more
  coolant flow and less heater current, and vice versa — so the controller
  starts from a sensible baseline rather than pure random exploration.
- **Action smoothing**: exponential momentum smooths successive actions to
  avoid actuator chatter. Heater current additionally has a forced
  low-amplitude oscillation superimposed, which keeps the heater action
  persistently exciting so its Q-values keep learning instead of collapsing
  onto a single value.

## Training loop

The notebook trains for 15 episodes, each running the full setpoint
trajectory. Per step: sample an action from the current policy, step the
environment, sample the next action (needed for the SARSA-style TD target),
store the transition in the replay buffer, and train on a random minibatch
from the buffer. Learning rate decays each episode. The final episode's
temperature tracking, tracking error, coolant flow, and heater current are
plotted against the setpoint.

## Files

| File | Description |
|---|---|
| `Miso_PMD+TD.ipynb` | Environment, controller, training loop, and plots |
| `Sample_Trajectory.csv` | Example setpoint trajectory used to drive an episode |
| `output.png` | Sample output plot from a training run |

## Requirements

`numpy`, `scipy`, `gym`, `pandas`, `matplotlib`

## Usage

The notebook loads the setpoint trajectory via `BR3('Trajectory2.csv')` —
change this path to point at `Sample_Trajectory.csv` (or your own setpoint
CSV) accordingly before running.
