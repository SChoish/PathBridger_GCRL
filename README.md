# PathBridger

PathBridger is an offline, long-horizon, goal-conditioned RL method for [OGBench](https://github.com/seohongpark/ogbench).
It couples a **flow-based subgoal model**, a **forward-bridge residual dynamics planner**, a **transitive
RL (TRL) critic**, and an **SPI actor** into a single training loop.

## Overview

At each training step the components are updated jointly:

1. A `DynamicsAgent` samples a flow subgoal endpoint from the current state and the final goal.
2. A `forward_bridge_residual` planner rolls out a state trajectory from the current state to the subgoal.
3. An inverse dynamics model (IDM) turns the trajectory prefix into action-chunk proposals.
4. The **TRL critic** learns an action-chunk Q value together with a transitive state-pair value.
5. The **SPI actor** is updated to follow the critic-ranked action-chunk proposals.

At evaluation time, planning is closed-loop: the agent replans action chunks and steps the environment
until it reaches the environment's maximum episode length (the OGBench `TimeLimit`).

## Installation

PathBridger requires Python 3.9+ and is based on JAX. The main dependencies are `jax >= 0.4.26`,
`flax >= 0.8.4`, and `ogbench`. Install everything with:

```bash
pip install -r requirements.txt
```

To render rollouts with MuJoCo, set `MUJOCO_GL=egl` (headless) or `MUJOCO_GL=glfw`.

## Usage

Training is driven by a single YAML run config passed to `main.py`:

```bash
# Train on OGBench antmaze-medium-navigate (the default config)
python main.py --run_config=config/antmaze-medium-navigate.yaml
```

Each config sets the environment name, planning horizon, dynamics/critic/actor blocks, and evaluation
schedule. Structural constants and fixed hyperparameters are hardcoded in the agents, so the configs only
carry the environment-varying knobs (e.g. `discount`, `value_distance_weight_power`, goal sampling, subgoal
gap/weight).

## Reproducing the main results

The `config/` directory ships one best config per environment (matching the 1M-step best runs):

```bash
python main.py --run_config=config/antmaze-medium-navigate.yaml
python main.py --run_config=config/antmaze-large-navigate.yaml
python main.py --run_config=config/antmaze-giant-navigate.yaml
python main.py --run_config=config/cube-single-play.yaml
python main.py --run_config=config/cube-double-play.yaml
python main.py --run_config=config/cube-triple-play.yaml
python main.py --run_config=config/puzzle-3x3-play.yaml
python main.py --run_config=config/puzzle-4x4-play.yaml
```

Ablations for the `cube-double` best config (dedup path loss, no subgoal/dynamics time embedding) live in
`config/ablation_cd_1m_best/`.

## Actor SPI finetuning

`train_actor_spi.py` freezes the dynamics/IDM/critic of an existing run and finetunes only the deterministic
SPI actor:

```bash
python train_actor_spi.py --pretrained_ckpt_dir=runs/<run_dir> --actor_spi_steps=100000
```

Actor SPI defaults can be provided via `--actor_spi_config=config/actor_spi/actor_spi_default.yaml`
(command-line flags take precedence).

## Evaluating a checkpoint

`eval_checkpoint.py` reloads a training run and reruns the same environment evaluation as training. It rolls
out to the environment's max episode length by default:

```bash
MUJOCO_GL=egl python eval_checkpoint.py --run_dir=runs/<run_dir> --epoch=1000
```

## Rollout and visualization

The `rollout/` package provides a unified launcher for qualitative rollouts (IDM, actor, and maze-only
state-space subgoal plots):

```bash
PYTHONPATH=. python -m rollout.run --run_dir=runs/<run_dir> --mode all --task_ids=1,2,3
```

## Repository layout

| Path | Role |
|------|------|
| `main.py` | Training entry point; jointly trains dynamics, TRL critic, and SPI actor. |
| `train_actor_spi.py` | Actor-only SPI finetuning from an existing checkpoint. |
| `eval_checkpoint.py` | Reruns environment evaluation from a saved checkpoint. |
| `agents/` | `dynamics.py` (flow subgoal + bridge planner + IDM), `critic.py` (TRL critic), `actor.py` (SPI actor). |
| `utils/` | Datasets, networks, env setup, flax/checkpoint utilities, evaluation helpers. |
| `rollout/` | Qualitative rollout and visualization tools. |
| `config/` | One best run config per environment, plus ablations and the actor-SPI default. |

## Acknowledgments

This codebase builds on [OGBench](https://github.com/seohongpark/ogbench) and is structured after
[Flow Q-Learning (FQL)](https://github.com/seohongpark/fql).

## License

MIT. See [LICENSE](LICENSE).
