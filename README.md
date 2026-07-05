# PathBridger

PathBridger is a production-oriented implementation of an offline, long-horizon,
goal-conditioned reinforcement learning pipeline for OGBench tasks. It combines a
flow-based subgoal model, forward-bridge residual dynamics, a transitive RL critic,
and an actor-SPI finetuning stage for reliable checkpoint evaluation and deployment.

The repository is intentionally slim: environment configs, training/evaluation entry
points, model definitions, and runtime utilities are kept; ablation-only manifests and
qualitative plotting/video rollout tools are excluded from the default tree.

## Overview

PathBridger uses a two-stage workflow:

1. **IDM / dynamics / critic training.** `main.py` trains the flow subgoal model,
   bridge dynamics, inverse dynamics model (IDM), and TRL critic. The standard
   deployment baseline uses 1M gradient steps unless a different `--train_steps` is
   specified.
2. **Actor-SPI finetuning.** `scripts/train_actor_spi.py` freezes the trained dynamics/IDM
   and critic, then finetunes only the deterministic SPI actor. The default actor
   finetuning horizon is 50K gradient steps.

At evaluation time, the runtime restores a saved run, rebuilds the matching OGBench
environment from the saved metadata, replans action chunks in closed loop, and records
success metrics under the run directory.

## Installation

PathBridger requires Python 3.9+ and is based on JAX, Flax, and OGBench.
Install the minimal runtime dependencies with:

```bash
pip install -r requirements.txt
```

or install the project package directly:

```bash
pip install .
```

Optional experiment and visualization dependencies are opt-in:

```bash
pip install '.[experiment]'      # Weights & Biases logging
pip install '.[visualization]'   # Matplotlib / MoviePy / Pillow tooling
```

For MuJoCo rendering on a headless machine, set `MUJOCO_GL=egl` before running
evaluation.

## Quick start

### 1. Train IDM / dynamics / critic

Use one of the environment configs in `config/` and train for the default production
horizon of 1M gradient steps, or pass your own step budget:

```bash
python main.py \
  --run_config=config/antmaze-medium-navigate.yaml \
  --train_steps=1000000 \
  --use_wandb=False
```

Training writes checkpoints, `flags.json`, logs, and evaluation outputs under
`runs/<run_dir>/`.

### 2. Finetune the SPI actor

After the IDM/dynamics/critic checkpoint is available, finetune the actor for 50K
steps:

```bash
python scripts/train_actor_spi.py \
  --pretrained_ckpt_dir=runs/<run_dir> \
  --actor_spi_steps=50000 \
  --actor_spi_config=config/actor_spi/actor_spi_default.yaml
```

The actor finetuning script keeps non-actor components frozen and stores actor-SPI
outputs under `checkpoints/actor_spi/` by default.

### 3. Evaluate a checkpoint

Evaluate an IDM-only checkpoint before actor finetuning:

```bash
MUJOCO_GL=egl python scripts/eval_checkpoint.py \
  --run_dir=runs/<run_dir> \
  --epoch=1000000 \
  --idm_only
```

Evaluate a run with an actor checkpoint by omitting `--idm_only`:

```bash
MUJOCO_GL=egl python scripts/eval_checkpoint.py \
  --run_dir=runs/<run_dir> \
  --epoch=1000000
```

Evaluation results are saved as JSON and appended to `eval_results/all.csv` inside
the run directory. If a saved run does not specify `eval_task_ids`, checkpoint
evaluation defaults to the standard OGBench task sweep `1,2,3,4,5`; pass
`--eval_task_ids=...` to override it.

## Configs

The repository keeps one canonical config per supported OGBench environment plus the
actor-SPI finetuning default:

| Config | Environment |
|--------|-------------|
| `config/antmaze-medium-navigate.yaml` | `antmaze-medium-navigate-v0` |
| `config/antmaze-large-navigate.yaml` | `antmaze-large-navigate-v0` |
| `config/antmaze-giant-navigate.yaml` | `antmaze-giant-navigate-v0` |
| `config/cube-single-play.yaml` | `cube-single-play-v0` |
| `config/cube-double-play.yaml` | `cube-double-play-v0` |
| `config/cube-triple-play.yaml` | `cube-triple-play-v0` |
| `config/puzzle-3x3-play.yaml` | `puzzle-3x3-play-v0` |
| `config/puzzle-4x4-play.yaml` | `puzzle-4x4-play-v0` |
| `config/actor_spi/actor_spi_default.yaml` | Actor-only SPI finetuning defaults |

A config sets the environment name, horizon, evaluation cadence, and environment-
specific dynamics/critic/actor hyperparameters. Command-line flags take precedence
where supported.

## Repository layout

| Path | Purpose |
|------|---------|
| `main.py` | IDM, dynamics, critic, and optional joint actor training entry point. |
| `scripts/train_actor_spi.py` | Actor-only SPI finetuning entry point used in the default deployment workflow. |
| `scripts/eval_checkpoint.py` | Checkpoint evaluation entry point. |
| `agents/` | Model and agent definitions for dynamics, critic, and actor components. |
| `utils/` | Dataset, checkpoint, eval-result I/O, logging, and model utility code required at runtime. |
| `rollout/` | Minimal rollout primitives needed by evaluation and MuJoCo setup. |
| `config/` | Environment configs and actor-SPI defaults. |

## Notes for deployment

- Keep `--train_actor_spi=False` during the IDM/dynamics/critic stage unless you
  intentionally want legacy joint actor training.
- Use `scripts/train_actor_spi.py` for the supported actor-only finetuning stage.
- Use `--idm_only` when evaluating checkpoints that do not contain a trained actor.
- W&B, Matplotlib, MoviePy, and Pillow are not required for the base runtime.
- The dynamics path objective trains the bridge path once over the supervised
  trajectory segment. The first-step diagnostic is still logged, but it is not
  added as a second standalone loss term.
- Ablation configs and qualitative video/plotting launchers are intentionally
  omitted from this production tree.

## Acknowledgments

This implementation builds on OGBench and follows the clean JAX project structure used
by the FQL codebase. See the FQL repository for the reference project style and related
flow-based RL implementation: https://github.com/seohongpark/fql.

## License

MIT. See [LICENSE](LICENSE).
