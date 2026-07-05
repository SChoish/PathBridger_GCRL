# PathBridger Production Runtime

This repository contains the minimal production-oriented PathBridger runtime for loading trained
checkpoints and running deterministic OGBench evaluation. Research-only assets such as ablation
configs and qualitative plotting launchers have been removed from the default tree, while the
required actor-SPI finetuning path is kept for the standard two-stage deployment workflow.

## What is included

| Path | Role |
|------|------|
| `eval_checkpoint.py` | Primary deployment entry point for loading a saved run and evaluating a checkpoint. |
| `main.py` | IDM/dynamics/critic training entry point; default deployment workflow trains IDM for 1M steps or a specified horizon. |
| `train_actor_spi.py` | Actor-only SPI finetuning entry point; default deployment workflow finetunes the actor for 50K steps after IDM training. |
| `agents/` | Dynamics, critic, and actor model definitions required to restore checkpoints. |
| `utils/` | Dataset, environment, checkpoint, logging, and evaluation helpers. |
| `rollout/env.py`, `rollout/common.py`, `rollout/episode_runner.py`, `rollout/maze_navigator.py` | Runtime rollout primitives used by evaluation and optional MuJoCo rendering setup. |
| `config/` | Canonical environment configs plus actor-SPI finetuning defaults. |

## Installation

Use the base install for production evaluation:

```bash
pip install -r requirements.txt
```

or, when installing as a package:

```bash
pip install .
```

Optional experiment/visualization tools are not installed by default:

```bash
pip install '.[experiment]'
pip install '.[visualization]'
```

## Evaluating a checkpoint

```bash
MUJOCO_GL=egl python eval_checkpoint.py --run_dir=runs/<run_dir> --epoch=1000 --idm_only
```

`eval_checkpoint.py` reads the saved `flags.json`, restores the checkpoint parameters, rebuilds the
matching OGBench environment, and writes JSON metrics under `runs/<run_dir>/eval_results/`. If the
saved run does not specify `eval_task_ids`, checkpoint evaluation defaults to the standard OGBench
task sweep `1,2,3,4,5`; pass `--eval_task_ids=...` to override it.

## Default deployment workflow

The supported baseline workflow is two-stage:

1. Train the IDM/dynamics/critic stack for 1M steps, or pass an explicit `--train_steps` value.
2. Finetune the SPI actor for 50K steps from the selected checkpoint.

```bash
python main.py --run_config=config/antmaze-medium-navigate.yaml --train_steps=1000000 --use_wandb=False
python train_actor_spi.py --pretrained_ckpt_dir=runs/<run_dir> --actor_spi_steps=50000 \
  --actor_spi_config=config/actor_spi/actor_spi_default.yaml
```

Weights & Biases is now optional. If `--use_wandb=True` is requested, install the `experiment` extra first.

The dynamics path objective trains the bridge path once over the supervised trajectory segment. The
first-step diagnostic is still logged, but it is not added as a second standalone loss term.

## Repository policy

The production tree intentionally excludes:

- ablation-only YAML manifests;
- qualitative plotting/video rollout scripts;
- default dependencies on W&B, Matplotlib, MoviePy, and Pillow.

Ablation and visualization tools can be restored from research branches when needed, but they are not part of the deployment surface.

## License

MIT. See [LICENSE](LICENSE).
