import os
import tempfile
from datetime import datetime

import absl.flags as flags
import ml_collections
import numpy as np


class CsvLogger:
    """CSV logger for logging metrics to a CSV file."""

    def __init__(self, path, *, resume: bool = False, flush_every_n: int = 20):
        self.path = path
        self.resume = bool(resume)
        self.flush_every_n = max(1, int(flush_every_n))
        self.header = None
        self.file = None
        self._rows_since_flush = 0
        self.disallowed_types = _wandb_media_types()

    def _filtered_row(self, row):
        return {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}

    def _ensure_file(self, row):
        if self.file is not None:
            return
        if self.resume and os.path.isfile(self.path):
            with open(self.path, 'r', encoding='utf-8') as rf:
                header_line = rf.readline().strip()
            if not header_line:
                raise ValueError(f'Cannot resume empty CSV: {self.path}')
            self.header = header_line.split(',')
            self.file = open(self.path, 'a', encoding='utf-8')
            return

        self.file = open(self.path, 'w', encoding='utf-8')
        filtered_row = self._filtered_row(row)
        self.header = list(filtered_row.keys())
        self.file.write(','.join(self.header) + '\n')

    def log(self, row, step):
        row['step'] = step
        self._ensure_file(row)
        filtered_row = self._filtered_row(row)
        self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        self._rows_since_flush += 1
        if self._rows_since_flush >= self.flush_every_n:
            self.file.flush()
            self._rows_since_flush = 0

    def close(self):
        if self.file is not None:
            self.file.flush()
            self.file.close()


def get_exp_name(seed, env_name=None, agent_name=None):
    """Return the experiment name (wandb run name / save subdir).

    Order: {agent_name}_{env_name}_sd{seed}_... when agent/env given; else sd{seed}_...
    """
    tail = f'sd{seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        tail += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        tail += f'{os.environ["SLURM_PROCID"]}.'
    tail += f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    prefix_parts = []
    if agent_name:
        prefix_parts.append(str(agent_name).replace('/', '_'))
    if env_name:
        prefix_parts.append(str(env_name).replace('/', '_'))
    if prefix_parts:
        return '_'.join(prefix_parts) + '_' + tail
    return tail


def get_flag_dict():
    """Return the dictionary of flags."""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def _import_wandb():
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            'Weights & Biases support is optional in the production install. '
            'Install the `experiment` extra or add `wandb` to use --use_wandb.'
        ) from exc
    return wandb


def _wandb_media_types():
    try:
        wandb = _import_wandb()
    except RuntimeError:
        return ()
    return (wandb.Image, wandb.Video, wandb.Histogram)


def setup_wandb(
    entity=None,
    project='project',
    group=None,
    name=None,
    mode=None,
):
    """Set up Weights & Biases for logging."""
    if mode is None:
        mode = os.environ.get('WANDB_MODE', 'online')
    wandb = _import_wandb()
    wandb_output_dir = tempfile.mkdtemp()
    tags = [group] if group is not None else None

    init_kwargs = dict(
        config=get_flag_dict(),
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=wandb.Settings(
            start_method='thread',
            _disable_stats=False,
        ),
        mode=mode,
        save_code=True,
    )

    run = wandb.init(**init_kwargs)

    return run


def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, (n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, (t, c, n_rows * h, n_cols * w))

    return v


def get_wandb_video(renders=None, n_cols=None, fps=15):
    """Return a Weights & Biases video.

    It takes a list of videos and reshapes them into a single video with the specified number of columns.

    Args:
        renders: List of videos. Each video should be a numpy array of shape (t, h, w, c).
        n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
    """
    # Pad videos to the same length.
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # Decrease brightness of the padded frames.
        final_frame = render[-1]
        try:
            from PIL import Image, ImageEnhance
        except ImportError as exc:
            raise RuntimeError(
                'Video logging requires Pillow. Install the `visualization` extra or add `pillow`.'
            ) from exc

        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        # Add borders.
        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

    wandb = _import_wandb()
    return wandb.Video(renders, fps=fps, format='mp4')
