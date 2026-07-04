"""2D plots and MP4 export for dynamics rollout scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from rollout.maze_navigator import MazeNavigatorMap


def axis_limits(
    traj: np.ndarray,
    roll: np.ndarray,
    hats: np.ndarray,
    d0: int,
    d1: int,
    s_g: np.ndarray,
    s0: np.ndarray,
    margin_frac: float = 0.05,
    navigator: MazeNavigatorMap | None = None,
    seg: np.ndarray | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = [traj[:, d0], roll[:, d0], s_g[d0 : d0 + 1], s0[d0 : d0 + 1]]
    ys = [traj[:, d1], roll[:, d1], s_g[d1 : d1 + 1], s0[d1 : d1 + 1]]
    if seg is not None:
        xs.append(seg[:, d0])
        ys.append(seg[:, d1])
    if navigator is not None and d0 == 0 and d1 == 1:
        xs.append(navigator.free_xy[:, 0])
        ys.append(navigator.free_xy[:, 1])
    if hats.size > 0:
        xs.append(hats[:, d0])
        ys.append(hats[:, d1])
    x_all = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in xs])
    y_all = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in ys])
    xr = float(np.ptp(x_all)) + 1e-6
    yr = float(np.ptp(y_all)) + 1e-6
    xm = margin_frac * xr
    ym = margin_frac * yr
    x_min, x_max = float(x_all.min() - xm), float(x_all.max() + xm)
    y_min, y_max = float(y_all.min() - ym), float(y_all.max() + ym)
    return (x_min, x_max), (y_min, y_max)


def maze_navigator_for_xy_plot(
    navigator: MazeNavigatorMap | None,
    env_name: str | None,
    d0: int,
    d1: int,
) -> MazeNavigatorMap | None:
    """Return the maze navigator used for **plotting** (tiles + axis padding).

    * If ``navigator`` is already set (e.g. ``--navigator snap``), use it for tiles + padding.
    * Otherwise, when plotting obs dims ``(0, 1)`` and ``env_name`` looks like an OGBench
      locomaze task, load maze metadata via :meth:`MazeNavigatorMap.from_env_name` so cell
      tiles render **without** changing rollout dynamics (snap/clamp stays off unless requested).
    """
    if d0 != 0 or d1 != 1:
        return navigator
    if navigator is not None:
        return navigator
    name = (env_name or '').strip()
    if not name:
        return None
    try:
        return MazeNavigatorMap.from_env_name(name)
    except Exception:
        return None


def plot_maze_cell_tiles(ax, nav: MazeNavigatorMap | None, d0: int, d1: int) -> None:
    """Draw wall + free cell tiles (xy plot only)."""
    if nav is None or d0 != 0 or d1 != 1:
        return
    nav.plot_free_skeleton(
        ax,
        d0,
        d1,
        edge_inset=0.06,
        fill_cells=True,
        facecolor='#7aa8d4',
        face_alpha=0.48,
        show_points=False,
        wall_facecolor=(0.94, 0.95, 0.97, 1.0),
        wall_edgecolor=(0.78, 0.82, 0.88, 1.0),
        free_edgecolor=(0.18, 0.24, 0.32, 1.0),
        wall_linewidth=0.6,
        free_linewidth=0.85,
    )


def draw_dataset_background(
    ax, traj: np.ndarray, d0: int, d1: int, *, alpha_line: float = 0.38, alpha_scatter: float = 0.32
) -> None:
    ax.plot(
        traj[:, d0],
        traj[:, d1],
        '-',
        color='0.4',
        lw=1.1,
        alpha=alpha_line,
        zorder=1,
        label='_nolegend_',
    )
    ax.scatter(
        traj[:, d0],
        traj[:, d1],
        c='0.45',
        s=9,
        alpha=alpha_scatter,
        zorder=2,
        linewidths=0,
        edgecolors='none',
    )


def _draw_rollout_step_frame(
    ax,
    traj: np.ndarray,
    roll: np.ndarray,
    hats: np.ndarray,
    s0: np.ndarray,
    s_g: np.ndarray,
    d0: int,
    d1: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    k: int,
    title: str,
    navigator: MazeNavigatorMap | None = None,
    chunk_hat_stride: int | None = None,
) -> None:
    plot_maze_cell_tiles(ax, navigator, d0, d1)
    draw_dataset_background(ax, traj, d0, d1)
    n_trans = int(roll.shape[0]) - 1
    if n_trans <= 0:
        ax.scatter([s0[d0]], [s0[d1]], c='black', s=90, marker='P', zorder=8, edgecolors='k', linewidths=0.4, label='$s_0$')
        ax.scatter([s_g[d0]], [s_g[d1]], c='limegreen', s=130, marker='*', zorder=9, edgecolors='k', linewidths=0.4, label='$s_g$')
        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f'obs dim {d0}')
        ax.set_ylabel(f'obs dim {d1}')
        if navigator is not None and d0 == 0 and d1 == 1:
            ax.grid(False)
        else:
            ax.grid(True, alpha=0.28)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.94)
        return

    k = int(np.clip(k, 0, n_trans - 1))
    ax.plot(
        roll[: k + 2, d0],
        roll[: k + 2, d1],
        '-',
        color='C1',
        lw=2.4,
        alpha=0.98,
        zorder=4,
        label='estimated traj',
    )
    if k >= 0:
        ax.scatter(
            roll[: k + 1, d0],
            roll[: k + 1, d1],
            c='C1',
            s=28,
            zorder=5,
            alpha=0.88,
            edgecolors='0.15',
            linewidths=0.35,
        )
    ax.scatter(
        [roll[k, d0]],
        [roll[k, d1]],
        c='royalblue',
        s=130,
        marker='o',
        zorder=10,
        alpha=0.95,
        edgecolors='k',
        linewidths=0.5,
        label='$s_k$',
    )
    hk = None
    if hats.shape[0] > 0:
        if chunk_hat_stride is not None and chunk_hat_stride > 0:
            # ``hats`` in ``rollout.subgoal`` already stores one repeated row per executed step,
            # so pick the first row of the current chunk rather than re-compressing by chunk index.
            hi = min((k // chunk_hat_stride) * int(chunk_hat_stride), int(hats.shape[0]) - 1)
            hk = hats[hi]
        elif hats.shape[0] > k:
            hk = hats[k]
    if hk is not None:
        ax.scatter(
            [hk[d0]],
            [hk[d1]],
            c='darkviolet',
            s=110,
            marker='D',
            zorder=9,
            alpha=0.98,
            edgecolors='k',
            linewidths=0.45,
            label='subgoal est',
        )
    ax.annotate(
        '',
        xy=(float(roll[k + 1, d0]), float(roll[k + 1, d1])),
        xytext=(float(roll[k, d0]), float(roll[k, d1])),
        arrowprops=dict(arrowstyle='->', color='orangered', lw=2.5, shrinkA=8, shrinkB=8),
        zorder=8,
    )
    ax.scatter([s_g[d0]], [s_g[d1]], c='limegreen', s=130, marker='*', zorder=11, edgecolors='k', linewidths=0.4, label='$s_g$')
    ax.scatter([s0[d0]], [s0[d1]], c='black', s=80, marker='P', zorder=11, edgecolors='k', linewidths=0.35, label='$s_0$')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'obs dim {d0}')
    ax.set_ylabel(f'obs dim {d1}')
    ax.set_title(title, fontsize=10)
    if navigator is not None and d0 == 0 and d1 == 1:
        ax.grid(False)
    else:
        ax.grid(True, alpha=0.28)
    h, leg_labels = ax.get_legend_handles_labels()
    by = dict(zip(leg_labels, h))
    ax.legend(by.values(), by.keys(), loc='upper right', fontsize=9, framealpha=0.94)


def _hstack_env_panel(env_rgb: np.ndarray, panel_rgb: np.ndarray) -> np.ndarray:
    eh, _ew, _ = env_rgb.shape
    ph, pw, _ = panel_rgb.shape
    if ph != eh:
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError('env/panel stacking requires Pillow (pip install pillow)') from e
        panel_rgb = np.asarray(
            Image.fromarray(panel_rgb).resize((int(round(pw * eh / ph)), eh), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
    return np.ascontiguousarray(np.hstack([env_rgb, panel_rgb]))


def overlay_rgb_frames_obs2d_panel(
    frames: np.ndarray,
    traj: np.ndarray,
    roll: np.ndarray,
    hats: np.ndarray,
    frame_plan_trajs: np.ndarray | None,
    s0: np.ndarray,
    s_g: np.ndarray,
    d0: int,
    d1: int,
    navigator: MazeNavigatorMap | None = None,
    *,
    env_name: str | None = None,
    panel_width: int = 360,
    panel_max_frac_of_env: float = 0.72,
    output_scale: float = 1.1,
    dpi: int = 120,
    chunk_hat_stride: int | None = None,
    traj_line_alpha: float = 0.98,
    traj_line_lw: float = 3.0,
    roll_point_alpha: float = 0.96,
    roll_point_size: float = 30.0,
    hats_alpha: float = 0.99,
    hats_size: float = 68.0,
    dataset_line_alpha: float = 0.44,
    dataset_scatter_alpha: float = 0.4,
    value_heatmap: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    value_heatmap_vmin: float | None = None,
    value_heatmap_vmax: float | None = None,
    value_heatmap_alpha: float = 0.5,
) -> np.ndarray:
    """Compose env frames with a right-side XY panel.

    The panel mirrors the reference layout: env render on the left, XY plot on the right.
    If ``frame_plan_trajs`` is provided, frame ``t`` draws only the current planned
    25-step segment for that replan, rather than the cumulative executed rollout.

    Pass ``navigator`` from ``--navigator snap`` if maze tiles should appear on the panel.

    Optional ``value_heatmap=(XX, YY, ZZ)`` draws a goal-conditioned scalar value field (e.g.
    ``sigmoid(V)``) under trajectories using ``pcolormesh`` (same ``xlim``/``ylim`` as the panel).
    The colormap uses log normalization so low-value regions remain visually separable.
    ``output_scale`` upsamples the final combined frame slightly for a larger MP4.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f'Expected uint8 frames (T,H,W,3), got {frames.shape}')

    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError('overlay_rgb_frames_obs2d_panel requires Pillow (pip install pillow)') from e

    T, H, W, _ = frames.shape
    if int(roll.shape[0]) != int(T):
        raise ValueError(f'RGB frames T={T} but roll has {roll.shape[0]} rows (expected equal)')

    nav_panel = maze_navigator_for_xy_plot(navigator, env_name, d0, d1)

    xlim, ylim = axis_limits(traj, roll, hats, d0, d1, s_g, s0, navigator=nav_panel, seg=None)
    base_frames = np.asarray(frames, dtype=np.uint8)
    out_frames: list[np.ndarray] = []
    panel_cap = int(round(float(panel_max_frac_of_env) * float(W)))
    pw = max(min(int(panel_width), max(panel_cap, 160)), 160)

    for t in range(T):
        fig, ax = plt.subplots(figsize=(pw / float(dpi), H / float(dpi)), dpi=int(dpi))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        if value_heatmap is not None:
            XX, YY, ZZ = value_heatmap
            zz_plot = np.asarray(ZZ, dtype=np.float32)
            finite = zz_plot[np.isfinite(zz_plot)]
            heat_norm = None
            if finite.size > 0:
                pos = finite[finite > 0.0]
                if pos.size > 0:
                    log_floor = max(float(np.min(pos)), 1e-6)
                    if value_heatmap_vmin is not None:
                        log_floor = max(log_floor, float(value_heatmap_vmin), 1e-6)
                    log_ceil = float(np.max(finite))
                    if value_heatmap_vmax is not None:
                        log_ceil = min(log_ceil, float(value_heatmap_vmax))
                    log_ceil = max(log_ceil, log_floor * 1.001)
                    zz_plot = np.maximum(zz_plot, log_floor)
                    heat_norm = mcolors.LogNorm(vmin=log_floor, vmax=log_ceil)
            ax.pcolormesh(
                XX,
                YY,
                zz_plot,
                shading='auto',
                cmap='magma',
                alpha=float(value_heatmap_alpha),
                norm=heat_norm,
                zorder=1,
                rasterized=True,
            )
        plot_maze_cell_tiles(ax, nav_panel, d0, d1)
        cur = roll[min(int(t), int(roll.shape[0]) - 1)]
        plan_seg = None
        if frame_plan_trajs is not None and int(frame_plan_trajs.shape[0]) == T:
            plan_seg = np.asarray(frame_plan_trajs[t], dtype=np.float32)
            if plan_seg.size == 0:
                plan_seg = None
        hk = None
        if hats.size > 0:
            if chunk_hat_stride is not None and chunk_hat_stride > 0:
                # ``hats`` is step-aligned, so select the first repeated row of the active chunk.
                hi = min((int(t) // int(chunk_hat_stride)) * int(chunk_hat_stride), int(hats.shape[0]) - 1)
            else:
                hi = min(int(t), int(hats.shape[0]) - 1)
            hk = hats[hi]
        if plan_seg is not None:
            ax.plot(
                plan_seg[:, d0],
                plan_seg[:, d1],
                '-',
                color='#ffd400',
                lw=float(traj_line_lw),
                alpha=float(traj_line_alpha),
                zorder=6,
                label='estimated traj',
            )
        elif int(t) + 1 > 0:
            upto = int(t) + 1
            ax.plot(
                roll[:upto, d0],
                roll[:upto, d1],
                '-',
                color='#ffd400',
                lw=float(traj_line_lw),
                alpha=float(traj_line_alpha),
                zorder=6,
                label='estimated traj',
            )
        if hk is not None:
            ax.plot(
                [float(cur[d0]), float(hk[d0])],
                [float(cur[d1]), float(hk[d1])],
                linestyle='--',
                color='#ffa502',
                alpha=0.72,
                linewidth=1.4,
                zorder=7,
            )
            ax.scatter(
                [hk[d0]],
                [hk[d1]],
                c='darkviolet',
                s=float(hats_size * 1.18),
                marker='D',
                zorder=9,
                alpha=float(hats_alpha),
                edgecolors='0.06',
                linewidths=0.55,
                label='subgoal est',
            )
        ax.scatter(
            [cur[d0]],
            [cur[d1]],
            c='#3db8ff',
            s=96,
            marker='o',
            zorder=10,
            alpha=0.98,
            edgecolors='white',
            linewidths=0.8,
            label='state',
        )
        ax.scatter(
            [s_g[d0]],
            [s_g[d1]],
            c='limegreen',
            s=88,
            marker='*',
            zorder=10,
            edgecolors='0.2',
            linewidths=0.32,
            label='goal',
        )
        ax.scatter(
            [s0[d0]],
            [s0[d1]],
            c='black',
            s=58,
            marker='P',
            zorder=10,
            edgecolors='0.2',
            linewidths=0.28,
            label='start',
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        # Maze tiles provide structure; avoid a second grid fighting cell edges.
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#666')
        fig.subplots_adjust(left=0.04, right=0.985, bottom=0.03, top=0.985)
        fig.canvas.draw()
        w_buf, h_buf = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h_buf, w_buf, 4))
        panel = buf[:, :, :3].copy()
        plt.close(fig)
        panel = np.asarray(Image.fromarray(panel).resize((pw, H), Image.Resampling.LANCZOS))
        combined = _hstack_env_panel(base_frames[t], panel)
        if float(output_scale) > 1.0:
            ch, cw, _ = combined.shape
            up_w = max(int(round(cw * float(output_scale))), cw)
            up_h = max(int(round(ch * float(output_scale))), ch)
            if up_w % 2 == 1:
                up_w += 1
            if up_h % 2 == 1:
                up_h += 1
            combined = np.asarray(Image.fromarray(combined).resize((up_w, up_h), Image.Resampling.LANCZOS))
        out_frames.append(combined)
    return np.stack(out_frames, axis=0)


def compose_state_subgoal_env_frames(
    state_frames: np.ndarray,
    subgoal_frames: np.ndarray,
    *,
    goal_frames: np.ndarray | None = None,
    output_scale: float = 1.0,
    label_left: str | None = 'state',
    label_right: str | None = 'subgoal',
    label_goal: str | None = 'goal',
    label_pad: int = 4,
    label_font_size: int = 14,
) -> np.ndarray:
    """Stack two (or three) same-length env-render streams horizontally into a single RGB video.

    ``state_frames`` and ``subgoal_frames`` are uint8 ``(T, H, W, 3)`` arrays. If their per-frame
    sizes differ, trailing panels are resized to match the left panel's height. A small label
    bar is drawn above each panel when labels are non-empty (best-effort PIL; falls back to no
    labels if PIL is unavailable). When ``goal_frames`` is provided, a third panel is appended.
    """
    if state_frames.ndim != 4 or state_frames.shape[-1] != 3:
        raise ValueError(f'Expected uint8 state_frames (T,H,W,3), got {state_frames.shape}')
    if subgoal_frames.ndim != 4 or subgoal_frames.shape[-1] != 3:
        raise ValueError(f'Expected uint8 subgoal_frames (T,H,W,3), got {subgoal_frames.shape}')
    T_left = int(state_frames.shape[0])
    T_right = int(subgoal_frames.shape[0])
    if T_left != T_right:
        raise ValueError(f'state/subgoal frame counts must match, got {T_left} vs {T_right}.')
    if goal_frames is not None:
        if goal_frames.ndim != 4 or goal_frames.shape[-1] != 3:
            raise ValueError(f'Expected uint8 goal_frames (T,H,W,3), got {goal_frames.shape}')
        if int(goal_frames.shape[0]) != T_left:
            raise ValueError(
                f'goal frame count must match state/subgoal, got {goal_frames.shape[0]} vs {T_left}.'
            )
    if T_left == 0:
        return np.zeros((0, 1, 2, 3), dtype=np.uint8)

    try:
        from PIL import Image, ImageDraw, ImageFont
        pil_ok = True
    except ImportError:
        pil_ok = False

    H_left = int(state_frames.shape[1])

    def _resize_panel_to_height(frames: np.ndarray, height: int) -> np.ndarray:
        if int(frames.shape[1]) == height or not pil_ok:
            return frames
        target_w = int(round(int(frames.shape[2]) * height / int(frames.shape[1])))
        resized: list[np.ndarray] = []
        for t in range(int(frames.shape[0])):
            resized.append(
                np.asarray(
                    Image.fromarray(frames[t]).resize((target_w, height), Image.Resampling.LANCZOS),
                    dtype=np.uint8,
                )
            )
        return np.stack(resized, axis=0)

    subgoal_frames = _resize_panel_to_height(subgoal_frames, H_left)
    if goal_frames is not None:
        goal_frames = _resize_panel_to_height(goal_frames, H_left)

    show_labels = bool(label_left or label_right or (goal_frames is not None and label_goal)) and pil_ok
    label_h = 0
    font = None
    if show_labels:
        for fp in (
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        ):
            try:
                font = ImageFont.truetype(fp, int(label_font_size))
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()
        label_h = int(label_font_size) + 2 * int(label_pad)

    out = []
    for t in range(T_left):
        panels = [state_frames[t], subgoal_frames[t]]
        if goal_frames is not None:
            panels.append(goal_frames[t])
        panel_starts = [0]
        for p in panels[:-1]:
            panel_starts.append(panel_starts[-1] + int(p.shape[1]))
        body = np.concatenate(panels, axis=1)
        if show_labels:
            full = np.full((label_h + body.shape[0], body.shape[1], 3), 22, dtype=np.uint8)
            full[label_h:] = body
            img = Image.fromarray(full)
            draw = ImageDraw.Draw(img)
            label_specs = [
                (label_left, panel_starts[0]),
                (label_right, panel_starts[1]),
            ]
            if goal_frames is not None:
                label_specs.append((label_goal, panel_starts[2]))
            for text, x0 in label_specs:
                if text:
                    draw.text((int(x0) + int(label_pad), int(label_pad)), str(text), fill=(240, 240, 245), font=font)
            body = np.asarray(img, dtype=np.uint8)
        if float(output_scale) > 1.0 and pil_ok:
            ch, cw, _ = body.shape
            up_w = max(int(round(cw * float(output_scale))), cw)
            up_h = max(int(round(ch * float(output_scale))), ch)
            if up_w % 2 == 1:
                up_w += 1
            if up_h % 2 == 1:
                up_h += 1
            body = np.asarray(
                Image.fromarray(body).resize((up_w, up_h), Image.Resampling.LANCZOS), dtype=np.uint8
            )
        out.append(body)
    return np.stack(out, axis=0)


def overlay_rgb_frames_english_caption(
    frames: np.ndarray,
    lines: list[str],
    *,
    font_size: int = 12,
    margin: int = 6,
    darken: float = 0.72,
) -> np.ndarray:
    """Draw multi-line Latin/ASCII text on a dark strip at the bottom of each RGB frame (uint8 T,H,W,3).

    Uses PIL if available; avoids CJK so default Latin fonts suffice.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f'Expected uint8 frames (T,H,W,3), got {frames.shape}')
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        raise ImportError('overlay_rgb_frames_english_caption requires Pillow (pip install pillow)') from e

    T, H, W, _ = frames.shape
    out = np.empty_like(frames)
    font = None
    for fp in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
    ):
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    line_h = font_size + 5
    n_lines = max(1, len(lines))
    bar_h = min(int(H * 0.38), n_lines * line_h + 2 * margin)
    y0 = H - bar_h

    for t in range(T):
        fr = frames[t].astype(np.float32)
        bot = fr[y0:H]
        bot = bot * (1.0 - darken) + np.array([10.0, 10.0, 12.0], dtype=np.float32)
        fr[y0:H] = bot
        img = Image.fromarray(np.clip(fr, 0.0, 255.0).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        y = y0 + margin
        for line in lines:
            draw.text((margin, y), line, fill=(245, 245, 250), font=font)
            y += line_h
        out[t] = np.asarray(img)
    return out


def write_rgb_array_mp4(
    frames: np.ndarray,
    path: Path,
    fps: float,
    *,
    caption_lines: list[str] | None = None,
) -> None:
    """Write uint8 RGB frames ``(T, H, W, 3)`` to an H.264 MP4 (requires imageio + ffmpeg)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise ValueError(f'Expected frames (T,H,W,C) with C=3 or 4, got {frames.shape}')
    to_write = frames
    if caption_lines:
        if to_write.shape[-1] != 3:
            raise ValueError('caption overlay only supports RGB (3 channels)')
        to_write = overlay_rgb_frames_english_caption(to_write, caption_lines)
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio  # type: ignore
    imageio.mimwrite(str(path), to_write, fps=float(fps), macro_block_size=None)


def _configure_matplotlib_ffmpeg() -> None:
    try:
        import imageio_ffmpeg
        import matplotlib as mpl

        mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass


def write_rollout_mp4(
    traj: np.ndarray,
    roll: np.ndarray,
    hats: np.ndarray,
    s0: np.ndarray,
    s_g: np.ndarray,
    d0: int,
    d1: int,
    mp4_path: Path,
    fps: float,
    title_prefix: str,
    navigator: MazeNavigatorMap | None = None,
    seg: np.ndarray | None = None,
    chunk_hat_stride: int | None = None,
    *,
    env_name: str | None = None,
) -> None:
    from matplotlib.animation import FFMpegWriter

    _configure_matplotlib_ffmpeg()

    nav = maze_navigator_for_xy_plot(navigator, env_name, d0, d1)
    xlim, ylim = axis_limits(traj, roll, hats, d0, d1, s_g, s0, navigator=nav, seg=seg)
    n_trans = max(0, int(roll.shape[0]) - 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    writer = FFMpegWriter(fps=fps)
    mp4_path = Path(mp4_path)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    with writer.saving(fig, str(mp4_path), dpi=110):
        if n_trans == 0:
            ax.clear()
            title = f'{title_prefix}  (0 planner steps)'
            _draw_rollout_step_frame(
                ax,
                traj,
                roll,
                hats,
                s0,
                s_g,
                d0,
                d1,
                xlim,
                ylim,
                0,
                title,
                navigator=nav,
                chunk_hat_stride=chunk_hat_stride,
            )
            writer.grab_frame()
        else:
            if chunk_hat_stride is not None and chunk_hat_stride > 0:
                n_chunk_rows = max(int(np.ceil(float(hats.shape[0]) / float(chunk_hat_stride))), 1)
            else:
                n_chunk_rows = max(int(hats.shape[0]), 1)
            for k in range(n_trans):
                ax.clear()
                if chunk_hat_stride is not None and chunk_hat_stride > 0:
                    c = k // chunk_hat_stride + 1
                    title = (
                        f'{title_prefix}  step {k + 1}/{n_trans}  '
                        f'chunk {c}/{n_chunk_rows}  subgoal est · {chunk_hat_stride} steps/chunk'
                    )
                else:
                    title = f'{title_prefix}  step {k + 1}/{n_trans}  subgoal est + next step'
                _draw_rollout_step_frame(
                    ax,
                    traj,
                    roll,
                    hats,
                    s0,
                    s_g,
                    d0,
                    d1,
                    xlim,
                    ylim,
                    k,
                    title,
                    navigator=nav,
                    chunk_hat_stride=chunk_hat_stride,
                )
                writer.grab_frame()

    plt.close(fig)
