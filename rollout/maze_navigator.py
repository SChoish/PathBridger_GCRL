"""Maze navigation helpers aligned with OGBench locomaze grid semantics.

When possible, maze metadata is loaded from the live registered gym env so the
rollout helper uses the same ``maze_map`` / offsets as ``MazeEnv``. Embedded
layouts remain as a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def infer_maze_type_from_env_name(env_name: str) -> str:
    """Infer OGBench locomaze maze type from env name."""
    name = env_name.lower()
    if 'teleport' in name:
        return 'teleport'
    if 'giant' in name:
        return 'giant'
    if 'large' in name:
        return 'large'
    if 'medium' in name:
        return 'medium'
    if 'arena' in name:
        return 'arena'
    raise ValueError(f'Could not infer maze type from env_name={env_name!r}')


def gymnasium_id_for_dataset(env_name: str) -> str:
    """Map OGBench dataset ids to registered gym ids when needed."""
    if env_name.endswith('-navigate-v0'):
        return env_name.replace('-navigate-v0', '-v0')
    return env_name


def _maze_map_from_type(maze_type: str) -> np.ndarray:
    if maze_type == 'arena':
        maze_map = [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
    elif maze_type == 'medium':
        maze_map = [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
    elif maze_type == 'large':
        maze_map = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    elif maze_type == 'giant':
        maze_map = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    elif maze_type == 'teleport':
        maze_map = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
            [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
            [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    else:
        raise ValueError(f'Unknown maze_type={maze_type!r}')
    return np.asarray(maze_map, dtype=np.int32)


@dataclass
class MazeNavigatorMap:
    maze_map: np.ndarray
    maze_type: str
    maze_unit: float = 4.0
    offset_x: float = 4.0
    offset_y: float = 4.0
    source: str = 'embedded'

    def __post_init__(self) -> None:
        self.maze_map = np.asarray(self.maze_map, dtype=np.int32)
        free = np.argwhere(self.maze_map == 0)
        self.free_ij = [tuple(map(int, ij)) for ij in free]
        self.free_xy = np.asarray([self.ij_to_xy(tuple(ij)) for ij in self.free_ij], dtype=np.float32)

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_maze_type_embedded(cls, maze_type: str, maze_unit: float = 4.0) -> MazeNavigatorMap:
        return cls(
            maze_map=_maze_map_from_type(maze_type),
            maze_type=maze_type,
            maze_unit=float(maze_unit),
            source=f'embedded:{maze_type}',
        )

    @classmethod
    def from_gym_env_name(cls, env_name: str) -> MazeNavigatorMap:
        """Load maze metadata directly from the registered OGBench env."""
        import ogbench  # noqa: F401
        import gymnasium as gym

        last_err: Exception | None = None
        for candidate in (gymnasium_id_for_dataset(env_name), env_name):
            try:
                env = gym.make(candidate)
                u = env.unwrapped
                depth = 0
                while not hasattr(u, 'maze_map') and hasattr(u, 'env') and depth < 8:
                    u = u.env
                    depth += 1
                if not hasattr(u, 'maze_map'):
                    raise AttributeError('unwrapped env has no maze_map')
                out = cls(
                    maze_map=np.asarray(u.maze_map, dtype=np.int32),
                    maze_type=str(getattr(u, '_maze_type', infer_maze_type_from_env_name(env_name))),
                    maze_unit=float(u._maze_unit),
                    offset_x=float(u._offset_x),
                    offset_y=float(u._offset_y),
                    source='gym',
                )
                env.close()
                return out
            except Exception as ex:
                last_err = ex
        raise RuntimeError(f'Could not load maze metadata from gym for {env_name!r}: {last_err!r}')

    @classmethod
    def from_env_name(cls, env_name: str, maze_unit: float = 4.0) -> MazeNavigatorMap:
        """Prefer live env metadata; fallback to embedded maze layout."""
        try:
            return cls.from_gym_env_name(env_name)
        except Exception:
            maze_type = infer_maze_type_from_env_name(env_name)
            return cls.from_maze_type_embedded(maze_type, maze_unit=maze_unit)

    # ------------------------------------------------------------------
    # coordinate transforms
    # ------------------------------------------------------------------
    def xy_to_ij(self, xy: np.ndarray | Tuple[float, float]) -> Tuple[int, int]:
        x, y = float(xy[0]), float(xy[1])
        i = int((y + self.offset_y + 0.5 * self.maze_unit) / self.maze_unit)
        j = int((x + self.offset_x + 0.5 * self.maze_unit) / self.maze_unit)
        return i, j

    def ij_to_xy(self, ij: Tuple[int, int]) -> Tuple[float, float]:
        i, j = int(ij[0]), int(ij[1])
        x = j * self.maze_unit - self.offset_x
        y = i * self.maze_unit - self.offset_y
        return float(x), float(y)

    def in_bounds(self, ij: Tuple[int, int]) -> bool:
        i, j = ij
        return 0 <= i < self.maze_map.shape[0] and 0 <= j < self.maze_map.shape[1]

    def is_free(self, ij: Tuple[int, int]) -> bool:
        return self.in_bounds(ij) and self.maze_map[ij[0], ij[1]] == 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def nearest_free_ij(self, xy: np.ndarray | Tuple[float, float]) -> Tuple[int, int]:
        xy = np.asarray(xy, dtype=np.float32).reshape(2)
        if len(self.free_xy) == 0:
            raise RuntimeError('No free cells in maze.')
        d2 = ((self.free_xy - xy[None]) ** 2).sum(axis=1)
        idx = int(np.argmin(d2))
        return self.free_ij[idx]

    def nearest_free_center_xy(self, xy: np.ndarray | Tuple[float, float]) -> np.ndarray:
        return np.asarray(self.ij_to_xy(self.nearest_free_ij(xy)), dtype=np.float32)

    def xy_is_in_free_cell(self, xy: np.ndarray | Tuple[float, float]) -> bool:
        """True when ``xy`` already lies in a walkable cell under ``xy_to_ij``."""
        return self.is_free(self.xy_to_ij(xy))

    def _rect_half(self, edge_inset: float) -> float:
        edge_inset = float(np.clip(edge_inset, 0.0, 0.99))
        return 0.5 * self.maze_unit * (1.0 - edge_inset)

    def project_to_free_union(self, xy: np.ndarray | Tuple[float, float], edge_inset: float = 0.08) -> np.ndarray:
        """Project xy to nearest point inside union of free-cell rectangles."""
        xy = np.asarray(xy, dtype=np.float32).reshape(2)
        half = self._rect_half(edge_inset)
        best: np.ndarray | None = None
        best_d2: float | None = None
        for cx, cy in self.free_xy:
            px = float(np.clip(xy[0], cx - half, cx + half))
            py = float(np.clip(xy[1], cy - half, cy + half))
            cand = np.asarray([px, py], dtype=np.float32)
            d2 = float(((cand - xy) ** 2).sum())
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = cand
        assert best is not None
        return best

    def _bfs_distances(self, goal_ij: Tuple[int, int]) -> np.ndarray:
        H, W = self.maze_map.shape
        dist = -np.ones((H, W), dtype=np.int32)
        if not self.is_free(goal_ij):
            goal_ij = self.nearest_free_ij(self.ij_to_xy(goal_ij))
        q = [goal_ij]
        dist[goal_ij] = 0
        head = 0
        while head < len(q):
            i, j = q[head]
            head += 1
            for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and self.maze_map[ni, nj] == 0 and dist[ni, nj] == -1:
                    dist[ni, nj] = dist[i, j] + 1
                    q.append((ni, nj))
        return dist

    def oracle_one_step_xy(self, start_xy: np.ndarray | Tuple[float, float], goal_xy: np.ndarray | Tuple[float, float]) -> np.ndarray:
        """One-step BFS oracle subgoal, matching OGBench maze logic."""
        start_ij = self.xy_to_ij(start_xy)
        goal_ij = self.xy_to_ij(goal_xy)

        if not self.is_free(start_ij):
            start_ij = self.nearest_free_ij(self.project_to_free_union(start_xy, edge_inset=0.0))
        if not self.is_free(goal_ij):
            goal_ij = self.nearest_free_ij(self.project_to_free_union(goal_xy, edge_inset=0.0))

        bfs_map = self._bfs_distances(goal_ij)
        subgoal_ij = start_ij
        best_val = int(bfs_map[start_ij[0], start_ij[1]])

        for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            ni, nj = start_ij[0] + di, start_ij[1] + dj
            if 0 <= ni < self.maze_map.shape[0] and 0 <= nj < self.maze_map.shape[1]:
                if self.maze_map[ni, nj] == 0 and bfs_map[ni, nj] >= 0:
                    if best_val < 0 or bfs_map[ni, nj] < best_val:
                        best_val = int(bfs_map[ni, nj])
                        subgoal_ij = (ni, nj)

        return np.asarray(self.ij_to_xy(subgoal_ij), dtype=np.float32)

    # ------------------------------------------------------------------
    # main API
    # ------------------------------------------------------------------
    def clamp_obs_xy(
        self,
        obs: np.ndarray,
        dim0: int,
        dim1: int,
        *,
        mode: str = 'ij',
        edge_inset: float = 0.08,
        goal_obs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Clamp observation x,y into walkable maze region.

        Modes:
        - ij     : keep continuous xy if already in a free cell; otherwise project to nearest walkable point
        - center : nearest free cell center
        - union  : nearest point inside union of free-cell rectangles
        - oracle : one-step BFS subgoal toward goal_obs
        """
        out = np.asarray(obs, dtype=np.float32).copy()
        xy = np.asarray([out[dim0], out[dim1]], dtype=np.float32)

        if mode == 'ij':
            if self.xy_is_in_free_cell(xy):
                clamped_xy = xy.copy()
            else:
                clamped_xy = self.project_to_free_union(xy, edge_inset=edge_inset)

        elif mode == 'center':
            clamped_xy = self.nearest_free_center_xy(xy)

        elif mode == 'union':
            clamped_xy = self.project_to_free_union(xy, edge_inset=edge_inset)

        elif mode == 'oracle':
            if goal_obs is None:
                raise ValueError('goal_obs must be provided when mode="oracle".')
            goal_xy = np.asarray([goal_obs[dim0], goal_obs[dim1]], dtype=np.float32)
            clamped_xy = self.oracle_one_step_xy(xy, goal_xy)

        else:
            raise ValueError(f'Unknown clamp mode: {mode!r}')

        out[dim0] = clamped_xy[0]
        out[dim1] = clamped_xy[1]
        return out

    def plot_free_skeleton(
        self,
        ax,
        dim0: int = 0,
        dim1: int = 1,
        *,
        edge_inset: float = 0.08,
        fill_cells: bool = False,
        facecolor: str = 'lightsteelblue',
        face_alpha: float = 0.12,
        show_points: bool = True,
        wall_facecolor: tuple[float, float, float, float] = (0.93, 0.935, 0.95, 1.0),
        wall_edgecolor: tuple[float, float, float, float] = (0.72, 0.76, 0.82, 1.0),
        free_edgecolor: tuple[float, float, float, float] = (0.22, 0.28, 0.36, 1.0),
        wall_linewidth: float = 0.55,
        free_linewidth: float = 0.78,
    ) -> None:
        """Lightweight maze overlay for x/y plots.

        When ``fill_cells`` is True, draws the **entire** maze grid (wall + free tiles) so cell
        boundaries stay crisp: face uses RGBA, edges use opaque RGBA (matplotlib ``alpha`` on the
        patch would fade edges together with the face).
        """
        if dim0 != 0 or dim1 != 1:
            return

        try:
            from matplotlib.patches import Rectangle
        except Exception:
            return

        half = self._rect_half(edge_inset)

        if fill_cells:
            # Parse free face color: allow '#rrggbb' or matplotlib name → RGBA face.
            try:
                import matplotlib.colors as mcolors

                fr, fg, fb = mcolors.to_rgb(facecolor)
            except Exception:
                fr, fg, fb = 0.62, 0.72, 0.84
            fa = float(np.clip(face_alpha, 0.0, 1.0))
            free_face = (float(fr), float(fg), float(fb), fa)

            H, W = int(self.maze_map.shape[0]), int(self.maze_map.shape[1])
            for i in range(H):
                for j in range(W):
                    cx, cy = self.ij_to_xy((i, j))
                    x0, y0 = float(cx - half), float(cy - half)
                    w = h = 2.0 * half
                    if int(self.maze_map[i, j]) != 0:
                        ax.add_patch(
                            Rectangle(
                                (x0, y0),
                                w,
                                h,
                                facecolor=wall_facecolor,
                                edgecolor=wall_edgecolor,
                                linewidth=float(wall_linewidth),
                                zorder=-2,
                            )
                        )
            for i in range(H):
                for j in range(W):
                    if int(self.maze_map[i, j]) != 0:
                        continue
                    cx, cy = self.ij_to_xy((i, j))
                    x0, y0 = float(cx - half), float(cy - half)
                    w = h = 2.0 * half
                    ax.add_patch(
                        Rectangle(
                            (x0, y0),
                            w,
                            h,
                            facecolor=free_face,
                            edgecolor=free_edgecolor,
                            linewidth=float(free_linewidth),
                            zorder=-1,
                        )
                    )
            return

        for cx, cy in self.free_xy:
            ax.add_patch(
                Rectangle(
                    (float(cx - half), float(cy - half)),
                    2.0 * half,
                    2.0 * half,
                    fill=False,
                    facecolor=facecolor,
                    edgecolor='lightsteelblue',
                    linewidth=0.5,
                    alpha=0.25,
                    zorder=-1,
                )
            )
        if show_points:
            ax.scatter(
                self.free_xy[:, 0],
                self.free_xy[:, 1],
                s=6,
                c='lightsteelblue',
                alpha=0.18,
                zorder=-1,
                linewidths=0,
            )
