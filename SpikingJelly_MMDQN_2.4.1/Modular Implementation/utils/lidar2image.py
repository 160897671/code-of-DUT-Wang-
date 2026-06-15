import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm as _cm

# Build viridis LUT once at import time (256x3 uint8)
_x = np.linspace(0.0, 1.0, 256)
_VIRIDIS_LUT = np.uint8(_cm.viridis(_x)[:, :3] * 255)


def draw_centered_vehicle(grid, length_cells, width_cells, heading_rad, value=1):
    H, W = grid.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    j = np.arange(W)[None, :]
    i = np.arange(H)[:, None]
    x = j - cx
    y = i - cy
    c, s  = np.cos(heading_rad), np.sin(heading_rad)
    x_loc =  c * x + s * y
    y_loc = -s * x + c * y
    mask = (np.abs(x_loc) <= length_cells / 2.0) & (np.abs(y_loc) <= width_cells / 2.0)
    grid[mask] = value
    return grid


class LidarToOccupancyGrid:
    def __init__(self, output_range, v_max, voxel_size=0.5,
                 x_range=(-60, 60), y_range=(-60, 60), max_distance=60):
        self.voxel_size   = voxel_size
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.max_distance = max_distance
        self.output_range = output_range
        self.v_max        = v_max
        self.grid_width  = int((self.x_max - self.x_min) / self.voxel_size)
        self.grid_height = int((self.y_max - self.y_min) / self.voxel_size)
        self.occupancy_grid = np.zeros(
            (self.grid_height + 1, self.grid_width + 1), dtype=np.float32
        )
        print(self.occupancy_grid.shape)
        self._cx = self.grid_width  // 2
        self._cy = self.grid_height // 2

        # Pre-compute decay weights for the maximum possible num points
        # max num = int((1 - 0) * 500) = 500
        self._max_pts = 500
        self._decay_base = 0.98 ** np.arange(self._max_pts)  # (500,)

    def process(self, obs, angles, v_vel, heading):
        self.occupancy_grid.fill(0)

        # ---- ego vehicle ----
        vl = int(2 // self.voxel_size)
        vh = int(5 // self.voxel_size)
        self.occupancy_grid = draw_centered_vehicle(
            self.occupancy_grid, vl, vh, np.pi - heading, v_vel
        )

        # ---- fully vectorised LiDAR projection ----
        obs      = np.asarray(obs, dtype=np.float32)   # (800, 2)
        presence = obs[:, 0]
        dist_raw = obs[:, 1]
        valid    = (presence > 0) & (presence != 1)    # (800,) bool

        if valid.any():
            # For each valid ray, num_i = int((1 - |presence_i|) * 500)
            # We use the SAME num for all rays = max across valid rays,
            # padding shorter rays so we can fully vectorise.
            # This trades a tiny amount of wasted computation for zero Python loops.
            pres_v   = presence[valid]                          # (M,)
            ang_v    = angles[valid]                            # (M,)
            dist_v   = dist_raw[valid]                          # (M,)
            nums     = np.maximum(1, ((1 - np.abs(pres_v)) * 500).astype(np.int32))
            N        = int(nums.max())                          # broadcast size

            # linspace per ray: shape (M, N)
            t        = np.linspace(0, 1, N)[None, :]           # (1, N)
            starts   = pres_v[:, None]                          # (M, 1)
            dist_mat = (starts + t * (1 - starts)) * self.max_distance  # (M, N)

            x_mat = dist_mat * np.cos(ang_v)[:, None]          # (M, N)
            y_mat = dist_mat * np.sin(ang_v)[:, None]          # (M, N)

            decay = self._decay_base[:N][None, :]               # (1, N)
            z_mat = decay * (dist_v[:, None] + v_vel)           # (M, N)

            # Mask out padding points beyond each ray's true num
            idx_n    = np.arange(N)[None, :]                    # (1, N)
            pad_mask = idx_n < nums[:, None]                    # (M, N)

            # Grid indices
            l_idx = (x_mat / self.voxel_size).astype(np.int32) + self._cx   # (M, N)
            k_idx = self._cy - (y_mat / self.voxel_size).astype(np.int32)   # (M, N)

            in_bounds = (
                (k_idx >= 0) & (k_idx < self.grid_height) &
                (l_idx >= 0) & (l_idx < self.grid_width) &
                pad_mask
            )

            # Write all valid points at once (last-write-wins for overlaps)
            self.occupancy_grid[l_idx[in_bounds], k_idx[in_bounds]] = z_mat[in_bounds]

        # ---- crop & normalise ----
        r0, r1 = self.output_range
        occ    = self.occupancy_grid[r0:r1, r0:r1] / self.v_max

        # ---- viridis via LUT ----
        idx    = np.clip((occ * 255).astype(np.int32), 0, 255)
        rgbocc = _VIRIDIS_LUT[idx]               # (H, W, 3)
        rgbocc = np.transpose(rgbocc, (2, 0, 1)) # (3, H, W)
        return rgbocc
