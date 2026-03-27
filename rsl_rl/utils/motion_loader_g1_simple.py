import glob
import os

import numpy as np
import torch


def _resolve_motion_paths(motion_files: str) -> list[str]:
    if os.path.isdir(motion_files):
        paths = sorted(glob.glob(os.path.join(motion_files, "*.npy")))
    else:
        paths = sorted(glob.glob(motion_files))
    if not paths:
        raise FileNotFoundError(f"No motion files matched: {motion_files}")
    return paths


class G1SimpleAMPLoader:
    # For 23-joint simple-format data (30 cols incl. base pose),
    # remove both wrist_roll joints to match the 21-DoF training robot.
    _DROP_COLS_FROM_23 = (17, 22)

    def __init__(
        self,
        device,
        time_between_frames,
        motion_files,
        preload_transitions=False,
        num_preload_transitions=1000000,
        num_frames=5,
    ):
        self.device = device
        self.time_between_frames = time_between_frames
        self.num_frames = num_frames
        self.preload_transitions = preload_transitions

        self.trajectories = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_weights = []
        self.motion_paths = _resolve_motion_paths(motion_files)

        self._joint_cols_keep: tuple[int, ...] | None = None
        self._observation_dim: int | None = None

        for i, motion_path in enumerate(self.motion_paths):
            motion_file = os.path.basename(motion_path)
            motion_data = np.load(motion_path, allow_pickle=True)
            if motion_data.ndim != 2:
                raise ValueError(f"{motion_file} expected 2D array, got shape {motion_data.shape}")

            joint_cols_keep = self._resolve_joint_cols(motion_file, motion_data.shape[1])
            if self._joint_cols_keep is None:
                self._joint_cols_keep = joint_cols_keep
                self._observation_dim = len(joint_cols_keep)
            elif self._joint_cols_keep != joint_cols_keep:
                raise ValueError(
                    f"Inconsistent AMP joint column layout: {motion_file} differs from previous files"
                )

            self.trajectory_names.append(motion_file)
            self.trajectories.append(
                torch.tensor(motion_data, dtype=torch.float32, device=self.device)
            )
            self.trajectory_idxs.append(i)
            self.trajectory_weights.append(1 / len(self.motion_paths))
            print(f"Loaded {motion_data.shape[0]} frames from {motion_file}.")

        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)

        if self.preload_transitions:
            print(f"Preloading {num_preload_transitions} transitions")
            self.preloaded_frames = [[] for _ in range(self.num_frames)]
            frame_offsets = [i - (self.num_frames - 2) for i in range(self.num_frames)]
            min_offset = min(frame_offsets)
            max_offset = max(frame_offsets)

            assert self._joint_cols_keep is not None
            for _ in range(num_preload_transitions):
                traj_idx = np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)
                trajectory = self.trajectories[traj_idx]
                min_index = -min_offset
                max_index = trajectory.shape[0] - 1 - max_offset
                frame_idx = np.random.randint(min_index, max_index + 1)
                for frame_list, offset in zip(self.preloaded_frames, frame_offsets):
                    frame_list.append(trajectory[frame_idx + offset, self._joint_cols_keep])

            self.preloaded_frames = [
                torch.stack(frame_list, dim=0) for frame_list in self.preloaded_frames
            ]
            print("Finished preloading multiple frames")

    @classmethod
    def _resolve_joint_cols(cls, motion_file: str, num_cols: int) -> tuple[int, ...]:
        # [base_pos(3), base_quat(4), joints(...)]
        if num_cols == 30:
            # 7 + 23 joints -> keep 21 by dropping wrist_roll indices.
            return tuple(7 + i for i in range(23) if i not in cls._DROP_COLS_FROM_23)
        if num_cols == 28:
            # 7 + 21 joints already prepared.
            return tuple(range(7, 28))
        raise ValueError(
            f"{motion_file} expected 30 (7+23) or 28 (7+21) columns, got {num_cols}"
        )

    def feed_forward_generator_23dof_multi(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            if not self.preload_transitions:
                raise NotImplementedError("Only preload_transitions=True is supported")
            idxs = np.random.choice(self.preloaded_frames[0].shape[0], size=mini_batch_size)
            frames = [frame[idxs] for frame in self.preloaded_frames]
            yield torch.stack(frames, dim=1)

    @property
    def observation_dim(self):
        if self._observation_dim is None:
            raise RuntimeError("AMP loader not initialized")
        return self._observation_dim

    @property
    def num_motions(self):
        return len(self.trajectory_names)
