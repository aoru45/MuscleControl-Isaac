import os
from pathlib import Path

import torch
import typer
import yaml
import tempfile
import math
from hydra.utils import get_class
from hydra import compose, initialize

from protomotions.utils.motion_lib import MotionLib
from protomotions.simulator.base_simulator.config import RobotConfig

from omegaconf import OmegaConf, ListConfig
from protomotions.utils.motion_lib import LoadedMotions

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("sqrt", lambda x: math.sqrt(float(x)))
OmegaConf.register_new_resolver("sum", lambda x: sum(x))
OmegaConf.register_new_resolver("ceil", lambda x: math.ceil(x))
OmegaConf.register_new_resolver("int", lambda x: int(x))
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("sum_list", lambda lst: sum(lst))
OmegaConf.register_new_resolver("len_or_int_value", lambda lst: len(lst) if isinstance(lst, ListConfig) else int(lst))



def checkpasses_exclude_motion_filter(
    rigid_body_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    fps: float,
    min_height_threshold: float = -0.05,
    max_velocity_threshold: float = 15.0,
    max_dof_vel_threshold: float = 40.0,
    duration_height_filter: float = 0.2,
    duration_height_seconds: float = 1.0,
    motion_name: str = None,
    verbose: bool = True,
):
    """
    Filter function to exclude motions that don't meet quality criteria.

    Args:
        rigid_body_pos: Rigid body positions with shape [T, B, 3]
        dof_vel: DOF velocities with shape [T, D]
        fps: Motion frame rate
        min_height_threshold: Minimum height threshold for any body part
        max_velocity_threshold: Maximum velocity threshold for any body part
        max_dof_vel_threshold: Maximum DOF velocity threshold
        duration_height_filter: Height threshold for duration filter
        duration_height_seconds: Duration in seconds for height filter
        motion_name: Optional name/path for logging
        verbose: Print skip reasons

    Returns:
        bool: True if motion passes all filters, False otherwise
    """
    name_prefix = f"[{motion_name}] " if motion_name else ""

    # Check if any global_translation has z smaller than min_height_threshold
    min_height = rigid_body_pos[..., 2].min()
    if min_height < min_height_threshold:
        if verbose:
            print(
                f"{name_prefix}Skipping because min height {min_height} < {min_height_threshold}"
            )
        return False

    # Check if any global_velocity is too large (using finite difference)
    # not calling rigid_body_vel directly because it's smoothed by gaussian filter
    global_velocity_fin_diff = (rigid_body_pos[1:] - rigid_body_pos[:-1]) * float(fps)
    max_vel = global_velocity_fin_diff.abs().max()
    if max_vel > max_velocity_threshold:
        if verbose:
            print(
                f"{name_prefix}Skipping because max finite-diff velocity {max_vel} > {max_velocity_threshold}"
            )
        return False

    # Check if any dof_vels is too large
    max_dof_vel = dof_vel.abs().max()
    if max_dof_vel > max_dof_vel_threshold:
        if verbose:
            print(
                f"{name_prefix}Skipping because max dof vel {max_dof_vel} > {max_dof_vel_threshold}"
            )
        return False

    if duration_height_filter is not None:
        floor_estimate = min_height
        lowest_point_per_frame = rigid_body_pos[..., 2].min(dim=-1).values
        lowest_distance_from_floor = lowest_point_per_frame - floor_estimate
        too_high_per_frame = lowest_distance_from_floor > duration_height_filter
        # check if too high for n consecutive frames
        consecutive_frames = int(duration_height_seconds * float(fps))
        num_consecutive = 0
        for i in range(len(too_high_per_frame)):
            if too_high_per_frame[i]:
                num_consecutive += 1
            else:
                num_consecutive = 0

            if num_consecutive >= consecutive_frames:
                if verbose:
                    print(
                        f"{name_prefix}Skipping because it has {num_consecutive} consecutive frames with height > {duration_height_filter}"
                    )
                return False

    return True


def passes_exclude_motion_filter(*args, **kwargs) -> bool:
    """
    Backwards-compatible alias.

    Historically this script used `passes_exclude_motion_filter(motion: RobotState)`,
    but during packaging we have access to per-motion tensors instead. Prefer calling
    `checkpasses_exclude_motion_filter(...)` directly.
    """
    return checkpasses_exclude_motion_filter(*args, **kwargs)


def main(
        motion_file: Path,
        amass_data_path: Path,
        outpath: Path,
        humanoid_type: str = "bio",
        num_data_splits: int = None,
        exclude_motion_filter: bool = False,
        min_height_threshold: float = -0.05,
        max_velocity_threshold: float = 15.0,
        max_dof_vel_threshold: float = 40.0,
        duration_height_filter: float = 0.2,
        duration_height_seconds: float = 1.0,
):
    config_path = "../../protomotions/config/robot"

    with initialize(version_base=None, config_path=config_path, job_name="test_app"):
        cfg = compose(config_name=humanoid_type)

    key_body_ids = torch.tensor(
        [
            cfg.robot.body_names.index(key_body_name)
            for key_body_name in cfg.robot.key_bodies
        ],
        dtype=torch.long,
    )

    # Process the robot config into a RobotConfig object
    robot_config: RobotConfig = RobotConfig.from_dict(cfg.robot)

    print("Creating motion state")
    motion_files = []
    if num_data_splits is not None:
        # Motion file is a yaml file
        # Just load the yaml and break it into num_data_splits
        # Save each split as a separate file
        with open(os.path.join(os.getcwd(), motion_file), "r") as f:
            motions = yaml.load(f, Loader=yaml.SafeLoader)["motions"]
        num_motions = len(motions)
        split_size = num_motions // num_data_splits
        for i in range(num_data_splits):
            if i == num_data_splits - 1:  # make sure we get all the remaining motions
                split_motions = motions[i * split_size:]
            else:
                split_motions = motions[i * split_size: (i + 1) * split_size]

            motion_idx = 0
            for motion in split_motions:
                motion["idx"] = motion_idx
                if "sub_motions" in motion:
                    for sub_motion in motion["sub_motions"]:
                        sub_motion["idx"] = motion_idx
                        motion_idx += 1
                else:
                    motion_idx += 1

            split_name = motion_file.with_name(
                motion_file.stem + f"_{i}" + motion_file.suffix
            )
            with open(split_name, "w") as f:
                yaml.dump({"motions": split_motions}, f)

            motion_files.append(
                (
                    str(split_name),
                    outpath.with_name(outpath.stem + f"_{i}" + outpath.suffix),
                )
            )
    else:
        motion_files.append((motion_file, outpath))

    for motion_file, outpath in motion_files:
        # Open and edit the motion file
        with open(motion_file, "r") as f:
            motion_data = yaml.safe_load(f)

        # Edit file paths
        for motion in motion_data["motions"]:
            motion["file"] = str(amass_data_path.resolve() / motion["file"])

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
        ) as temp_file:
            yaml.dump(motion_data, temp_file)
            temp_file_path = temp_file.name

        # Use the temporary file for MotionLib
        cfg.motion_lib.motion_file = temp_file_path

        MotionLibClass = get_class(cfg.motion_lib._target_)
        motion_lib_params = {}
        for key, value in cfg.motion_lib.items():
            if key != "_target_":
                motion_lib_params[key] = value

        mlib: MotionLib = MotionLibClass(
            robot_config=robot_config,
            key_body_ids=key_body_ids,
            device="cpu",
            skeleton_tree=None,
            **motion_lib_params
        )

        state_to_save = mlib.state
        if exclude_motion_filter:
            keep_indices = []
            for i, motion in enumerate(mlib.state.motions):
                # `motion` is a `SkeletonMotion` and includes per-frame `global_translation`
                # and a cached `dof_vels` computed during load.
                motion_name = None
                if len(mlib.state.motion_files) == len(mlib.state.motions):
                    motion_name = str(mlib.state.motion_files[i])

                # NOTE: This matches MotionLib's internal representation:
                # - `self.gts` is `cat([m.global_translation ...])` and is used as `rigid_body_pos`
                # - `self.dvs` is `cat([m.dof_vels ...])` and is used as `dof_vel`
                rigid_body_pos = motion.global_translation
                dof_vel = mlib._compute_motion_dof_vels(motion)

                if checkpasses_exclude_motion_filter(
                    rigid_body_pos=rigid_body_pos,
                    dof_vel=dof_vel,
                    fps=float(motion.fps),
                    min_height_threshold=min_height_threshold,
                    max_velocity_threshold=max_velocity_threshold,
                    max_dof_vel_threshold=max_dof_vel_threshold,
                    duration_height_filter=duration_height_filter,
                    duration_height_seconds=duration_height_seconds,
                    motion_name=motion_name or f"motion_{i}",
                ):
                    keep_indices.append(i)

            if len(keep_indices) == 0:
                raise ValueError(
                    "All motions were filtered out by exclude_motion_filter; nothing to package."
                )

            keep_idx = torch.tensor(keep_indices, dtype=torch.long)

            motion_weights = mlib.state.motion_weights[keep_idx].clone()
            motion_weights_sum = float(motion_weights.sum())
            if motion_weights_sum <= 0:
                raise ValueError(
                    "Filtered motion weights sum to 0; cannot renormalize."
                )
            motion_weights /= motion_weights_sum

            motion_files = mlib.state.motion_files
            if len(motion_files) == len(mlib.state.motions):
                motion_files = tuple(motion_files[i] for i in keep_indices)

            state_to_save = LoadedMotions(
                motions=tuple(mlib.state.motions[i] for i in keep_indices),
                motion_lengths=mlib.state.motion_lengths[keep_idx],
                motion_weights=motion_weights,
                motion_fps=mlib.state.motion_fps[keep_idx],
                motion_dt=mlib.state.motion_dt[keep_idx],
                motion_num_frames=mlib.state.motion_num_frames[keep_idx],
                motion_files=tuple(motion_files),
                ref_respawn_offsets=mlib.state.ref_respawn_offsets[keep_idx],
                text_embeddings=mlib.state.text_embeddings[keep_idx]
                if hasattr(mlib.state, "text_embeddings")
                else None,
                has_text_embeddings=mlib.state.has_text_embeddings[keep_idx]
                if hasattr(mlib.state, "has_text_embeddings")
                else None,
            )
            print(
                f"Exclude filter kept {len(keep_indices)}/{len(mlib.state.motions)} motions."
            )

        print("Saving motion state")

        with open(outpath, "wb") as file:
            torch.save(state_to_save, file)

        # Remove the temporary file
        os.unlink(temp_file_path)


if __name__ == "__main__":
    typer.run(main)
