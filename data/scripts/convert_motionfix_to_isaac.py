"""
Convert MotionFix dataset dumps (e.g. motionfix_test.pth.tar) into Isaac/poselib motion files.

MotionFix entries (loaded via joblib) look like:
  data[sample_id].keys() -> {"motion_source", "motion_target", "text"}
  data[sample_id]["motion_source"].keys() -> {"rots", "trans", "joint_positions", "timestamp"}

This script converts `rots` (axis-angle, shape [T, 66]) + `trans` (shape [T, 3]) into
`poselib.skeleton.skeleton3d.SkeletonMotion` files (.npy) compatible with this repo's pipelines.
"""

from __future__ import annotations

import os
# Avoid Intel OpenMP SHM usage in sandboxed envs (/dev/shm may be unavailable).
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_DISABLE_SHM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import torch
import numpy as np
import joblib
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

# Make local `poselib` importable without requiring pip-install.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_POSELIB_ROOT = _REPO_ROOT / "poselib"
if _LOCAL_POSELIB_ROOT.exists():
    sys.path.insert(0, str(_LOCAL_POSELIB_ROOT))

from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree  # noqa: E402

TMP_SMPL_DIR = "/tmp/smpl"

# NOTE: For parallelization we rely on `fork` so the large joblib-loaded dict can be shared
# copy-on-write across worker processes. With `spawn`, each worker would need to reload the file.
_GLOBAL_DATA: Optional[dict[str, Any]] = None
_GLOBAL_CFG: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class ConvertedPaths:
    source: str
    target: str
    text: str
    fps_source: float
    fps_target: float


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _retarget_motion_pose_copy(motion: SkeletonMotion, robot_type: str) -> SkeletonMotion:
    """
    Retarget by directly copying (mapped) joint orientations from the source skeleton
    onto the target robot skeleton, without IK.

    This is intended for cases where the source and target share compatible joint
    local frames (up to a small name/structure mismatch).
    """

    robot_name = robot_type.split("_")[0] if robot_type.endswith("_humanoid") else robot_type
    if robot_name != "bio":
        raise ValueError(f"pose-copy retarget currently supports robot_type='bio', got {robot_type!r}")

    target_tree = SkeletonTree.from_mjcf(f"protomotions/data/assets/mjcf/{robot_name}.xml")

    src_names = list(motion.skeleton_tree.node_names)
    tgt_names = list(target_tree.node_names)
    src_name_to_idx = {n: i for i, n in enumerate(src_names)}
    tgt_name_to_idx = {n: i for i, n in enumerate(tgt_names)}

    # Target->source name candidates.
    # This mapping is inspired by `data/scripts/retargeting/mink_retarget.py`'s BIO keypoint map,
    # but extended to cover intermediate limb segments present in bio.xml (ForeArm*, FootPinky*).
    tgt_to_src_candidates: dict[str, list[str]] = {
        # Root / spine.
        "Pelvis": ["Pelvis"],
        # bio has Pelvis->Spine->Torso, SMPL has Pelvis->Torso->Spine->Chest.
        "Spine": ["Torso", "Spine"],
        "Torso": ["Chest", "Spine", "Torso"],
        "Neck": ["Neck"],
        "Head": ["Head"],
        # Arms.
        # bio: Torso->Shoulder*->Arm*->ForeArm*->Hand*
        # smpl: Chest->*Thorax->*Shoulder->*Elbow->*Wrist->*Hand
        "ShoulderL": ["L_Thorax", "L_Shoulder"],
        "ArmL": ["L_Shoulder"],
        "ForeArmL": ["L_Elbow"],
        "HandL": ["L_Wrist", "L_Hand"],
        "ShoulderR": ["R_Thorax", "R_Shoulder"],
        "ArmR": ["R_Shoulder"],
        "ForeArmR": ["R_Elbow"],
        "HandR": ["R_Wrist", "R_Hand"],
        # Legs.
        "FemurL": ["L_Hip"],
        "TibiaL": ["L_Knee"],
        "TalusL": ["L_Ankle"],
        "FootThumbL": ["L_Toe", "L_Ankle"],
        "FootPinkyL": ["L_Toe", "L_Ankle"],
        "FemurR": ["R_Hip"],
        "TibiaR": ["R_Knee"],
        "TalusR": ["R_Ankle"],
        "FootThumbR": ["R_Toe", "R_Ankle"],
        "FootPinkyR": ["R_Toe", "R_Ankle"],
    }

    # Pick the first available source joint for each target joint.
    tgt_to_src: dict[str, str] = {}
    for tgt, candidates in tgt_to_src_candidates.items():
        for src in candidates:
            if src in src_name_to_idx and tgt in tgt_name_to_idx:
                tgt_to_src[tgt] = src
                break

    if "Pelvis" not in tgt_to_src:
        raise ValueError("pose-copy retarget requires a Pelvis mapping")

    # Copy local joint rotations where possible. This is generally more robust than
    # copying global rotations when the source/target hierarchies differ.
    src_local_rot = _as_numpy(motion.local_rotation).astype(np.float64)
    T = int(src_local_rot.shape[0])
    nb_tgt = len(tgt_names)

    tgt_local_rot = np.zeros((T, nb_tgt, 4), dtype=np.float64)
    tgt_local_rot[..., 3] = 1.0  # identity

    # Set mapped locals.
    for tgt_name, src_name in tgt_to_src.items():
        ti = tgt_name_to_idx[tgt_name]
        si = src_name_to_idx[src_name]
        tgt_local_rot[:, ti] = src_local_rot[:, si]

    # Root translations in this converter are built as:
    #   root_translation = trans_np + skeleton_tree.local_translation[0]
    # When switching skeleton trees, adjust by removing the source root offset and
    # adding the target root offset, otherwise the character may appear floating.
    src_root_translation = _as_numpy(motion.root_translation).astype(np.float64)
    src_root_offset = (
        _as_numpy(motion.skeleton_tree.local_translation[0]).astype(np.float64).reshape(1, 3)
    )
    tgt_root_offset = _as_numpy(target_tree.local_translation[0]).astype(np.float64).reshape(1, 3)
    tgt_root_translation = src_root_translation - src_root_offset + tgt_root_offset

    tgt_state = SkeletonState.from_rotation_and_root_translation(
        target_tree,
        torch.from_numpy(tgt_local_rot).float(),
        torch.from_numpy(tgt_root_translation).float(),
        is_local=True,
    )
    return SkeletonMotion.from_skeleton_state(tgt_state, fps=float(motion.fps))


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _infer_fps_from_timestamps(timestamps: Any, default_fps: float) -> float:
    if timestamps is None:
        return float(default_fps)
    ts = _as_numpy(timestamps).astype(np.float64).reshape(-1)
    if ts.size < 2:
        return float(default_fps)
    diffs = np.diff(ts)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 1e-8]
    if diffs.size == 0:
        return float(default_fps)
    dt = float(np.median(diffs))
    fps = 1.0 / dt
    if not np.isfinite(fps) or fps < 1.0 or fps > 240.0:
        return float(default_fps)
    return float(fps)


def _build_smpl_skeleton_tree(
    humanoid_type: str,
    upright_start: bool,
    humanoid_mjcf_path: Optional[str],
) -> SkeletonTree:
    if humanoid_mjcf_path is not None:
        return SkeletonTree.from_mjcf(humanoid_mjcf_path)

    try:
        from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency `smpl_sim`. Either install it, or pass `--humanoid-mjcf-path` "
            "to build the skeleton from an MJCF directly."
        ) from e

    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": True,
        "upright_start": upright_start,
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "replace_feet": True,
        "masterfoot": False,
        "big_ankle": True,
        "freeze_hand": False,
        "box_body": False,
        "master_range": 50,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "model": humanoid_type,
        "sim": "isaacgym",
    }

    smpl_local_robot = SMPL_Robot(robot_cfg, data_dir="data/smpl")
    uuid_str = uuid.uuid4()
    _ensure_dir(Path(TMP_SMPL_DIR))

    # MotionFix doesn't provide betas/gender, so default to neutral.
    betas = torch.zeros((1, 10), dtype=torch.float32)
    gender_number = [0]
    smpl_local_robot.load_from_skeleton(betas=betas, gender=gender_number, objs_info=None)
    xml_path = f"{TMP_SMPL_DIR}/smpl_humanoid_{uuid_str}.xml"
    smpl_local_robot.write_xml(xml_path)
    return SkeletonTree.from_mjcf(xml_path)


def _convert_axis_angle_to_motion(
    rots_aa: Any,
    trans: Any,
    timestamps: Any,
    skeleton_tree: SkeletonTree,
    mujoco_joint_names: list[str],
    joint_names: list[str],
    left_to_right_index: list[int],
    upright_start: bool,
    generate_flipped: bool,
    default_fps: float,
    force_retarget: bool,
    robot_type: str,
    fix_coord: bool,
    fix_ground: bool,
    ground_clearance: float,
) -> dict[str, SkeletonMotion]:
    pose_aa = _as_numpy(rots_aa).astype(np.float64)
    trans_np = _as_numpy(trans).astype(np.float64)

    if pose_aa.ndim != 2:
        raise ValueError(f"Expected rots shape [T, D], got {pose_aa.shape}")
    if trans_np.ndim != 2 or trans_np.shape[1] != 3:
        raise ValueError(f"Expected trans shape [T, 3], got {trans_np.shape}")
    if pose_aa.shape[0] != trans_np.shape[0]:
        raise ValueError(
            f"rots/trans frame mismatch: {pose_aa.shape[0]} vs {trans_np.shape[0]}"
        )

    mocap_fps = _infer_fps_from_timestamps(timestamps, default_fps=default_fps)
    batch_size = pose_aa.shape[0]

    if pose_aa.shape[1] not in (66, 72, 156):  # SMPL(22*3), SMPL(24*3), SMPLH/X-ish
        raise ValueError(f"Unexpected rots dim {pose_aa.shape[1]} (expected 66/72/156)")

    smpl_2_mujoco = [joint_names.index(q) for q in mujoco_joint_names if q in joint_names]

    # MotionFix SMPL dumps typically use a Y-up convention, while MuJoCo/Isaac uses Z-up.
    # Rotate -90deg about X: (x, y, z) -> (x, z, -y).
    # Apply the same basis change to local joint rotations via conjugation.
    coord_fix = sRot.from_euler("z", 180, degrees=True)

    if pose_aa.shape[1] == 66:
        pose_aa_full = np.concatenate([pose_aa, np.zeros((batch_size, 6))], axis=1)
        pose_aa_mj = pose_aa_full.reshape(batch_size, 24, 3)[:, smpl_2_mujoco]
        pose_quat_local = (
            sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
            .as_quat()
            .reshape(batch_size, 24, 4)
        )
    elif pose_aa.shape[1] == 72:
        pose_aa_mj = pose_aa.reshape(batch_size, 24, 3)[:, smpl_2_mujoco]
        pose_quat_local = (
            sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
            .as_quat()
            .reshape(batch_size, 24, 4)
        )
    else:
        # Keep the same convention used in convert_amass_to_isaac.py for smplh/smplx:
        #   concatenate [:66] + [75:] to drop hand pose parameters in a consistent way.
        pose_aa_full = np.concatenate([pose_aa[:, :66], pose_aa[:, 75:]], axis=-1)
        pose_aa_mj = pose_aa_full.reshape(batch_size, 52, 3)[:, smpl_2_mujoco]
        pose_quat_local = (
            sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
            .as_quat()
            .reshape(batch_size, 52, 4)
        )

    # if fix_coord:
    #     flat = pose_quat_local.reshape(-1, 4)
    #     pose_quat_local = (
    #         (coord_fix * sRot.from_quat(flat) * coord_fix.inv())
    #         .as_quat()
    #         .reshape(pose_quat_local.shape)
    #     )

    root_trans_offset = torch.from_numpy(trans_np).float() + skeleton_tree.local_translation[0]

    sk_state_local = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat_local).float(),
        root_trans_offset,
        is_local=True,
    )

    formats = ["regular"]
    if generate_flipped:
        formats.append("flipped")

    out: dict[str, SkeletonMotion] = {}
    for fmt in formats:
        if upright_start:
            b = pose_quat_local.shape[0]
            pose_quat_global = (
                (
                    sRot.from_quat(sk_state_local.global_rotation.reshape(-1, 4).numpy())
                    * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
                )
                .as_quat()
                .reshape(b, -1, 4)
            )
        else:
            pose_quat_global = sk_state_local.global_rotation.numpy()

        trans_out = root_trans_offset.clone()
        if fmt == "flipped":
            pose_quat_global = pose_quat_global[:, left_to_right_index]
            pose_quat_global[..., 0] *= -1
            pose_quat_global[..., 2] *= -1
            trans_out[..., 1] *= -1

        sk_state_global = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat_global).float(),
            trans_out,
            is_local=False,
        )
        motion = SkeletonMotion.from_skeleton_state(sk_state_global, fps=mocap_fps)

        if fix_ground:
            z = motion.global_translation[..., 2]
            min_z = float(z.min().item())
            if np.isfinite(min_z) and min_z < ground_clearance:
                dz = float(ground_clearance - min_z)
                lifted_root = motion.root_translation.clone()
                lifted_root[:, 2] += dz
                lifted_state = SkeletonState.from_rotation_and_root_translation(
                    motion.skeleton_tree,
                    motion.local_rotation.clone(),
                    lifted_root,
                    is_local=True,
                )
                motion = SkeletonMotion.from_skeleton_state(lifted_state, fps=motion.fps)

        if force_retarget:
            # `retarget_motion` expects the *target* robot key (e.g. "bio", "h1", "g1").
            # Do not pass "*_humanoid" here: the implementation indexes `_KEYPOINT_TO_JOINT_MAP[robot_type]`.
            target_robot_type = (
                robot_type.split("_")[0] if robot_type.endswith("_humanoid") else robot_type
            )
            if target_robot_type not in ("bio", "h1", "g1"):
                raise ValueError(
                    f"force_retarget=True requires robot_type in {{bio,h1,g1}}, got {robot_type!r}"
                )
            if target_robot_type == "bio":
                motion = _retarget_motion_pose_copy(motion=motion, robot_type=target_robot_type)
            else:
                from data.scripts.retargeting.mink_retarget import retarget_motion

                # Speed up IK retargeting by downsampling to ~30 FPS.
                if motion.fps > 30:
                    skip = max(1, int(motion.fps // 30))
                    sliced_state = SkeletonState.from_rotation_and_root_translation(
                        motion.skeleton_tree,
                        motion.local_rotation[::skip].clone(),
                        motion.root_translation[::skip].clone(),
                        is_local=True,
                    )
                    motion = SkeletonMotion.from_skeleton_state(
                        sliced_state, fps=motion.fps / skip
                    )

                motion = retarget_motion(
                    motion=motion,
                    robot_type=target_robot_type,
                    render=False,
                    progress=False,
                )

            if fix_ground:
                z = motion.global_translation[..., 2]
                min_z = float(z.min().item())
                if np.isfinite(min_z) and min_z < ground_clearance:
                    dz = float(ground_clearance - min_z)
                    lifted_root = motion.root_translation.clone()
                    lifted_root[:, 2] += dz
                    lifted_state = SkeletonState.from_rotation_and_root_translation(
                        motion.skeleton_tree,
                        motion.local_rotation.clone(),
                        lifted_root,
                        is_local=True,
                    )
                    motion = SkeletonMotion.from_skeleton_state(lifted_state, fps=motion.fps)

        out[fmt] = motion

    return out


def _save_motion_for_robot(motion: SkeletonMotion, path: Path) -> None:
    if path.suffix == ".pt":
        torch.save(motion, str(path))
    else:
        motion.to_file(str(path))


def _process_one_sample(sample_id: str) -> tuple[str, Optional[ConvertedPaths], Optional[str]]:
    global _GLOBAL_DATA, _GLOBAL_CFG
    try:
        assert _GLOBAL_DATA is not None
        assert _GLOBAL_CFG is not None

        # Keep each worker single-threaded to avoid oversubscription.
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        sample = _GLOBAL_DATA.get(sample_id, None)
        if not isinstance(sample, dict):
            return sample_id, None, None
        if "motion_source" not in sample or "motion_target" not in sample:
            return sample_id, None, None

        text = sample.get("text", "")
        src = sample["motion_source"]
        tgt = sample["motion_target"]

        save_ext = ".pt" if _GLOBAL_CFG["robot_type"] in ("h1", "g1") else ".npy"
        out_src_path: Path = _GLOBAL_CFG["source_dir"] / f"{sample_id}{save_ext}"
        out_tgt_path: Path = _GLOBAL_CFG["target_dir"] / f"{sample_id}{save_ext}"
        if (not _GLOBAL_CFG["force_remake"]) and out_src_path.exists() and out_tgt_path.exists():
            return (
                sample_id,
                ConvertedPaths(
                    source=str(out_src_path.relative_to(_GLOBAL_CFG["output_dir"])),
                    target=str(out_tgt_path.relative_to(_GLOBAL_CFG["output_dir"])),
                    text=str(text),
                    fps_source=float("nan"),
                    fps_target=float("nan"),
                ),
                None,
            )

        for part_name, part in (("source", src), ("target", tgt)):
            if not isinstance(part, dict):
                raise ValueError(f"{sample_id}:{part_name} expected dict, got {type(part)}")
            for need in ("rots", "trans"):
                if need not in part:
                    raise ValueError(f"{sample_id}:{part_name} missing key {need}")

        src_motions = _convert_axis_angle_to_motion(
            rots_aa=src["rots"],
            trans=src["trans"],
            timestamps=src.get("timestamp", None),
            skeleton_tree=_GLOBAL_CFG["skeleton_tree"],
            mujoco_joint_names=_GLOBAL_CFG["mujoco_joint_names"],
            joint_names=_GLOBAL_CFG["joint_names"],
            left_to_right_index=_GLOBAL_CFG["left_to_right_index"],
            upright_start=_GLOBAL_CFG["upright_start"],
            generate_flipped=_GLOBAL_CFG["generate_flipped"],
            default_fps=_GLOBAL_CFG["default_fps"],
            force_retarget=_GLOBAL_CFG["force_retarget"],
            robot_type=_GLOBAL_CFG["robot_type"],
            fix_coord=_GLOBAL_CFG["fix_coord"],
            fix_ground=_GLOBAL_CFG["fix_ground"],
            ground_clearance=_GLOBAL_CFG["ground_clearance"],
        )
        tgt_motions = _convert_axis_angle_to_motion(
            rots_aa=tgt["rots"],
            trans=tgt["trans"],
            timestamps=tgt.get("timestamp", None),
            skeleton_tree=_GLOBAL_CFG["skeleton_tree"],
            mujoco_joint_names=_GLOBAL_CFG["mujoco_joint_names"],
            joint_names=_GLOBAL_CFG["joint_names"],
            left_to_right_index=_GLOBAL_CFG["left_to_right_index"],
            upright_start=_GLOBAL_CFG["upright_start"],
            generate_flipped=_GLOBAL_CFG["generate_flipped"],
            default_fps=_GLOBAL_CFG["default_fps"],
            force_retarget=_GLOBAL_CFG["force_retarget"],
            robot_type=_GLOBAL_CFG["robot_type"],
            fix_coord=_GLOBAL_CFG["fix_coord"],
            fix_ground=_GLOBAL_CFG["fix_ground"],
            ground_clearance=_GLOBAL_CFG["ground_clearance"],
        )

        _save_motion_for_robot(src_motions["regular"], out_src_path)
        _save_motion_for_robot(tgt_motions["regular"], out_tgt_path)

        if _GLOBAL_CFG["generate_flipped"]:
            _save_motion_for_robot(
                src_motions["flipped"],
                out_src_path.with_name(out_src_path.stem + "_flipped" + out_src_path.suffix),
            )
            _save_motion_for_robot(
                tgt_motions["flipped"],
                out_tgt_path.with_name(out_tgt_path.stem + "_flipped" + out_tgt_path.suffix),
            )

        fps_src = float(src_motions["regular"].fps)
        fps_tgt = float(tgt_motions["regular"].fps)
        return (
            sample_id,
            ConvertedPaths(
                source=str(out_src_path.relative_to(_GLOBAL_CFG["output_dir"])),
                target=str(out_tgt_path.relative_to(_GLOBAL_CFG["output_dir"])),
                text=str(text),
                fps_source=fps_src,
                fps_target=fps_tgt,
            ),
            None,
        )
    except Exception as e:
        return sample_id, None, str(e)


def main(
    motionfix_file: Path,
    output_dir: Path = Path("motionfix"),
    split_name: Optional[str] = None,
    robot_type: str = "bio",
    humanoid_type: str = "smpl",
    humanoid_mjcf_path: Optional[str] = None,
    generate_flipped: bool = False,
    # MotionFix exports used in this repo are already Z-up; applying the AMASS-style
    # "upright_start" correction tends to rotate the character incorrectly.
    not_upright_start: bool = True,
    default_fps: float = 30.0,
    force_remake: bool = False,
    limit: Optional[int] = None,
    force_retarget: Optional[bool] = None,
    # MotionFix root translations are already in the repo's Z-up convention.
    fix_coord: bool = False,
    fix_ground: bool = True,
    ground_clearance: float = 0.02,
    num_workers: int = 0,
    chunksize: int = 4,
):
    """
    Args:
        motionfix_file: Path to MotionFix dump (e.g. motionfix_test.pth.tar).
        output_dir: Base output directory. Files go to `{output_dir}/npy/{split}/(source|target)/`.
        split_name: Output split folder name (default: derived from filename, e.g. "test"/"val").
        robot_type: Robot type (used only for save format parity with other scripts; default: "smpl").
        humanoid_type: One of "smpl", "smplh", "smplx".
        humanoid_mjcf_path: Optional MJCF path to define the skeleton tree directly.
        generate_flipped: Save `_flipped` variants like other converters.
        not_upright_start: If set, skip the upright-start rotation fix.
        default_fps: Used if timestamps are missing/unreliable.
        force_remake: Overwrite outputs if they already exist.
        limit: Process only the first N samples (for debugging).
        force_retarget: If None, defaults to True for robot_type in {bio,h1,g1}, else False.
        fix_coord: Convert MotionFix Y-up coords to MuJoCo Z-up (usually False for MotionFix here).
        fix_ground: Lift motions if below ground (never pushes down).
        ground_clearance: Minimum Z height after lifting.
        num_workers: Number of worker processes (0 -> use os.cpu_count()).
        chunksize: Pool imap chunksize; increase to reduce IPC overhead.
    """
    if not motionfix_file.exists():
        raise FileNotFoundError(str(motionfix_file))

    upright_start = not not_upright_start

    assert humanoid_type in ("smpl", "smplx", "smplh"), "humanoid_type must be smpl/smplx/smplh"

    if force_retarget is None:
        force_retarget = robot_type in ("bio", "h1", "g1")

    if humanoid_mjcf_path is not None:
        skeleton_tree = SkeletonTree.from_mjcf(humanoid_mjcf_path)
        mujoco_joint_names = list(skeleton_tree.node_names)
        joint_names = list(skeleton_tree.node_names)
    else:
        try:
            from smpl_sim.smpllib.smpl_joint_names import (
                SMPL_BONE_ORDER_NAMES,
                SMPL_MUJOCO_NAMES,
                SMPLH_BONE_ORDER_NAMES,
                SMPLH_MUJOCO_NAMES,
            )
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency `smpl_sim`. Either install it, or pass `--humanoid-mjcf-path` "
                "to build the skeleton from an MJCF directly."
            ) from e

        if humanoid_type == "smpl":
            mujoco_joint_names = list(SMPL_MUJOCO_NAMES)
            joint_names = list(SMPL_BONE_ORDER_NAMES)
        else:
            mujoco_joint_names = list(SMPLH_MUJOCO_NAMES)
            joint_names = list(SMPLH_BONE_ORDER_NAMES)

    left_to_right_index: list[int] = []
    for idx, entry in enumerate(mujoco_joint_names):
        if entry.startswith("R_"):
            left_to_right_index.append(mujoco_joint_names.index("L_" + entry[2:]))
        elif entry.startswith("L_"):
            left_to_right_index.append(mujoco_joint_names.index("R_" + entry[2:]))
        else:
            left_to_right_index.append(idx)

    if humanoid_mjcf_path is None:
        skeleton_tree = _build_smpl_skeleton_tree(
            humanoid_type=humanoid_type,
            upright_start=upright_start,
            humanoid_mjcf_path=None,
        )

    if split_name is None:
        name = motionfix_file.name.lower()
        if "test" in name:
            split_name = "test"
        elif "val" in name:
            split_name = "val"
        else:
            split_name = "train"

    base_dir = output_dir / "npy" / split_name
    source_dir = base_dir / "source"
    target_dir = base_dir / "target"
    _ensure_dir(source_dir)
    _ensure_dir(target_dir)

    meta: dict[str, ConvertedPaths] = {}

    start_time = torch.tensor(0.0)  # placeholder for mypy-friendly var
    del start_time
    import time as _time

    start_time = _time.time()

    print(f"Loading MotionFix dump: {motionfix_file}")
    data: dict[str, Any] = joblib.load(motionfix_file)
    sample_ids = list(data.keys())
    sample_ids.sort()
    if limit is not None:
        sample_ids = sample_ids[: int(limit)]

    global _GLOBAL_DATA, _GLOBAL_CFG
    _GLOBAL_DATA = data
    _GLOBAL_CFG = {
        "output_dir": output_dir,
        "source_dir": source_dir,
        "target_dir": target_dir,
        "robot_type": robot_type,
        "humanoid_type": humanoid_type,
        "generate_flipped": generate_flipped,
        "upright_start": upright_start,
        "default_fps": float(default_fps),
        "force_remake": bool(force_remake),
        "force_retarget": bool(force_retarget),
        "fix_coord": bool(fix_coord),
        "fix_ground": bool(fix_ground),
        "ground_clearance": float(ground_clearance),
        "skeleton_tree": skeleton_tree,
        "mujoco_joint_names": mujoco_joint_names,
        "joint_names": joint_names,
        "left_to_right_index": left_to_right_index,
    }

    if num_workers <= 0:
        num_workers = max(1, int(os.cpu_count() or 1))

    processed = 0
    errors: list[tuple[str, str]] = []

    if num_workers == 1:
        iterator = tqdm(sample_ids, desc=f"Converting MotionFix[{split_name}]")
        for sample_id in iterator:
            sid, converted, err = _process_one_sample(sample_id)
            if err is not None:
                errors.append((sid, err))
                continue
            if converted is None:
                continue
            meta[sid] = converted
            processed += 1
    else:
        import multiprocessing as mp

        # Prefer fork so workers share the already-loaded dict via copy-on-write.
        # If fork is unavailable, fallback to spawn (slower) with a clear error.
        if hasattr(mp, "get_context"):
            try:
                ctx = mp.get_context("fork")
            except ValueError:
                ctx = mp.get_context("spawn")
        else:
            ctx = mp

        if getattr(ctx, "get_start_method", lambda: "spawn")() != "fork":
            raise RuntimeError(
                "Parallel mode requires start_method='fork' to avoid re-loading the large MotionFix file in each worker. "
                "Run on Linux with fork available, or set --num-workers 1."
            )

        with ctx.Pool(processes=num_workers) as pool:
            iterator = pool.imap_unordered(_process_one_sample, sample_ids, chunksize=max(1, int(chunksize)))
            for res in tqdm(iterator, total=len(sample_ids), desc=f"Converting MotionFix[{split_name}]"):
                sid, converted, err = res
                if err is not None:
                    errors.append((sid, err))
                    continue
                if converted is None:
                    continue
                meta[sid] = converted
                processed += 1

    if errors:
        print(f"Encountered {len(errors)} errors (showing first 10):")
        for sid, msg in errors[:10]:
            print(f"  {sid}: {msg}")

    meta_path = base_dir / "meta.joblib"
    joblib.dump(meta, meta_path)
    print(f"Saved {processed} samples to {base_dir}")
    print(f"Saved metadata to {meta_path} (keys: source/target/text/fps_source/fps_target)")


if __name__ == "__main__":
    try:
        import typer  # type: ignore

        typer.run(main)
    except ModuleNotFoundError:
        import argparse

        parser = argparse.ArgumentParser(
            description="Convert MotionFix (.pth.tar via joblib) into poselib SkeletonMotion .npy files."
        )
        parser.add_argument("motionfix_file", type=Path)
        parser.add_argument("--output-dir", type=Path, default=Path("motionfix"))
        parser.add_argument("--split-name", type=str, default=None)
        parser.add_argument("--robot-type", type=str, default="bio")
        parser.add_argument("--humanoid-type", type=str, default="smpl")
        parser.add_argument("--humanoid-mjcf-path", type=str, default=None)
        parser.add_argument("--generate-flipped", action="store_true")
        upright_group = parser.add_mutually_exclusive_group()
        upright_group.add_argument("--upright-start", action="store_true")
        upright_group.add_argument("--not-upright-start", action="store_true")
        parser.add_argument("--default-fps", type=float, default=30.0)
        parser.add_argument("--force-remake", action="store_true",default=True)
        parser.add_argument("--limit", type=int, default=None)
        coord_group = parser.add_mutually_exclusive_group()
        coord_group.add_argument("--fix-coord", action="store_true")
        coord_group.add_argument("--no-fix-coord", action="store_true")
        parser.add_argument("--no-fix-ground", action="store_true")
        parser.add_argument("--ground-clearance", type=float, default=0.02)
        parser.add_argument("--num-workers", type=int, default=12)
        parser.add_argument("--chunksize", type=int, default=8)
        retarget_group = parser.add_mutually_exclusive_group()
        retarget_group.add_argument("--force-retarget", action="store_true")
        retarget_group.add_argument("--no-force-retarget", action="store_true")

        args = parser.parse_args()
        if args.force_retarget:
            force_retarget = True
        elif args.no_force_retarget:
            force_retarget = False
        else:
            force_retarget = None
        if args.upright_start:
            not_upright_start = False
        elif args.not_upright_start:
            not_upright_start = True
        else:
            not_upright_start = True
        if args.fix_coord:
            fix_coord = True
        elif args.no_fix_coord:
            fix_coord = False
        else:
            fix_coord = False
        main(
            motionfix_file=args.motionfix_file,
            output_dir=args.output_dir,
            split_name=args.split_name,
            robot_type=args.robot_type,
            humanoid_type=args.humanoid_type,
            humanoid_mjcf_path=args.humanoid_mjcf_path,
            generate_flipped=bool(args.generate_flipped),
            not_upright_start=bool(not_upright_start),
            default_fps=float(args.default_fps),
            force_remake=bool(args.force_remake),
            limit=args.limit,
            force_retarget=force_retarget,
            fix_coord=bool(fix_coord),
            fix_ground=not bool(args.no_fix_ground),
            ground_clearance=float(args.ground_clearance),
            num_workers=int(args.num_workers),
            chunksize=int(args.chunksize),
        )
