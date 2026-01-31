# This code is adapted from https://github.com/zhengyiluo/phc/ and generalized to work with any humanoid.
# https://github.com/ZhengyiLuo/PHC/blob/master/scripts/data_process/convert_amass_isaac.py

import os
import uuid
from pathlib import Path
from typing import Optional

import ipdb
import yaml
import numpy as np
import torch
import typer
from scipy.spatial.transform import Rotation as sRot
import pickle
from smpl_sim.smpllib.smpl_joint_names import (
    SMPL_BONE_ORDER_NAMES,
    SMPL_MUJOCO_NAMES,
    SMPLH_BONE_ORDER_NAMES,
    SMPLH_MUJOCO_NAMES,
)
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from tqdm import tqdm

from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree
import time
from datetime import timedelta

TMP_SMPL_DIR = "/tmp/smpl"


def _as_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _retarget_motion_pose_copy_bio(motion: SkeletonMotion) -> SkeletonMotion:
    """
    Retarget AMASS -> BIO by directly copying (mapped) joint local rotations
    from the source skeleton onto BIO, without IK.
    """
    target_tree = SkeletonTree.from_mjcf("protomotions/data/assets/mjcf/bio.xml")

    src_names = list(motion.skeleton_tree.node_names)
    tgt_names = list(target_tree.node_names)
    src_name_to_idx = {n: i for i, n in enumerate(src_names)}
    tgt_name_to_idx = {n: i for i, n in enumerate(tgt_names)}

    tgt_to_src_candidates: dict[str, list[str]] = {
        # Root / spine.
        "Pelvis": ["Pelvis"],
        "Spine": ["Torso", "Spine"],
        "Torso": ["Chest", "Spine", "Torso"],
        "Neck": ["Neck"],
        "Head": ["Head"],
        # Arms.
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

    tgt_to_src: dict[str, str] = {}
    for tgt, candidates in tgt_to_src_candidates.items():
        if tgt not in tgt_name_to_idx:
            continue
        for src in candidates:
            if src in src_name_to_idx:
                tgt_to_src[tgt] = src
                break
    if "Pelvis" not in tgt_to_src:
        raise ValueError("pose-copy retarget requires a Pelvis mapping")

    src_local_rot = _as_numpy(motion.local_rotation).astype(np.float64)
    T = int(src_local_rot.shape[0])
    nb_tgt = len(tgt_names)
    tgt_local_rot = np.zeros((T, nb_tgt, 4), dtype=np.float64)
    tgt_local_rot[..., 3] = 1.0

    for tgt_name, src_name in tgt_to_src.items():
        ti = tgt_name_to_idx[tgt_name]
        si = src_name_to_idx[src_name]
        tgt_local_rot[:, ti] = src_local_rot[:, si]

    # Adjust root translation for differing root offsets between source and bio skeleton trees.
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
    tgt_motion = SkeletonMotion.from_skeleton_state(tgt_state, fps=float(motion.fps))

    def _rotate_global(m: SkeletonMotion, r: sRot) -> SkeletonMotion:
        gr = _as_numpy(m.global_rotation).astype(np.float64)
        rt = _as_numpy(m.root_translation).astype(np.float64)
        gr_new = (
            (sRot.from_quat(gr.reshape(-1, 4)) * r).as_quat().reshape(gr.shape)
        )
        st = SkeletonState.from_rotation_and_root_translation(
            m.skeleton_tree,
            torch.from_numpy(gr_new).float(),
            torch.from_numpy(rt).float(),
            is_local=False,
        )
        return SkeletonMotion.from_skeleton_state(st, fps=float(m.fps))


    r_upright = sRot.from_quat([0.5, 0.5, 0.5, 0.5])

    
    return _rotate_global(tgt_motion, r_upright)


def main(
    amass_root_dir: Path,
    robot_type: str = None,
    humanoid_type: str = "smpl",
    force_remake: bool = True,
    force_neutral_body: bool = True,
    generate_flipped: bool = False,
    not_upright_start: bool = False,  # By default, let's start upright (for consistency across all models).
    humanoid_mjcf_path: Optional[str] = None,
    force_retarget: bool = True,
    output_dir: Path = None,
):
    if output_dir is None:
        output_dir = amass_root_dir

    if robot_type is None:
        robot_type = humanoid_type
    elif robot_type in ["h1", "g1", "bio"]:
        assert (
            force_retarget
        ), f"Data is either SMPL or SMPL-X. The {robot_type} robot must use the retargeting pipeline."
    assert humanoid_type in [
        "smpl",
        "smplx",
        "smplh",
    ], "Humanoid type must be one of smpl, smplx, smplh"
    append_name = robot_type
    if force_retarget:
        append_name += "_retargeted"
    upright_start = not not_upright_start

    if humanoid_type == "smpl":
        mujoco_joint_names = SMPL_MUJOCO_NAMES
        joint_names = SMPL_BONE_ORDER_NAMES
    elif humanoid_type == "smplx" or humanoid_type == "smplh":
        mujoco_joint_names = SMPLH_MUJOCO_NAMES
        joint_names = SMPLH_BONE_ORDER_NAMES
    else:
        raise NotImplementedError

    left_to_right_index = []
    for idx, entry in enumerate(mujoco_joint_names):
        # swap text "R_" and "L_"
        if entry.startswith("R_"):
            left_to_right_index.append(mujoco_joint_names.index("L_" + entry[2:]))
        elif entry.startswith("L_"):
            left_to_right_index.append(mujoco_joint_names.index("R_" + entry[2:]))
        else:
            left_to_right_index.append(idx)

    folder_names = [
        f.path.split("/")[-1] for f in os.scandir(amass_root_dir) if f.is_dir()
    ]

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

    smpl_local_robot = SMPL_Robot(
        robot_cfg,
        data_dir="data/smpl",
    )

    if humanoid_mjcf_path is not None:
        skeleton_tree = SkeletonTree.from_mjcf(humanoid_mjcf_path)
    else:
        skeleton_tree = None

    uuid_str = uuid.uuid4()

    # Count total number of files that need processing
    start_time = time.time()
    total_files = 0
    total_files_to_process = 0
    processed_files = 0
    for folder_name in folder_names:
        if "retarget" in folder_name or "smpl" in folder_name or "h1" in folder_name:
            continue
        data_dir = amass_root_dir / folder_name
        save_dir = output_dir / f"{folder_name}-{append_name}"

        all_files_in_folder = [
            f
            for f in Path(data_dir).glob("**/*.[np][pk][lz]")
            if (f.name != "shape.npz" and "stagei.npz" not in f.name)
        ]

        if not force_remake:
            # Only count files that don't already have outputs
            files_to_process = [
                f
                for f in all_files_in_folder
                if not (
                    save_dir
                    / f.relative_to(data_dir).parent
                    / f.name.replace(".npz", ".npy")
                    .replace(".pkl", ".npy")
                    .replace("-", "_")
                    .replace(" ", "_")
                    .replace("(", "_")
                    .replace(")", "_")
                ).exists()
            ]
        else:
            files_to_process = all_files_in_folder
        print(
            f"Processing {len(files_to_process)}/{len(all_files_in_folder)} files in {folder_name}"
        )
        total_files_to_process += len(files_to_process)
        total_files += len(all_files_in_folder)

    print(f"Total files to process: {total_files_to_process}/{total_files}")

    for folder_name in folder_names:
        if "retarget" in folder_name or "smpl" in folder_name or "h1" in folder_name:
            # Ignore folders where we store motions retargeted to AMP
            continue

        data_dir = amass_root_dir / folder_name
        save_dir = output_dir / f"{folder_name}-{append_name}"

        print(f"Processing subset {folder_name}")
        os.makedirs(save_dir, exist_ok=True)

        files = [
            f
            for f in Path(data_dir).glob("**/*.[np][pk][lz]")
            if (f.name != "shape.npz" and "stagei.npz" not in f.name)
        ]
        print(f"Processing {len(files)} files")

        files.sort()

        for filename in tqdm(files):
            try:
                relative_path_dir = filename.relative_to(data_dir).parent
                outpath = (
                    save_dir
                    / relative_path_dir
                    / filename.name.replace(".npz", ".npy")
                    .replace(".pkl", ".npy")
                    .replace("-", "_")
                    .replace(" ", "_")
                    .replace("(", "_")
                    .replace(")", "_")
                )

                # Check if the output file already exists
                if not force_remake and outpath.exists():
                    # print(f"Skipping {filename} as it already exists.")
                    continue

                # Create the output directory if it doesn't exist
                os.makedirs(save_dir / relative_path_dir, exist_ok=True)

                print(f"Processing {filename}")
                if filename.suffix == ".npz":
                    motion_data = np.load(filename)

                    betas = motion_data["betas"]
                    gender = motion_data["gender"]
                    amass_pose = motion_data["poses"]
                    amass_trans = motion_data["trans"]
                    if humanoid_type == "smplx":
                        # Load the fps from the yaml file
                        fps_yaml_path = Path("data/yaml_files/motion_fps_amassx.yaml")
                        with open(fps_yaml_path, "r") as f:
                            fps_dict = yaml.safe_load(f)

                        # Convert filename to match yaml format
                        yaml_key = (
                            folder_name
                            + "/"
                            + str(
                                relative_path_dir
                                / filename.name.replace(".npz", ".npy")
                                .replace("-", "_")
                                .replace(" ", "_")
                                .replace("(", "_")
                                .replace(")", "_")
                            )
                        )

                        if yaml_key in fps_dict:
                            mocap_fr = fps_dict[yaml_key]
                        elif "mocap_framerate" in motion_data:
                            mocap_fr = motion_data["mocap_framerate"]
                        elif "mocap_frame_rate" in motion_data:
                            mocap_fr = motion_data["mocap_frame_rate"]
                        else:
                            raise Exception(f"FPS not found for {yaml_key}")
                    else:
                        if "mocap_framerate" in motion_data:
                            mocap_fr = motion_data["mocap_framerate"]
                        else:
                            mocap_fr = motion_data["mocap_frame_rate"]
                else:
                    print(f"Skipping {filename} as it is not a valid file")
                    continue

                pose_aa = torch.tensor(amass_pose)
                amass_trans = torch.tensor(amass_trans)
                betas = torch.from_numpy(betas)

                if force_neutral_body:
                    betas[:] = 0
                    gender = "neutral"

                motion_data = {
                    "pose_aa": pose_aa.numpy(),
                    "trans": amass_trans.numpy(),
                    "beta": betas.numpy(),
                    "gender": gender,
                }

                smpl_2_mujoco = [
                    joint_names.index(q) for q in mujoco_joint_names if q in joint_names
                ]
                batch_size = motion_data["pose_aa"].shape[0]

                if humanoid_type == "smpl":
                    pose_aa = np.concatenate(
                        [motion_data["pose_aa"][:, :66], np.zeros((batch_size, 6))],
                        axis=1,
                    )  # TODO: need to extract correct handle rotations instead of zero
                    pose_aa_mj = pose_aa.reshape(batch_size, 24, 3)[:, smpl_2_mujoco]
                    pose_quat = (
                        sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
                        .as_quat()
                        .reshape(batch_size, 24, 4)
                    )
                else:
                    pose_aa = np.concatenate(
                        [
                            motion_data["pose_aa"][:, :66],
                            motion_data["pose_aa"][:, 75:],
                        ],
                        axis=-1,
                    )
                    pose_aa_mj = pose_aa.reshape(batch_size, 52, 3)[:, smpl_2_mujoco]
                    pose_quat = (
                        sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
                        .as_quat()
                        .reshape(batch_size, 52, 4)
                    )

                if isinstance(gender, np.ndarray):
                    gender = gender.item()

                if isinstance(gender, bytes):
                    gender = gender.decode("utf-8")
                if gender == "neutral":
                    gender_number = [0]
                elif gender == "male":
                    gender_number = [1]
                elif gender == "female":
                    gender_number = [2]
                else:
                    ipdb.set_trace()
                    raise Exception("Gender Not Supported!!")

                if skeleton_tree is None:
                    smpl_local_robot.load_from_skeleton(
                        betas=betas[None,], gender=gender_number, objs_info=None
                    )
                    smpl_local_robot.write_xml(
                        f"{TMP_SMPL_DIR}/smpl_humanoid_{uuid_str}.xml"
                    )
                    skeleton_tree = SkeletonTree.from_mjcf(
                        f"{TMP_SMPL_DIR}/smpl_humanoid_{uuid_str}.xml"
                    )

                root_trans_offset = (
                    torch.from_numpy(motion_data["trans"])
                    + skeleton_tree.local_translation[0]
                )

                sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here.
                    torch.from_numpy(pose_quat),
                    root_trans_offset,
                    is_local=True,
                )

                if generate_flipped:
                    formats = ["regular", "flipped"]
                else:
                    formats = ["regular"]

                for format in formats:
                    if robot_cfg["upright_start"]:
                        B = pose_aa.shape[0]
                        pose_quat_global = (
                            (
                                sRot.from_quat(
                                    sk_state.global_rotation.reshape(-1, 4).numpy()
                                )
                                * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
                            )
                            .as_quat()
                            .reshape(B, -1, 4)
                        )
                    else:
                        pose_quat_global = sk_state.global_rotation.numpy()

                    trans = root_trans_offset.clone()
                    if format == "flipped":
                        pose_quat_global = pose_quat_global[:, left_to_right_index]
                        pose_quat_global[..., 0] *= -1
                        pose_quat_global[..., 2] *= -1
                        trans[..., 1] *= -1

                    new_sk_state = SkeletonState.from_rotation_and_root_translation(
                        skeleton_tree,
                        torch.from_numpy(pose_quat_global),
                        trans,
                        is_local=False,
                    )

                    new_sk_motion = SkeletonMotion.from_skeleton_state(
                        new_sk_state, fps=mocap_fr
                    )

                    if force_retarget:
                        from data.scripts.retargeting.mink_retarget import (
                            retarget_motion,
                        )

                        target_robot_type = (
                            robot_type.split("_")[0]
                            if isinstance(robot_type, str) and robot_type.endswith("_humanoid")
                            else robot_type
                        )

                        if target_robot_type == "bio":
                            print("Force retargeting motion using pose-copy (bio, no IK)...")
                            new_sk_motion = _retarget_motion_pose_copy_bio(new_sk_motion)
                        else:
                            print("Force retargeting motion using mink retargeter...")
                            # Convert to 30 fps to speedup Mink retargeting
                            skip = int(mocap_fr // 30)
                            new_sk_state = SkeletonState.from_rotation_and_root_translation(
                                skeleton_tree,
                                torch.from_numpy(pose_quat_global[::skip]),
                                trans[::skip],
                                is_local=False,
                            )
                            new_sk_motion = SkeletonMotion.from_skeleton_state(
                                new_sk_state, fps=30
                            )

                            if robot_type in ["smpl", "smplx", "smplh"]:
                                robot_type = f"{robot_type}_humanoid"
                            new_sk_motion = retarget_motion(
                                motion=new_sk_motion, robot_type=robot_type, render=False
                            )
                        try:
                            if hasattr(new_sk_motion, "skeleton_tree"):
                                node_names = list(new_sk_motion.skeleton_tree.node_names)
                                print(f"Retargeted skeleton nodes ({len(node_names)}): {node_names[:8]} ...")
                        except Exception:
                            pass

                    if format == "flipped":
                        outpath = outpath.with_name(
                            outpath.stem + "_flipped" + outpath.suffix
                        )
                    # Use proper format per robot type
                    if robot_type in ["h1", "g1"]:
                        print(f"Saving to {outpath}")
                        os.makedirs(outpath.parent, exist_ok=True)
                        torch.save(new_sk_motion, str(outpath))
                    else:
                        # Save generic robots (e.g., bio) as poselib .npy for play/eval pipelines
                        out_npy = outpath.with_suffix(".npy")
                        print(f"Saving to {out_npy}")
                        os.makedirs(out_npy.parent, exist_ok=True)
                        new_sk_motion.to_file(str(out_npy))

                    processed_files += 1
                    elapsed_time = time.time() - start_time
                    avg_time_per_file = elapsed_time / processed_files
                    remaining_files = total_files_to_process - processed_files
                    estimated_time_remaining = avg_time_per_file * remaining_files

                    print(
                        f"\nProgress: {processed_files}/{total_files_to_process} files"
                    )
                    print(
                        f"Average time per file: {timedelta(seconds=int(avg_time_per_file))}"
                    )
                    print(
                        f"Estimated time remaining: {timedelta(seconds=int(estimated_time_remaining))}"
                    )
                    print(
                        f"Estimated completion time: {time.strftime('%H:%M:%S', time.localtime(time.time() + estimated_time_remaining))}\n"
                    )
            except Exception as e:
                print(f"Error processing {filename}")
                print(f"Error: {e}")
                print(f"Line: {e.__traceback__.tb_lineno}")
                continue


if __name__ == "__main__":
    with torch.no_grad():
        typer.run(main)
