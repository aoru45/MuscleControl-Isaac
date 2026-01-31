import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import typer
from pathlib import Path
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

# Import necessary classes from poselib
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree

app = typer.Typer()

# Define the mapping from Bio (Target) to MASS BVH (Source)
# Target Body Name -> Source Joint Name
# Note: Adjust these names based on your specific BVH file if needed.
_BIO_TO_MASS_MAP = {
    "Pelvis": "Character1_Hips",
    "Spine": "Character1_Spine",
    "Torso": "Character1_Spine1",
    "Neck": "Character1_Neck",
    "Head": "Character1_Head",
    "ShoulderL": "Character1_LeftShoulder",
    "ArmL": "Character1_LeftArm",
    "ForeArmL": "Character1_LeftForeArm",
    "HandL": "Character1_LeftHand",
    "ShoulderR": "Character1_RightShoulder",
    "ArmR": "Character1_RightArm",
    "ForeArmR": "Character1_RightForeArm",
    "HandR": "Character1_RightHand",
    "FemurL": "Character1_LeftUpLeg",
    "TibiaL": "Character1_LeftLeg",
    "TalusL": "Character1_LeftFoot",
    "FootThumbL": "Character1_LeftToeBase",
    "FootPinkyL": "Character1_LeftToeBase",
    "FemurR": "Character1_RightUpLeg",
    "TibiaR": "Character1_RightLeg",
    "TalusR": "Character1_RightFoot",
    "FootThumbR": "Character1_RightToeBase",
    "FootPinkyR": "Character1_RightToeBase",
}

def parse_bvh(bvh_path):
    print(f"Parsing BVH file: {bvh_path}")
    with open(bvh_path, 'r') as f:
        content = f.read().splitlines()
    
    node_names = []
    parent_indices = []
    local_translations = [] # OFFSETS
    
    node_stack = []
    
    iterator = iter(content)
    line = next(iterator).strip()
    while line == "":
        line = next(iterator).strip()

    if line != "HIERARCHY":
        raise ValueError("Invalid BVH file: HIERARCHY not found")
    
    channels_map = {} # node_index -> list of channels
    
    current_index = -1
    
    # We need to track names to index mapping to handle parents
    
    for line in iterator:
        line = line.strip()
        if line == "MOTION":
            break
        
        parts = line.split()
        if not parts:
            continue
            
        token = parts[0]
        
        if token == "ROOT" or token == "JOINT":
            name = parts[1]
            node_names.append(name)
            current_index += 1
            parent_idx = node_stack[-1] if node_stack else -1
            parent_indices.append(parent_idx)
            node_stack.append(current_index)
            
        elif token == "End":
            # End Site
            name = node_names[node_stack[-1]] + "_End"
            node_names.append(name)
            current_index += 1
            parent_idx = node_stack[-1]
            parent_indices.append(parent_idx)
            node_stack.append(current_index)
            
        elif token == "}":
            node_stack.pop()
            
        elif token == "OFFSET":
            offset = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            local_translations.append(offset)
            
        elif token == "CHANNELS":
            node_idx = node_stack[-1]
            num_channels = int(parts[1])
            channels = parts[2:]
            channels_map[node_idx] = channels
            
    # Read Motion
    frames_line = next(iterator).strip()
    while not frames_line.startswith("Frames:"):
        frames_line = next(iterator).strip()
        
    num_frames = int(frames_line.split()[1])
    
    frame_time_line = next(iterator).strip()
    frame_time = float(frame_time_line.split()[2])
    fps = int(round(1.0 / frame_time))
    
    print(f"FPS: {fps}, Frames: {num_frames}")
    
    motion_data = []
    for line in iterator:
        line = line.strip()
        if not line:
            continue
        try:
            parts = line.split()
            # Handle cases where multiple lines might be used or extra whitespace
            motion_data.append([float(x) for x in parts])
        except ValueError:
            continue
        
    motion_data = np.array(motion_data, dtype=np.float32)
    
    parent_indices = torch.tensor(parent_indices, dtype=torch.long)
    local_translations = torch.tensor(np.array(local_translations), dtype=torch.float32)
    
    skeleton_tree = SkeletonTree(node_names, parent_indices, local_translations)
    
    return skeleton_tree, motion_data, channels_map, fps

def bvh_to_skeleton_motion(skeleton_tree, motion_data, channels_map, fps):
    num_frames = motion_data.shape[0]
    num_joints = skeleton_tree.num_joints
    
    # Local rotation (quaternion) and Root translation
    local_rotation = torch.zeros((num_frames, num_joints, 4), dtype=torch.float32)
    local_rotation[..., 3] = 1.0 # Identity quaternion (x, y, z, w) = (0, 0, 0, 1)
    
    root_translation = torch.zeros((num_frames, 3), dtype=torch.float32)
    
    col_idx = 0
    
    # Iterate through joints in the order they appear in the file (which matches node_names order usually, but channels are mapped by index)
    # The motion data columns correspond to the order of channels encountered in the HIERARCHY section.
    # So we need to iterate joints in order of definition (0 to num_joints-1)
    
    for i in range(num_joints):
        if i not in channels_map:
            continue
            
        channels = channels_map[i]
        
        rot_order = ""
        euler_angles = []
        
        for ch in channels:
            if ch == "Xposition":
                if i == 0:
                    root_translation[:, 0] = torch.from_numpy(motion_data[:, col_idx])
                col_idx += 1
            elif ch == "Yposition":
                if i == 0:
                    root_translation[:, 1] = torch.from_numpy(motion_data[:, col_idx])
                col_idx += 1
            elif ch == "Zposition":
                if i == 0:
                    root_translation[:, 2] = torch.from_numpy(motion_data[:, col_idx])
                col_idx += 1
            elif ch == "Xrotation":
                rot_order += "x"
                euler_angles.append(motion_data[:, col_idx])
                col_idx += 1
            elif ch == "Yrotation":
                rot_order += "y"
                euler_angles.append(motion_data[:, col_idx])
                col_idx += 1
            elif ch == "Zrotation":
                rot_order += "z"
                euler_angles.append(motion_data[:, col_idx])
                col_idx += 1
        
        if rot_order:
            euler_vals = np.stack(euler_angles, axis=1)
            # Convert degrees to radians
            # scipy Rotation expects degrees=True if specified
            rot = sRot.from_euler(rot_order.upper(), euler_vals, degrees=True)
            quat = rot.as_quat() # (x, y, z, w)
            local_rotation[:, i] = torch.from_numpy(quat)
            
    # Add initial offset to root translation if it's not 0 in SkeletonTree
    # Actually, BVH motion data usually contains absolute root position (if Xposition etc are present).
    # If not present, it uses offset.
    # SkeletonTree local_translation[0] is the root offset from world origin (usually 0).
    
    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        local_rotation,
        root_translation,
        is_local=True
    )
    
    return SkeletonMotion.from_skeleton_state(sk_state, fps)

def retarget_bvh_to_bio(motion: SkeletonMotion, output_path: str, render: bool = False, scale: float = 0.01, target_fps: int = 30):
    del render
    fps = motion.fps

    # Pose-copy retargeting: directly apply (mapped) global rotations to BIO, no IK.
    target_tree = SkeletonTree.from_mjcf("protomotions/data/assets/mjcf/bio.xml")
    src_names = list(motion.skeleton_tree.node_names)
    tgt_names = list(target_tree.node_names)
    src_name_to_idx = {n: i for i, n in enumerate(src_names)}
    tgt_name_to_idx = {n: i for i, n in enumerate(tgt_names)}

    print("Applying Y-up to Z-up transformation...")
    r_fix = sRot.from_euler("x", 90, degrees=True)

    src_global_rot = motion.global_rotation.detach().cpu().numpy().astype(np.float64)
    T = int(src_global_rot.shape[0])
    src_global_rot_zup = (
        (r_fix * sRot.from_quat(src_global_rot.reshape(-1, 4)))
        .as_quat()
        .reshape(src_global_rot.shape)
    )

    src_pelvis_name = _BIO_TO_MASS_MAP.get("Pelvis", src_names[0] if src_names else "")
    src_pelvis_idx = src_name_to_idx.get(src_pelvis_name, 0)
    src_pelvis_pos = (
        motion.global_translation.detach().cpu().numpy()[:, src_pelvis_idx].astype(np.float64)
        * float(scale)
    )
    tgt_root_translation = r_fix.apply(src_pelvis_pos)

    tgt_global_rot = np.zeros((T, len(tgt_names), 4), dtype=np.float64)
    tgt_global_rot[..., 3] = 1.0
    for tgt_joint, src_joint in _BIO_TO_MASS_MAP.items():
        if tgt_joint not in tgt_name_to_idx or src_joint not in src_name_to_idx:
            continue
        ti = tgt_name_to_idx[tgt_joint]
        si = src_name_to_idx[src_joint]
        tgt_global_rot[:, ti] = src_global_rot_zup[:, si]

    # --- Ankle Fix: Correct "Tiptoe" Artifact ---
    # Apply local dorsiflexion (rotate foot UP around X-axis)
    ankle_fix_euler = np.array([-30.0, 0.0, 0.0])
    q_ankle_fix = sRot.from_euler('xyz', ankle_fix_euler, degrees=True)
    
    for foot_name in ["TalusL", "TalusR"]:
        if foot_name in tgt_name_to_idx:
            ti = tgt_name_to_idx[foot_name]
            # q_global_new = q_global_old * q_local_fix (apply fix in local frame)
            q_old = sRot.from_quat(tgt_global_rot[:, ti])
            q_new = (q_old * q_ankle_fix).as_quat()
            tgt_global_rot[:, ti] = q_new
            print(f"Applied Ankle Fix ({ankle_fix_euler} deg) to {foot_name}")
    # --------------------------------------------

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        target_tree,
        torch.from_numpy(tgt_global_rot).float(),
        torch.from_numpy(tgt_root_translation).float(),
        is_local=False,
    )

    if fps != target_fps:
        print(f"Resampling from {fps}Hz to {target_fps}Hz...")
        new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)
        if fps % target_fps == 0:
            skip = int(fps / target_fps)
            sliced_state = SkeletonState.from_rotation_and_root_translation(
                new_sk_state.skeleton_tree,
                new_sk_state.global_rotation[::skip],
                new_sk_state.root_translation[::skip],
                is_local=False,
            )
            new_motion = SkeletonMotion.from_skeleton_state(sliced_state, fps=target_fps)
        else:
            print(
                f"Warning: FPS {fps} is not a multiple of {target_fps}. Using nearest neighbor slicing."
            )
            indices = np.round(np.linspace(0, T - 1, int(T * target_fps / fps))).astype(int)
            sliced_state = SkeletonState.from_rotation_and_root_translation(
                new_sk_state.skeleton_tree,
                new_sk_state.global_rotation[indices],
                new_sk_state.root_translation[indices],
                is_local=False,
            )
            new_motion = SkeletonMotion.from_skeleton_state(sliced_state, fps=target_fps)
    else:
        new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)

    ground_clearance = 0.02
    z = new_motion.global_translation[..., 2]
    min_z = float(z.min().item())
    if np.isfinite(min_z) and min_z < ground_clearance:
        dz = float(ground_clearance - min_z)
        lifted_root = new_motion.root_translation.clone()
        lifted_root[:, 2] += dz
        lifted_state = SkeletonState.from_rotation_and_root_translation(
            new_motion.skeleton_tree,
            new_motion.local_rotation.clone(),
            lifted_root,
            is_local=True,
        )
        new_motion = SkeletonMotion.from_skeleton_state(lifted_state, fps=new_motion.fps)

    print(f"Saving to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_motion.to_file(output_path)
    return
    


def process_single_bvh(bvh_path: str, output_path: str, render: bool, scale: float, target_fps: int):
    try:
        skeleton_tree, motion_data, channels_map, fps = parse_bvh(bvh_path)
        motion = bvh_to_skeleton_motion(skeleton_tree, motion_data, channels_map, fps)
        retarget_bvh_to_bio(motion, output_path, render=render, scale=scale, target_fps=target_fps)
    except Exception as e:
        print(f"Failed to process {bvh_path}: {e}")

@app.command()
def main(
    bvh_path: str,
    output_path: str,
    render: bool = False,
    scale: float = 0.01,
    target_fps: int = 30
):
    """
    Convert BVH file(s) to Isaac Gym motion format (bio.xml compatible).
    Supports single file or directory (recursive).
    """
    input_p = Path(bvh_path)
    output_p = Path(output_path)

    if input_p.is_dir():
        # Recursive processing
        bvh_files = sorted(list(input_p.rglob("*.bvh")))
        print(f"Found {len(bvh_files)} BVH files in {input_p}")
        
        # We assume output_path is a directory
        
        for bvh_file in tqdm(bvh_files, desc="Processing BVH files"):
            # Determine output path
            rel_path = bvh_file.relative_to(input_p)
            target_file = output_p / rel_path.with_suffix(".npy")
            
            process_single_bvh(str(bvh_file), str(target_file), render, scale, target_fps)
    else:
        # Single file
        process_single_bvh(bvh_path, output_path, render, scale, target_fps)

if __name__ == "__main__":
    app()
