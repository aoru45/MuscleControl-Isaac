import os
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import numpy as np
import torch
import typer
from pathlib import Path
from scipy.spatial.transform import Rotation as sRot
import mujoco
import mink
from tqdm import tqdm
import xml.etree.ElementTree as ET
from data.scripts.retargeting.mink_retarget import construct_model

# Import necessary classes from poselib
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree


# Mapping from Bio skeleton to LAFAN1/Nokov skeleton
_BIO_TO_LAFAN_MAP = {
    "Pelvis": "Hips",
    "Spine": "Spine",
    "Torso": "Spine2",  # Prefer Spine2, fallback to Spine1
    "Neck": "Neck",
    "Head": "Head",
    "ShoulderL": "LeftShoulder",
    "ArmL": "LeftArm",
    "ForeArmL": "LeftForeArm",
    "HandL": "LeftHand",
    "ShoulderR": "RightShoulder",
    "ArmR": "RightArm",
    "ForeArmR": "RightForeArm",
    "HandR": "RightHand",
    "FemurL": "LeftUpLeg",
    "TibiaL": "LeftLeg",
    "TalusL": "LeftFoot",
    "FootThumbL": "LeftToeBase",
    "FootPinkyL": "LeftToeBase",
    "FemurR": "RightUpLeg",
    "TibiaR": "RightLeg",
    "TalusR": "RightFoot",
    "FootThumbR": "RightToeBase",
    "FootPinkyR": "RightToeBase",
}



def parse_bvh(bvh_path, verbose=False):
    if verbose:
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

    if verbose:
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



def retarget_bvh_to_bio(motion: SkeletonMotion, output_path: str, render: bool = False, scale: float = 0.01, target_fps: int = 30, rot_fix: bool = True, show_progress: bool = True, verbose: bool = False):
    robot_name = "bio"
    fps = motion.fps
    _RETARGET_WEIGHTS = {
        "Head": 3.0,
        "Pelvis": 2.0,
        "FemurL": 1.0,
        "FemurR": 1.0,
        "TibiaL": 1.0,
        "TibiaR": 1.0,
        "TalusL": 3.0,
        "TalusR": 3.0,
        "FootThumbL": 3.0,
        "FootThumbR": 3.0,
        "ArmL": 1.5,
        "ArmR": 1.5,
        "HandL": 3.0,
        "HandR": 3.0,
        "ShoulderL": 0.,
        "ShoulderR": 0.,
        "Torso": 1.5,
        "Spine": 1.5,
        "Neck": 1.5,
        "ForeArmL": 1.5,
        "ForeArmR": 1.5,
        "FootPinkyL": 3.0,
        "FootPinkyR": 3.0,
    }
    
    
    # Get Global positions and rotations from Source Motion
    global_translations = motion.global_translation.numpy()
    pose_quat_global = motion.global_rotation.numpy()
    
    # Apply Scaling (e.g. cm to m)
    global_translations *= scale
    
    # Apply Coordinate System Transformation (Y-up to Z-up)
    # Rotate 90 degrees around X-axis
    # (x, y, z) -> (x, -z, y)
    if verbose:
        print("Applying Y-up to Z-up transformation...")
    r_fix = sRot.from_euler('x', 90, degrees=True) 
    
    # Transform Translations
    # Shape: (T, J, 3)
    orig_shape = global_translations.shape
    flat_trans = global_translations.reshape(-1, 3)
    flat_trans = r_fix.apply(flat_trans)
    global_translations = flat_trans.reshape(orig_shape)
    
    # Transform Rotations
    # Shape: (T, J, 4)
    orig_rot_shape = pose_quat_global.shape
    flat_rot = pose_quat_global.reshape(-1, 4)
    # R_new = R_fix * R_old
    r_old = sRot.from_quat(flat_rot)
    r_new = r_fix * r_old
    pose_quat_global = r_new.as_quat().reshape(orig_rot_shape)
    
    # --- Fix Ground Intersection (Pre-process) ---
    source_joint_names = motion.skeleton_tree.node_names
    # Identify foot joints to determine ground level
    # We use the mapping to find source joints corresponding to feet
    feet_keys = ["TalusL", "TalusR", "FootThumbL", "FootThumbR"]
    src_feet_names = []
    for k in feet_keys:
        if k in _BIO_TO_LAFAN_MAP:
             src_feet_names.append(_BIO_TO_LAFAN_MAP[k])
             
    valid_src_feet = [n for n in src_feet_names if n in source_joint_names]
    
    if valid_src_feet:
        feet_indices = [source_joint_names.index(n) for n in valid_src_feet]
        # Calculate min Z across all frames and foot joints
        min_z = global_translations[:, feet_indices, 2].min()
        
        # Target ground clearance (approximate foot radius/sole thickness)
        ground_clearance = 0.02
        
        offset_z = min_z - ground_clearance
        if verbose:
            print(f"Pre-adjusting height by {-offset_z:.4f} m to fix ground intersection (Min Z: {min_z:.4f} -> {ground_clearance})")
        global_translations[..., 2] -= offset_z
    # ---------------------------------------------
    
    timeseries_length = global_translations.shape[0]
    
    # Identify which joints in BVH correspond to which bodies in Bio
    
    # Create the Bio model
    # We need to pass the list of Source Keys (BVH joint names) that we want to visualize/track
    # But construct_model adds sites for them.
    
    source_joint_names = motion.skeleton_tree.node_names
    
    # We only care about the joints that are in our map values
    mapped_source_joints = list(set(_BIO_TO_LAFAN_MAP.values()))
    # Filter out joints that don't exist in the actual BVH
    valid_source_joints = [name for name in mapped_source_joints if name in source_joint_names]

    if verbose:
        print(f"Constructing model with keypoints: {valid_source_joints}")
    model = construct_model(robot_name, valid_source_joints)
    configuration = mink.Configuration(model)
    
    tasks = []
    
    # Setup Tasks
    for bio_body, bvh_joint in _BIO_TO_LAFAN_MAP.items():
        if bvh_joint not in source_joint_names:
            if verbose:
                print(f"Warning: Source joint {bvh_joint} not found in BVH. Skipping task for {bio_body}")
            continue
        orientation_cost = 0.
        task = mink.FrameTask(
            frame_name=bio_body,
            frame_type="body",
            position_cost=10. * _RETARGET_WEIGHTS[bio_body],
            orientation_cost=orientation_cost,
            lm_damping=1.0,
        )
        tasks.append(task)
        
    posture_task = mink.PostureTask(model, cost=1.0)
    tasks.append(posture_task)
    
    # Prepare MuJoCo data
    data = configuration.data
    
    # Render context
    if render:
        viewer_context = mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
        )
    else:
        from contextlib import nullcontext
        viewer_context = nullcontext()
        
    retargeted_poses = []
    retargeted_trans = []
    
    with viewer_context as viewer:
        # Initialize
        data.qpos[:] = 0
        # Set root to first frame root
        # Bio root is Pelvis
        # Source root is Hips
        # We can map them roughly
        hips_idx = source_joint_names.index("Hips") if "Hips" in source_joint_names else 0
        
        data.qpos[0:3] = global_translations[0, hips_idx]
        data.qpos[3:7] = pose_quat_global[0, hips_idx]
        
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        posture_task.set_target_from_configuration(configuration)
        
        optimization_steps_per_frame = 8
        try:
            from loop_rate_limiters import RateLimiter
            rate = RateLimiter(frequency=fps * optimization_steps_per_frame)
        except ImportError:
            # Fallback or simple sleep if loop_rate_limiters is not available
            import time
            class RateLimiter:
                def __init__(self, frequency):
                    self.dt = 1.0 / frequency
                def sleep(self):
                    time.sleep(self.dt)
            rate = RateLimiter(frequency=fps * optimization_steps_per_frame)
            
        solver = "daqp"
        
        pbar = tqdm(total=timeseries_length, desc="Retargeting", disable=not show_progress)
        
        for t in range(timeseries_length):
            # Update targets
            task_idx = 0
            for bio_body, bvh_joint in _BIO_TO_LAFAN_MAP.items():
                if bvh_joint not in source_joint_names:
                    continue
                
                bvh_idx = source_joint_names.index(bvh_joint)
                
                target_pos = global_translations[t, bvh_idx].copy()
                target_rot = pose_quat_global[t, bvh_idx].copy()
                
                rot_matrix = sRot.from_quat(target_rot).as_matrix()
                rot = mink.SO3.from_matrix(rot_matrix)
                tasks[task_idx].set_target(
                    mink.SE3.from_rotation_and_translation(rot, target_pos)
                )
                task_idx += 1
                
                # Update mocap sites for visualization
                try:
                    mid = model.body(f"keypoint_{bvh_joint}").mocapid[0]
                    data.mocap_pos[mid] = target_pos
                except:
                    pass

            # Solve IK
            for _ in range(optimization_steps_per_frame):
                vel = mink.solve_ik(
                    configuration, tasks, rate.dt, solver, 1e-1, limits=[mink.ConfigurationLimit(model)]
                )
                configuration.integrate_inplace(vel, rate.dt)
                
                if render and viewer.is_running():
                    viewer.sync()
                    
            retargeted_poses.append(data.qpos[7:].copy())
            retargeted_trans.append(data.qpos[:7].copy())
            
            pbar.update(1)
            
        pbar.close()
        
    # Create SkeletonMotion for Bio
    retargeted_poses = np.stack(retargeted_poses)
    retargeted_trans = np.stack(retargeted_trans)
    
    # Create SkeletonMotion from MuJoCo data
    skeleton_tree = SkeletonTree.from_mjcf("protomotions/data/assets/mjcf/bio.xml")
    
    # Map Bio nodes to MuJoCo body ids
    body_ids = []
    for name in skeleton_tree.node_names:
        try:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        except:
            bid = -1
        body_ids.append(bid)
        
    T = retargeted_poses.shape[0]
    nb = len(body_ids)
    pose_quat_global_new = np.zeros((T, nb, 4), dtype=np.float64)
    global_pos_new = np.zeros((T, nb, 3), dtype=np.float64)

    if verbose:
        print("Computing FK for retargeted motion...")
    for t in range(T):
        data.qpos[:] = 0
        data.qpos[0:7] = retargeted_trans[t]
        data.qpos[7:7+retargeted_poses.shape[1]] = retargeted_poses[t]
        mujoco.mj_forward(model, data)
        
        for i, bid in enumerate(body_ids):
            if bid >= 0:
                wxyz = data.xquat[bid]
                pose_quat_global_new[t, i] = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
                global_pos_new[t, i] = data.xpos[bid]
                
    # Height adjustment (optional, but good)
    # Match lowest foot height
    # Source feet: LeftFoot, RightFoot
    src_feet = [n for n in ["LeftFoot", "RightFoot"] if n in source_joint_names]
    src_feet_idx = [source_joint_names.index(n) for n in src_feet]
    
    tgt_feet = ["TalusL", "TalusR"]
    tgt_feet_idx = []
    for n in tgt_feet:
        try:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            tgt_feet_idx.append(bid)
        except:
            pass
            
    if src_feet_idx and tgt_feet_idx:
        # Calculate lowest foot height in original motion (after scaling and Y-up to Z-up)
        # We need to find the minimum Z value across all foot joints and all frames
        orig_lowest = global_translations[:, src_feet_idx, 2].min()
        
        # Calculate lowest foot height in retargeted motion
        # We need the indices in the new skeleton corresponding to tgt_feet
        tgt_feet_skel_idx = [skeleton_tree.node_names.index(n) for n in tgt_feet if n in skeleton_tree.node_names]
        tgt_lowest = global_pos_new[:, tgt_feet_skel_idx, 2].min()
        
        # Calculate offset needed to bring retargeted motion to match original ground level
        # However, we often want the character to be ON the ground (Z=0), not necessarily matching the original BVH's absolute Z if it was floating or sinking.
        # If the original BVH had feet at Z=0, then orig_lowest ~ 0.
        # If the retargeted character is floating, tgt_lowest > 0.
        # If the retargeted character is sinking, tgt_lowest < 0.
        
        # Strategy: Shift the retargeted motion so its lowest foot point is at Z=0 (or slightly above for safety)
        # This ignores orig_lowest, assuming we want to enforce ground contact.
        
        offset = tgt_lowest
        if verbose:
            print(f"Adjusting height by {-offset:.4f} (Original lowest: {orig_lowest:.4f}, Target lowest: {tgt_lowest:.4f})")
        
        global_pos_new[..., 2] -= offset
        
        # Also update the root translation in retargeted_trans so if we reconstruct it stays correct?
        # Actually we are rebuilding SkeletonState from global_pos_new, so we are good.
        
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat_global_new),
        torch.from_numpy(global_pos_new[:, 0]), # Root pos (Pelvis is 0)
        is_local=False
    )
    
    # Resample to 30Hz if needed
    if fps != target_fps:
        if verbose:
            print(f"Resampling from {fps}Hz to {target_fps}Hz...")
        new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)
        # Calculate skip factor
        # If fps is a multiple of target_fps, we can slice.
        # Otherwise we might need interpolation. poselib's SkeletonMotion doesn't have a generic resample.
        # But we can try using the crop/slice logic if it's an integer multiple.
        
        # Simple slicing if multiple
        if fps % target_fps == 0:
            skip = int(fps / target_fps)
            # Slice the underlying tensor
            # SkeletonMotion stores data in .tensor property (N, D)
            # We can just recreate from sliced state
            sliced_state = SkeletonState.from_rotation_and_root_translation(
                new_sk_state.skeleton_tree,
                new_sk_state.global_rotation[::skip],
                new_sk_state.root_translation[::skip],
                is_local=False
            )
            new_motion = SkeletonMotion.from_skeleton_state(sliced_state, fps=target_fps)
        else:
            if verbose:
                print(f"Warning: FPS {fps} is not a multiple of {target_fps}. Using nearest neighbor slicing.")
            # Nearest neighbor
            indices = np.round(np.linspace(0, T - 1, int(T * target_fps / fps))).astype(int)
            sliced_state = SkeletonState.from_rotation_and_root_translation(
                new_sk_state.skeleton_tree,
                new_sk_state.global_rotation[indices],
                new_sk_state.root_translation[indices],
                is_local=False
            )
            new_motion = SkeletonMotion.from_skeleton_state(sliced_state, fps=target_fps)
    else:
        new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)

    if verbose:
        print(f"Saving to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_motion.to_file(output_path)
   

def main(
    bvh_path: str,
    output_path="data/output/dance1_subject2.npy",
    target_fps=30,
    render=False,
    scale = 0.01,
    show_progress=True,
    verbose=False,
):

    skeleton_tree, motion_data, channels_map, fps = parse_bvh(bvh_path, verbose=verbose)
    motion = bvh_to_skeleton_motion(skeleton_tree, motion_data, channels_map, fps)
    retarget_bvh_to_bio(motion, output_path, render=render, scale=scale, target_fps=target_fps, show_progress=show_progress, verbose=verbose)
   
   


def process_single_file(args):
    """Worker function for multiprocessing."""
    bvh_file, output_path, target_fps = args
    try:
        main(
            bvh_path=bvh_file,
            output_path=output_path,
            target_fps=target_fps,
            show_progress=False,  # Disable per-file IK progress
        )
        return True, os.path.basename(bvh_file), None
    except Exception as e:
        return False, os.path.basename(bvh_file), str(e)


if __name__ == "__main__":
    import glob
    from multiprocessing import Pool, cpu_count, Manager
    import time

    bvh_files = glob.glob("/media/lab4dv/备份/ubisoft-laforge-animation-dataset/bvh/*.bvh")

    # Prepare arguments for each file
    task_args = []
    for bvh_file in bvh_files:
        output_path = os.path.join("data/motions_lafan_bio", os.path.basename(bvh_file).replace(".bvh", ".npy"))
        task_args.append((bvh_file, output_path, 30))

    # Process with multiprocessing
    num_workers = max(1, cpu_count() - 1)  # Use all CPUs except one

    with Pool(num_workers) as pool:
        pbar = tqdm(total=len(task_args), desc="BVH Processing", unit="file")

        for success, filename, error in pool.imap_unordered(process_single_file, task_args):
            if success:
                print(f"✓ {filename}")
            else:
                print(f"✗ {filename}: {error}")
            pbar.update(1)

        pbar.close()

    print(f"\nCompleted processing {len(task_args)} files!")

