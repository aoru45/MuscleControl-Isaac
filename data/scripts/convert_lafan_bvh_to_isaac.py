import os
import sys
import numpy as np
import torch
import argparse
import glob
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# Add repo root to path to import lafan_utils
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from data.scripts.lafan_utils import read_bvh, quat_fk, quat_mul, quat_inv

# Import poselib
try:
    from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree
except ImportError:
    print("Error: poselib not found. Please ensure it is installed and in the python path.")
    sys.exit(1)

def _as_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def save_bvh(filename, anim, channel_order='XYZ'):
    """
    Saves the animation to a BVH file.
    Matches SMPL format: Real Offsets in Hierarchy, 6-Channels (Pos+Rot), XYZ Rotation Order.
    """
    
    # Identify children for hierarchy
    children = {i: [] for i in range(len(anim.parents))}
    roots = []
    for i, p in enumerate(anim.parents):
        if p == -1:
            roots.append(i)
        else:
            children[p].append(i)
            
    # Convert quaternions to euler for motion
    quats_scipy = np.roll(anim.quats, -1, axis=-1)
    
    # Use specified order (default XYZ for SMPL)
    euler_order = channel_order.upper()
    r = R.from_quat(quats_scipy.reshape(-1, 4))
    eulers = r.as_euler(euler_order, degrees=True).reshape(anim.quats.shape[0], anim.quats.shape[1], 3)
    
    with open(filename, 'w') as f:
        f.write("HIERARCHY\n")
        
        def write_node(idx, level):
            indent = "\t" * level
            name = anim.bones[idx]
            
            # Check if this is truly an end site (no children)
            is_end_site = (len(children[idx]) == 0)
            
            if is_end_site and (name.endswith('_End') or name.startswith('End')):
                f.write(f"{indent}End Site\n")
                f.write(f"{indent}{{\n")
                f.write(f"{indent}\tOFFSET {anim.offsets[idx][0]:.6f} {anim.offsets[idx][1]:.6f} {anim.offsets[idx][2]:.6f}\n")
                f.write(f"{indent}}}\n")
            else:
                if level == 0:
                    f.write(f"{indent}ROOT {name}\n")
                else:
                    f.write(f"{indent}JOINT {name}\n")
                f.write(f"{indent}{{\n")
                f.write(f"{indent}\tOFFSET {anim.offsets[idx][0]:.6f} {anim.offsets[idx][1]:.6f} {anim.offsets[idx][2]:.6f}\n")
                
                # FORCE 6 CHANNELS (SMPL Style)
                # Position channels + Rotation channels in requested order
                rot_channels = f"{channel_order[0]}rotation {channel_order[1]}rotation {channel_order[2]}rotation"
                f.write(f"{indent}\tCHANNELS 6 Xposition Yposition Zposition {rot_channels}\n")
                
                for child_idx in children[idx]:
                    write_node(child_idx, level + 1)
                
                f.write(f"{indent}}}\n")

        for r in roots:
            write_node(r, 0)
            
        f.write("MOTION\n")
        f.write(f"Frames: {anim.quats.shape[0]}\n")
        f.write(f"Frame Time: 0.033333\n")
        
        for t in range(anim.quats.shape[0]):
            line_parts = []
            for i in range(len(anim.bones)):
                # Write Position (anim.pos contains relative offsets for children, abs pos for root)
                pos = anim.pos[t, i]
                line_parts.extend([f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}"])
                
                # Write Euler
                # Check if this node is End Site (skip Euler)
                is_end_site = len(children[i]) == 0 and (anim.bones[i].endswith('_End') or anim.bones[i].startswith('End'))
                
                if not is_end_site:
                     line_parts.extend([f"{eulers[t, i, 0]:.6f}", f"{eulers[t, i, 1]:.6f}", f"{eulers[t, i, 2]:.6f}"])
            
            f.write(" ".join(line_parts) + "\n")

def _retarget_motion_pose_copy_bio(motion: SkeletonMotion, src_tpose_root_global: np.ndarray) -> SkeletonMotion:
    """
    Retarget LaFAN (T-Pose aligned) -> BIO by directly copying (mapped) joint local rotations.
    """
    # Load Bio Skeleton (Target)
    bio_xml_path = os.path.join(repo_root, "protomotions/data/assets/mjcf/bio.xml")
    if not os.path.exists(bio_xml_path):
        raise FileNotFoundError(f"Bio XML not found at {bio_xml_path}")
        
    target_tree = SkeletonTree.from_mjcf(bio_xml_path)

    src_names = list(motion.skeleton_tree.node_names)
    tgt_names = list(target_tree.node_names)
    src_name_to_idx = {n: i for i, n in enumerate(src_names)}
    tgt_name_to_idx = {n: i for i, n in enumerate(tgt_names)}

    # LaFAN to Bio Mapping
    # Bio Joints: Pelvis, Spine, Torso, Neck, Head, ShoulderL, ArmL, ForeArmL, HandL, ...
    tgt_to_src_map = {
        "Pelvis": "Hips",
        "Spine": "Spine",
        "Torso": "Spine2", # Skipping Spine1? Or Spine2 is better match for Torso? LaFAN Spine2 is Chest level usually.
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
        # "FootThumbL": "LeftToeBase",
        # "FootPinkyL": "LeftToeBase", # Map both toes to same source
        
        "FemurR": "RightUpLeg",
        "TibiaR": "RightLeg",
        "TalusR": "RightFoot",
        # "FootThumbR": "RightToeBase",
        # "FootPinkyR": "RightToeBase",
    }

    src_local_rot = _as_numpy(motion.local_rotation).astype(np.float64) # (T, J, 4) - (x, y, z, w) expected by poselib? Yes, checking below.
    T = int(src_local_rot.shape[0])
    nb_tgt = len(tgt_names)
    
    # Target Local Rotation (Initialize with Identity 0,0,0,1)
    tgt_local_rot = np.zeros((T, nb_tgt, 4), dtype=np.float64)
    tgt_local_rot[..., 3] = 1.0 # (x, y, z, w) -> w=1

    for tgt_name, src_name in tgt_to_src_map.items():
        if tgt_name in tgt_name_to_idx and src_name in src_name_to_idx:
            ti = tgt_name_to_idx[tgt_name]
            si = src_name_to_idx[src_name]
            tgt_local_rot[:, ti] = src_local_rot[:, si]
        else:
            # if tgt_name not in tgt_name_to_idx:
            #     print(f"Warning: Target joint {tgt_name} not found in Bio skeleton.")
            pass

    # Adjust root translation
    # src_root_translation is motion.root_translation (Global translation of root)
    src_root_translation = _as_numpy(motion.root_translation).astype(np.float64)
    
    # We ignore offset logic here because we will do global rotation and height alignment next.
    tgt_root_translation = src_root_translation

    # --- Ankle Fix: Correct "High Heels" Artifact ---
    # Rotate feet UP (Dorsiflexion) by applying a local offset
    ankle_fix_euler = np.array([-60.0, 0.0, 0.0]) 
    q_ankle_fix = R.from_euler('xyz', ankle_fix_euler, degrees=True).as_quat() # (x, y, z, w)
    
    # Apply to TalusL and TalusR
    if "TalusL" in tgt_name_to_idx:
        ti = tgt_name_to_idx["TalusL"]
        q_old = R.from_quat(tgt_local_rot[:, ti])
        q_new = (q_old * R.from_quat(q_ankle_fix)).as_quat()
        tgt_local_rot[:, ti] = q_new
        # print(f"Applied Ankle Fix ({ankle_fix_euler} deg) to TalusL")

    if "TalusR" in tgt_name_to_idx:
        ti = tgt_name_to_idx["TalusR"]
        q_old = R.from_quat(tgt_local_rot[:, ti])
        q_new = (q_old * R.from_quat(q_ankle_fix)).as_quat()
        tgt_local_rot[:, ti] = q_new
        # print(f"Applied Ankle Fix ({ankle_fix_euler} deg) to TalusR")
    # ------------------------------------------------

    tgt_state = SkeletonState.from_rotation_and_root_translation(
        target_tree,
        torch.from_numpy(tgt_local_rot).float(),
        torch.from_numpy(tgt_root_translation).float(),
        is_local=True,
    )
    
    tgt_motion = SkeletonMotion.from_skeleton_state(tgt_state, fps=float(motion.fps))
    
    def _rotate_global(m: SkeletonMotion, r: R) -> SkeletonMotion:
        # Get Global Rotation and Translation
        gr = _as_numpy(m.global_rotation).astype(np.float64)
        rt = _as_numpy(m.root_translation).astype(np.float64)
        
        # Rotate Orientation (Left Multiply for Global Coordinate Transform)
        # R_new = R_transform * R_old
        gr_new = (r * R.from_quat(gr.reshape(-1, 4))).as_quat().reshape(gr.shape)
        
        # Rotate Translation
        # P_new = R_transform * P_old
        rt_new = r.apply(rt)
        
        st = SkeletonState.from_rotation_and_root_translation(
            m.skeleton_tree,
            torch.from_numpy(gr_new).float(),
            torch.from_numpy(rt_new).float(),
            is_local=False,
        )
        return SkeletonMotion.from_skeleton_state(st, fps=float(m.fps))

    # Define Y-up to Z-up Rotation (Rotate +90 deg around X-axis)
    # (x, y, z) -> (x, -z, y)
    r_upright = R.from_euler('x', 90, degrees=True)
    
    # 1. Rotate Motion to Z-up
    tgt_motion = _rotate_global(tgt_motion, r_upright)
    
    # 2. Align Height (Root Offset) - DELTA LOGIC RESTORED
    # Source Rest Root (Y-up) -> Z-up
    src_rest_zup = r_upright.apply(src_tpose_root_global)
    
    # Target Rest Root (Z-up)
    tgt_rest_zup = _as_numpy(target_tree.local_translation[0]).astype(np.float64)
    
    # Calculate Delta
    delta = tgt_rest_zup - src_rest_zup
    
    # print(f"Applying Root Delta: {delta}")
    
    # Apply Delta
    curr_rt = _as_numpy(tgt_motion.root_translation)
    new_rt = curr_rt + delta
    
    final_state = SkeletonState.from_rotation_and_root_translation(
        tgt_motion.skeleton_tree,
        tgt_motion.global_rotation, # Already rotated
        torch.from_numpy(new_rt).float(),
        is_local=False
    )
    
    return SkeletonMotion.from_skeleton_state(final_state, fps=float(motion.fps))


def convert_lafan_to_bio_npy(input_path, output_path):
    # print(f"Reading {input_path}...")
    anim = read_bvh(input_path)
    
    # -------------------------------------------------------------------------
    # PART 1: T-Pose Alignment
    # -------------------------------------------------------------------------
    # print("Aligning to T-Pose (In-Memory)...")
    
    # 1. Identify Target Joints
    try:
        r_shoulder_idx = anim.bones.index("RightShoulder")
        l_shoulder_idx = anim.bones.index("LeftShoulder")
    except ValueError:
        print(f"Skipping {input_path}: RightShoulder or LeftShoulder not found.")
        return

    num_joints = len(anim.bones)
    
    # 2. Define T-Pose Adjustments
    q_adjust_all = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (num_joints, 1))
    
    r_rot = R.from_euler('z', 90, degrees=True).as_quat() # (x, y, z, w)
    q_adjust_all[r_shoulder_idx] = np.array([r_rot[3], r_rot[0], r_rot[1], r_rot[2]]) # w,x,y,z
    
    l_rot = R.from_euler('z', -90, degrees=True).as_quat()
    q_adjust_all[l_shoulder_idx] = np.array([l_rot[3], l_rot[0], l_rot[1], l_rot[2]])
    
    # 3. Compute T-Pose Global Transforms
    local_q_tpose = q_adjust_all[np.newaxis, ...] 
    local_p_tpose = anim.offsets[np.newaxis, ...] 
    global_q_tpose, global_p_tpose = quat_fk(local_q_tpose, local_p_tpose, anim.parents)
    
    # SCALE T-Pose Global (cm -> m)
    global_p_tpose *= 0.01
    
    # 4. Define New Hierarchy Offsets
    new_offsets = np.zeros_like(anim.offsets)
    new_offsets[0] = 0.0 # Force zero root offset for new skeleton
    
    for i in range(1, num_joints):
        parent = anim.parents[i]
        raw_offset = global_p_tpose[0, i] - global_p_tpose[0, parent]
        raw_offset[np.abs(raw_offset) < 1e-4] = 0.0
        new_offsets[i] = raw_offset
        
    # 5. Convert Animation Data
    old_local_p = anim.pos.copy()
    root_offset = anim.offsets[0]
    old_local_p[:, 0] += root_offset
    
    old_global_q, old_global_p = quat_fk(anim.quats, old_local_p, anim.parents)
    
    # SCALE Global Pos (cm -> m)
    old_global_p *= 0.01
    
    inv_tpose_q = quat_inv(global_q_tpose)
    new_global_q = quat_mul(old_global_q, inv_tpose_q)
    new_global_p = old_global_p.copy() # Global positions match original
    
    # IK to get new local rotations
    from data.scripts.lafan_utils import quat_ik
    new_local_q, _ = quat_ik(new_global_q, new_global_p, anim.parents)
    
    # Construct "final_pos" for SkeletonMotion
    
    # -------------------------------------------------------------------------
    # PART 2: Convert to poselib SkeletonMotion
    # -------------------------------------------------------------------------
    # print("Constructing Source SkeletonMotion...")
    
    # Create Source SkeletonTree
    src_local_translation = torch.from_numpy(new_offsets).float()
    
    # Parent indices
    src_parent_indices = torch.tensor(anim.parents, dtype=torch.long)
    
    # Node names
    src_node_names = anim.bones
    
    src_skeleton_tree = SkeletonTree(src_node_names, src_parent_indices, src_local_translation)
    
    # Create Source SkeletonState
    new_local_q_xyzw = np.roll(new_local_q, -1, axis=-1)
    src_local_rotation = torch.from_numpy(new_local_q_xyzw).float()
    
    # Root Translation
    src_root_translation = torch.from_numpy(new_global_p[:, 0]).float()
    
    src_state = SkeletonState.from_rotation_and_root_translation(
        src_skeleton_tree,
        src_local_rotation,
        src_root_translation,
        is_local=True
    )
    
    src_motion = SkeletonMotion.from_skeleton_state(src_state, fps=float(anim.fps))
    
    # -------------------------------------------------------------------------
    # PART 3: Retarget to Bio
    # -------------------------------------------------------------------------
    # print("Retargeting to Bio (Pose Copy)...")
    
    new_motion = _retarget_motion_pose_copy_bio(src_motion, global_p_tpose[0, 0])


    fps_src = round(anim.fps)
    target_fps = 30
    if fps_src != target_fps:
        print(f"Resampling from {fps_src}Hz to {target_fps}Hz...")
        # Simple slicing if multiple
        if fps_src % target_fps == 0:
            skip = int(fps_src / target_fps)
            sliced_state = SkeletonState.from_rotation_and_root_translation(
                new_motion.skeleton_tree,
                new_motion.global_rotation[::skip],
                new_motion.root_translation[::skip],
                is_local=False
            )
            new_motion = SkeletonMotion.from_skeleton_state(sliced_state, fps=target_fps)
        else:
            print(f"Warning: FPS {fps_src} is not a multiple of {target_fps}. Using nearest neighbor slicing.")
            # Nearest neighbor
            indices = np.round(np.linspace(0, T - 1, int(T * target_fps / fps_src))).astype(int)
            sliced_state = SkeletonState.from_rotation_and_root_translation(
                new_motion.skeleton_tree,
                new_motion.global_rotation[indices],
                new_motion.root_translation[indices],
                is_local=False
            )
            new_motion = SkeletonMotion.from_skeleton_state(sliced_state, fps=target_fps)
    else:
        new_motion = SkeletonMotion.from_skeleton_state(new_motion, fps=target_fps)
    
    # -------------------------------------------------------------------------
    # PART 4: Save to NPY
    # -------------------------------------------------------------------------
    # print(f"Saving to {output_path}...")
    new_motion.to_file(output_path)
    # print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input BVH file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output NPY file or directory")
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        # Batch Mode
        print(f"Batch processing BVH files from {args.input}...")
        files = glob.glob(os.path.join(args.input, "**/*.bvh"), recursive=True)
        print(f"Found {len(files)} BVH files.")
        
        for f in tqdm(files):
            rel_path = os.path.relpath(f, args.input)
            # Replace extension and join with output dir
            out_f = os.path.join(args.output, os.path.splitext(rel_path)[0] + "_bio.npy")
            
            os.makedirs(os.path.dirname(out_f), exist_ok=True)
            
            try:
                convert_lafan_to_bio_npy(f, out_f)
            except Exception as e:
                print(f"Failed to process {f}: {e}")
                
    else:
        # Single File Mode
        convert_lafan_to_bio_npy(args.input, args.output)
