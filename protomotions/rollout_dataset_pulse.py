import os
import sys
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

has_robot_arg = False
simulator = None
for arg in sys.argv:
    # Ensure isaacgym is imported before any torch modules (multi-process behavior).
    if "robot" in arg:
        has_robot_arg = True

    if "simulator" in arg:
        if not has_robot_arg:
            raise ValueError("+robot argument should be provided before +simulator")
        if "isaacgym" in arg.split("=")[-1]:
            import isaacgym  # noqa: F401

            simulator = "isaacgym"
        elif "isaaclab" in arg.split("=")[-1]:
            from isaaclab.app import AppLauncher

            simulator = "isaaclab"
        elif "genesis" in arg.split("=")[-1]:
            simulator = "genesis"

from utils.config_utils import *  # noqa: E402, F403

def _partition_range(num_items: int, world_size: int, rank: int) -> Tuple[int, int]:
    base = num_items // world_size
    extra = num_items % world_size
    start = base * rank + min(rank, extra)
    end = start + base + (1 if rank < extra else 0)
    return start, end

def _reset_envs_to_motions(env, env_ids, motion_ids):
    # Keep sampling logic (incl. init_start_prob) but force which motion is sampled.
    env.motion_manager.config.fixed_motion_per_env = False
    env.motion_manager.config.fixed_motion_id = None
    env.motion_manager.sample_motions(env_ids, new_motion_ids=motion_ids)

    if hasattr(env.motion_manager, "reset_track"):
        env.motion_manager.reset_track(env_ids)
    if getattr(env.config, "masked_mimic", None) is not None and env.config.masked_mimic.enabled:
        env.masked_mimic_obs_cb.reset_track(env_ids)

    new_states, reset_motion_ids, reset_motion_times = env.reset_ref_state_init(env_ids)
    env.simulator.reset_envs(new_states, env_ids)

    env.self_obs_cb.reset_envs(
        env_ids,
        reset_default_env_ids=[],
        reset_ref_env_ids=env_ids,
        reset_ref_motion_ids=reset_motion_ids,
        reset_ref_motion_times=reset_motion_times,
    )
    env.progress_buf[env_ids] = 0
    env.reset_buf[env_ids] = 0
    env.terminate_buf[env_ids] = 0

    env.compute_observations(env_ids)
    return env.get_obs()

def _masked_obs_for_act(obs, act_mask):
    if act_mask.all():
        # Clone to avoid modifying original obs when injecting vae_noise
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
    masked = {}
    for key, value in obs.items():
        if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == act_mask.shape[0]:
            value = value.clone()
            value[~act_mask] = 0
        masked[key] = value
    return masked

@hydra.main(config_path="config")
def main(override_config: OmegaConf):
    os.chdir(hydra.utils.get_original_cwd())
    import torch

    if override_config.checkpoint is None:
        raise ValueError("Please provide `+checkpoint=...` so rollout uses the trained policy config.")

    checkpoint = Path(override_config.checkpoint)
    config_path = checkpoint.parent / "config.yaml"
    if not config_path.exists():
        config_path = checkpoint.parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find training config.yaml next to checkpoint: {checkpoint}")

    with open(config_path) as file:
        train_config = OmegaConf.load(file)
    if train_config.eval_overrides is not None:
        train_config = OmegaConf.merge(train_config, train_config.eval_overrides)
    config = OmegaConf.merge(train_config, override_config)
    config.num_envs = 1024

    output_dir = Path(getattr(config, "output_dir", "outputs/rollouts_pulse")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    max_steps_per_motion: Optional[int] = getattr(config, "max_steps_per_motion", None)

    # Prefer rollouts starting at t=0 for each motion.
    if getattr(config, "motion_manager", None) is not None:
        config.motion_manager.motion_sampling.init_start_prob = 1.0

    simulation_app = None
    if simulator == "isaaclab":
        app_launcher = AppLauncher({"headless": bool(config.headless)})
        simulation_app = app_launcher.app

    from lightning.fabric import Fabric
    
    # We use PPO/MaskedMimic agent class
    fabric: Fabric = instantiate(config.fabric)
    fabric.launch()

    if simulator == "isaaclab":
        env = instantiate(config.env, device=fabric.device, simulation_app=simulation_app)
    else:
        env = instantiate(config.env, device=fabric.device)

    # Instantiate agent (could be PPO or MaskedMimic)
    agent = instantiate(config.agent, env=env, fabric=fabric)
    agent.setup()
    agent.load(config.checkpoint)
    agent.eval()
    sim = agent.env.simulator

    agent.reset_vae_noise(None)

    num_motions = env.motion_lib.num_motions()
    start, end = _partition_range(num_motions, fabric.world_size, fabric.global_rank)
    motion_ids_all = list(range(start, end))

    NUM_REPEATS = 1
    motion_ids_all = motion_ids_all * NUM_REPEATS
    
    motion_counts = defaultdict(int)

    num_envs = int(env.config.num_envs)
    print(f"num_envs: {num_envs}")
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")

    total_motions = 0
    successful_motions = 0

    FAIL_GT_ERR = 0.5 

    for batch_start in range(0, len(motion_ids_all), num_envs):
        batch_motion_ids = motion_ids_all[batch_start : batch_start + num_envs]
        active_count = len(batch_motion_ids)

        padded_motion_ids = batch_motion_ids + ([0] * (num_envs - active_count))
        active_mask = torch.zeros(num_envs, device=fabric.device, dtype=torch.bool)
        active_mask[:active_count] = True

        env_ids = torch.arange(num_envs, device=fabric.device, dtype=torch.long)
        motion_ids_tensor = torch.tensor(padded_motion_ids, device=fabric.device, dtype=torch.long)

        with torch.no_grad():
            obs = _reset_envs_to_motions(env, env_ids, motion_ids_tensor)
            agent.reset_vae_noise(env_ids)

        per_env_actions: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        per_env_body_pos: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        per_env_body_rot: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        per_env_body_vel: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        per_env_body_ang_vel: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        per_env_dof_pos: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        per_env_dof_vel: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        per_env_latents: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]

        finished = torch.zeros(num_envs, device=fabric.device, dtype=torch.bool)
        has_failed = torch.zeros(num_envs, device=fabric.device, dtype=torch.bool)
        
        step = 0
        dt = env.config.simulator.config.sim.decimation / env.config.simulator.config.sim.fps

        while True:
            act_mask = active_mask & ~finished
            if not act_mask.any():
                break
            if max_steps_per_motion is not None and step >= int(max_steps_per_motion):
                break

            with torch.no_grad():
                act_obs = _masked_obs_for_act(obs, act_mask)
                vae_noise = agent.vae_noise.clone()
                vae_noise[~act_mask] = 0
                act_obs["vae_noise"] = vae_noise
                actions, prior_out, encoder_out  = agent.model.get_action_and_vae_outputs(act_obs)
                # Manually compute latent z because in-place modification of act_obs inside the model 
                # might not persist through Fabric/DDP wrappers.
                mu = prior_out["mu"] + encoder_out["mu"]
                std = torch.exp(0.5 * encoder_out["logvar"])
                current_latents = mu + std * act_obs["vae_noise"]

            bodies_state = sim.get_bodies_state(env_ids)
            dof_state = env.simulator.get_dof_state()

            for env_i in act_mask.nonzero(as_tuple=False).flatten().tolist():
                per_env_actions[env_i].append(actions[env_i].detach().cpu())
                
                if current_latents is not None:
                    per_env_latents[env_i].append(current_latents[env_i].detach().cpu())

                if bodies_state.rigid_body_pos is not None:
                    per_env_body_pos[env_i].append(bodies_state.rigid_body_pos[env_i].detach().cpu())
                if bodies_state.rigid_body_rot is not None:
                    per_env_body_rot[env_i].append(bodies_state.rigid_body_rot[env_i].detach().cpu())
                if bodies_state.rigid_body_vel is not None:
                    per_env_body_vel[env_i].append(bodies_state.rigid_body_vel[env_i].detach().cpu())
                if bodies_state.rigid_body_ang_vel is not None:
                    per_env_body_ang_vel[env_i].append(bodies_state.rigid_body_ang_vel[env_i].detach().cpu())

                if dof_state.dof_pos is not None:
                    per_env_dof_pos[env_i].append(dof_state.dof_pos[env_i].detach().cpu())
                if dof_state.dof_vel is not None:
                    per_env_dof_vel[env_i].append(dof_state.dof_vel[env_i].detach().cpu())

            with torch.no_grad():
                obs, rewards, dones, extras = env.step(actions)
            
            if "gt_err" in env.mimic_info_dict:
                gt_err = env.mimic_info_dict["gt_err"]
                failed_mask = gt_err > FAIL_GT_ERR
                has_failed[failed_mask & act_mask] = True
            
            if "terminate" in extras:
                is_env_fail = extras["terminate"].to(torch.bool)
                has_failed[is_env_fail & act_mask] = True

            dones = dones.to(torch.bool)
            finished |= dones & active_mask
            step += 1

        for env_i in range(active_count):
            motion_id = batch_motion_ids[env_i]
            
            save_idx = motion_counts[motion_id]
            motion_counts[motion_id] += 1
            out_path = output_dir / f"motion_{motion_id:06d}_{save_idx:02d}.npz"

            actions_np = torch.stack(per_env_actions[env_i], dim=0).numpy().astype(np.float32, copy=False)
            
            motion_file = str(env.motion_lib.state.motion_files[motion_id])
            motion_fps = float(env.motion_lib.state.motion_fps[motion_id].item())
            
            is_success = not has_failed[env_i].item()
            
            if is_success:
                successful_motions += 1
            total_motions += 1

            save_dict = {
                "motion_id": np.int64(motion_id),
                "motion_file": np.array(motion_file),
                "motion_fps": np.float32(motion_fps),
                "dt": np.float32(env.dt),
                "success": bool(is_success),
                "actions": actions_np,
                "body_pos": torch.stack(per_env_body_pos[env_i], dim=0).numpy().astype(np.float32, copy=False),
                "body_rot": torch.stack(per_env_body_rot[env_i], dim=0).numpy().astype(np.float32, copy=False),
                "body_vel": torch.stack(per_env_body_vel[env_i], dim=0).numpy().astype(np.float32, copy=False),
                "body_ang_vel": torch.stack(per_env_body_ang_vel[env_i], dim=0).numpy().astype(np.float32, copy=False),
            }

            if len(per_env_latents[env_i]) > 0:
                save_dict["latents"] = torch.stack(per_env_latents[env_i], dim=0).numpy().astype(np.float32, copy=False)

            if len(per_env_dof_pos[env_i]) > 0:
                save_dict["dof_pos"] = torch.stack(per_env_dof_pos[env_i], dim=0).numpy().astype(np.float32, copy=False)
            if len(per_env_dof_vel[env_i]) > 0:
                save_dict["dof_vel"] = torch.stack(per_env_dof_vel[env_i], dim=0).numpy().astype(np.float32, copy=False)

            np.savez_compressed(out_path, **save_dict)

    if total_motions > 0:
        print(f"Rollout complete. Success rate: {successful_motions}/{total_motions} = {successful_motions/total_motions:.2%}")
    else:
        print("No motions processed.")

    fabric.barrier()


if __name__ == "__main__":
    main()
