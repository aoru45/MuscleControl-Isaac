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




def _one_euro_alpha(cutoff: torch.Tensor, dt: float) -> torch.Tensor:
    cutoff = torch.clamp(cutoff, min=1e-4)
    r = (2.0 * math.pi) * cutoff * dt
    return r / (r + 1.0)


def _one_euro_filter(
    x: torch.Tensor,
    dt: float,
    state: Optional[Dict[str, torch.Tensor]],
    *,
    min_cutoff: float,
    beta: float,
    d_cutoff: float,
    min_alpha: float = 0.0,
    reset_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if state is None:
        state = {
            "x_prev": x.clone(),
            "x_hat_prev": x.clone(),
            "dx_hat_prev": torch.zeros_like(x),
        }
        return x, state

    x_prev = state["x_prev"]
    x_hat_prev = state["x_hat_prev"]
    dx_hat_prev = state["dx_hat_prev"]

    if reset_indices is not None and reset_indices.numel() > 0:
        x_prev[reset_indices] = x[reset_indices]
        x_hat_prev[reset_indices] = x[reset_indices]
        dx_hat_prev[reset_indices] = 0.0

    dx = (x - x_prev) / dt
    a_d = _one_euro_alpha(torch.tensor(d_cutoff, device=x.device, dtype=x.dtype), dt)
    dx_hat = a_d * dx + (1.0 - a_d) * dx_hat_prev

    cutoff = min_cutoff + beta * dx_hat.abs()
    a = _one_euro_alpha(cutoff, dt)
    if min_alpha > 0.0:
        a = torch.clamp(a, min=min_alpha, max=1.0)
    x_hat = a * x + (1.0 - a) * x_hat_prev

    state["x_prev"] = x
    state["x_hat_prev"] = x_hat
    state["dx_hat_prev"] = dx_hat
    return x_hat, state




def optimize_act(JtA: torch.Tensor, b: torch.Tensor, tau: torch.Tensor, method: str = "lbfgs", max_iter = 20):
    # Detach inputs to treat them as constants
    JtA = JtA.detach()
    b = b.detach()
    tau_target = tau.detach()

    B, M, D = JtA.shape

    if method == "lbfgs":
        x = torch.zeros((B, M), device=JtA.device, requires_grad=True)
        optimizer = torch.optim.LBFGS([x], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad()
            a = torch.sigmoid(x)
            tau_pred = torch.einsum("bm,bmd->bd", a, JtA) + b
            loss = (tau_pred - tau_target).pow(2).sum() + 0.01 * a.pow(2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            a_final = torch.sigmoid(x)
            tau_final = torch.einsum("bm,bmd->bd", a_final, JtA) + b

        return a_final, tau_final

    elif method == "ls":
        target_torque = (tau_target - b).unsqueeze(-1) # (B, D, 1)
        matrix = JtA.transpose(1, 2) # (B, D, M)

        result = torch.linalg.lstsq(matrix, target_torque)
        a_sol = result.solution.squeeze(-1) # (B, M)

        epsilon = 1e-4
        a_clamped = torch.clamp(a_sol, epsilon, 1.0 - epsilon)
        x_init = torch.logit(a_clamped)

        x = x_init.clone().detach().requires_grad_(True)

        # Use fewer iterations for speed since we start close
        optimizer = torch.optim.LBFGS([x], lr=0.5, max_iter=5, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            a = torch.sigmoid(x)
            tau_pred = torch.einsum("bm,bmd->bd", a, JtA) + b
            loss = (tau_pred - tau_target).pow(2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            a_final = torch.sigmoid(x)
            tau_final = torch.einsum("bm,bmd->bd", a_final, JtA) + b

        return a_final, tau_final

    else:
        raise ValueError(f"Unknown optimization method: {method}")


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
        return obs
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

    output_dir = Path(getattr(config, "output_dir", "outputs/rollouts_sa40")).resolve()
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
    from protomotions.agents.ppo.agent import PPO

    fabric: Fabric = instantiate(config.fabric)
    fabric.launch()

    if simulator == "isaaclab":
        env = instantiate(config.env, device=fabric.device, simulation_app=simulation_app)
    else:
        env = instantiate(config.env, device=fabric.device)

    agent: PPO = instantiate(config.agent, env=env, fabric=fabric)
    agent.setup()
    agent.load(config.checkpoint)
    agent.eval()
    sim = agent.env.simulator

    num_motions = env.motion_lib.num_motions()
    start, end = _partition_range(num_motions, fabric.world_size, fabric.global_rank)
    motion_ids_all = list(range(start, end))

    # Expand dataset 40x
    NUM_REPEATS = 40
    motion_ids_all = motion_ids_all * NUM_REPEATS
    
    motion_counts = defaultdict(int)

    num_envs = int(env.config.num_envs)
    print(f"num_envs: {num_envs}")
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")

    total_motions = 0
    successful_motions = 0

    # Strict termination thresholds
    # We define strict thresholds to mark "failure" even if the environment does not terminate early.
    FAIL_GT_ERR = 0.5  # Global translation error threshold (meters)
    # FAIL_ROOT_HEIGHT = 0.5 # Root height threshold (optional, usually implied by gt_err or specific fall check)

    # Process in batches of size num_envs (multi-env parallelism).
    for batch_start in range(0, len(motion_ids_all), num_envs):
        batch_motion_ids = motion_ids_all[batch_start : batch_start + num_envs]
        active_count = len(batch_motion_ids)

        # Pad to full num_envs so we can keep stepping the environment in a fixed shape.
        padded_motion_ids = batch_motion_ids + ([0] * (num_envs - active_count))
        active_mask = torch.zeros(num_envs, device=fabric.device, dtype=torch.bool)
        active_mask[:active_count] = True

        env_ids = torch.arange(num_envs, device=fabric.device, dtype=torch.long)
        motion_ids_tensor = torch.tensor(padded_motion_ids, device=fabric.device, dtype=torch.long)

        with torch.no_grad():
            obs = _reset_envs_to_motions(env, env_ids, motion_ids_tensor)

        per_env_actions: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        # per_env_muscles: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        per_env_body_pos: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        per_env_body_rot: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        per_env_body_vel: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        per_env_body_ang_vel: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        per_env_dof_pos: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        per_env_dof_vel: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        # per_env_activations: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        # per_env_obs: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        finished = torch.zeros(num_envs, device=fabric.device, dtype=torch.bool)
        
        # Track failures explicitly based on tracking error
        has_failed = torch.zeros(num_envs, device=fabric.device, dtype=torch.bool)
        
        step = 0

        # # Initialize activation filter state
        # activation_filter_min_cutoff = 5.0
        # activation_filter_beta = 10.0
        # activation_filter_d_cutoff = 1.0
        # activation_filter_min_alpha = 0.
        # activation_filter_state = None
        dt = env.config.simulator.config.sim.decimation / env.config.simulator.config.sim.fps

        while True:
            act_mask = active_mask & ~finished
            if not act_mask.any():
                break
            if max_steps_per_motion is not None and step >= int(max_steps_per_motion):
                break

            with torch.no_grad():
                act_obs = _masked_obs_for_act(obs, act_mask)
                clean_actions = agent.model.act(act_obs, mean=False)
                
                # Inject noise to perturb state
                noise_std = 0.01
                noise = torch.randn_like(clean_actions) * noise_std
                noisy_actions = clean_actions + noise
                noisy_actions = torch.clamp(noisy_actions, -1.0, 1.0)
                
                # Use noisy actions for stepping, but we will record clean actions
                actions = noisy_actions

                # feats = sim.get_muscle_features()
                # JtA, b = feats["JtA"], feats["b"]
                # common_dof_state = env.simulator.get_dof_state()
                # q = common_dof_state.dof_pos
                # qd = common_dof_state.dof_vel
                # tau = sim._compute_tau_from_actions(actions, q, qd)

                # # Calculate and filter activations for each frame
                # a_opt, _ = optimize_act(JtA, b, tau)
                # a_opt = torch.clamp(a_opt.detach(), 0.0, 1.0)

                # a_filt, activation_filter_state = _one_euro_filter(
                #     a_opt,
                #     dt,
                #     activation_filter_state,
                #     min_cutoff=activation_filter_min_cutoff,
                #     beta=activation_filter_beta,
                #     d_cutoff=activation_filter_d_cutoff,
                #     min_alpha=activation_filter_min_alpha,
                #     reset_indices=None,
                # )
                # a_filt = torch.clamp(a_filt, 0.0, 1.0)

                # mus = torch.cat([
                #     JtA.reshape(-1, 284, 50).flatten(1),
                #     b.reshape(-1, 50),
                #     tau.reshape(-1,50)
                # ], dim=1)

            bodies_state = sim.get_bodies_state(env_ids)
            dof_state = env.simulator.get_dof_state()

            for env_i in act_mask.nonzero(as_tuple=False).flatten().tolist():
                per_env_actions[env_i].append(clean_actions[env_i].detach().cpu())
                # per_env_muscles[env_i].append(mus[env_i].detach().cpu())
                # per_env_activations[env_i].append(a_filt[env_i].detach().cpu())
                # per_env_obs[env_i].append(act_obs[env_i].detach().cpu())

                # Record full body state for replay
                if bodies_state.rigid_body_pos is not None:
                    per_env_body_pos[env_i].append(bodies_state.rigid_body_pos[env_i].detach().cpu())
                if bodies_state.rigid_body_rot is not None:
                    per_env_body_rot[env_i].append(bodies_state.rigid_body_rot[env_i].detach().cpu())
                if bodies_state.rigid_body_vel is not None:
                    per_env_body_vel[env_i].append(bodies_state.rigid_body_vel[env_i].detach().cpu())
                if bodies_state.rigid_body_ang_vel is not None:
                    per_env_body_ang_vel[env_i].append(bodies_state.rigid_body_ang_vel[env_i].detach().cpu())

                # Record DOF state (from dof_state, not bodies_state)
                if dof_state.dof_pos is not None:
                    per_env_dof_pos[env_i].append(dof_state.dof_pos[env_i].detach().cpu())
                if dof_state.dof_vel is not None:
                    per_env_dof_vel[env_i].append(dof_state.dof_vel[env_i].detach().cpu())

            with torch.no_grad():
                obs, rewards, dones, extras = env.step(actions)
            
            # Explicitly check for failure conditions (tracking error)
            # This is robust to config overrides that might disable termination
            if "gt_err" in env.mimic_info_dict:
                gt_err = env.mimic_info_dict["gt_err"]
                failed_mask = gt_err > FAIL_GT_ERR
                has_failed[failed_mask & act_mask] = True
            
            # Also check if terminated by env (could be other reasons)
            # Note: If env terminates, extras['terminate'] is true.
            if "terminate" in extras:
                is_env_fail = extras["terminate"].to(torch.bool)
                has_failed[is_env_fail & act_mask] = True

            dones = dones.to(torch.bool)
            finished |= dones & active_mask
            step += 1

        # Save one file per (active) motion.
        for env_i in range(active_count):
            motion_id = batch_motion_ids[env_i]
            
            save_idx = motion_counts[motion_id]
            motion_counts[motion_id] += 1
            out_path = output_dir / f"motion_{motion_id:06d}_{save_idx:02d}.npz"

            actions_np = torch.stack(per_env_actions[env_i], dim=0).numpy().astype(np.float32, copy=False)
            # mus_torch = torch.stack(per_env_muscles[env_i], dim=0)
            # mus_np = mus_torch.numpy().astype(np.float32, copy=False)
            # activations_torch = torch.stack(per_env_activations[env_i], dim=0)
            # activations_np = activations_torch.numpy().astype(np.float32, copy=False)
            # obs_np = torch.stack(per_env_obs[env_i], dim=0).numpy().astype(np.float32, copy=False)


            #########################################################
            # JtA = mus_torch[:, :284*50].reshape(-1, 284, 50).cuda()
            # b = mus_torch[:, 284*50:284*50+50].cuda()
            # tau = mus_torch[:, 284*50+50:].cuda()

            # # Choose optimization method: "pgd_fast" (default, fastest), "admm", or "lbfgs"
            # method = "adam"
            # a, pd_tau_des = optimize_act(JtA, b, tau, method=method, max_iter=200, sigma=1.0)   
            # print(a.shape)
            # print(f"Method: {method}, Error: {(pd_tau_des - tau).abs().mean():.6f}")


            motion_file = str(env.motion_lib.state.motion_files[motion_id])
            motion_fps = float(env.motion_lib.state.motion_fps[motion_id].item())
            
            # Determine success based on our explicit tracking
            is_success = not has_failed[env_i].item()
            
            if is_success:
                successful_motions += 1
            total_motions += 1


            # Prepare body state tensors (if available)
            save_dict = {
                "motion_id": np.int64(motion_id),
                "motion_file": np.array(motion_file),
                "motion_fps": np.float32(motion_fps),
                "dt": np.float32(env.dt),
                "success": bool(is_success),
                "actions": actions_np,
                # "mus": mus_np,
                # "obs": obs_np,
                # "activations": activations_np,
                "body_pos": torch.stack(per_env_body_pos[env_i], dim=0).numpy().astype(np.float32, copy=False),
                "body_rot": torch.stack(per_env_body_rot[env_i], dim=0).numpy().astype(np.float32, copy=False),
                "body_vel": torch.stack(per_env_body_vel[env_i], dim=0).numpy().astype(np.float32, copy=False),
                "body_ang_vel": torch.stack(per_env_body_ang_vel[env_i], dim=0).numpy().astype(np.float32, copy=False),
            }

            # Add DOF state if available
            if len(per_env_dof_pos[env_i]) > 0:
                save_dict["dof_pos"] = torch.stack(per_env_dof_pos[env_i], dim=0).numpy().astype(np.float32, copy=False)
            if len(per_env_dof_vel[env_i]) > 0:
                save_dict["dof_vel"] = torch.stack(per_env_dof_vel[env_i], dim=0).numpy().astype(np.float32, copy=False)

            np.savez_compressed(out_path, **save_dict)
            # exit()

    if total_motions > 0:
        print(f"Rollout complete. Success rate: {successful_motions}/{total_motions} = {successful_motions/total_motions:.2%}")
    else:
        print("No motions processed.")

    fabric.barrier()


if __name__ == "__main__":
    main()