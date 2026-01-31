import torch
import time
import math
import torch.nn.functional as F
import logging
from pathlib import Path
from protomotions.simulator.base_simulator.config import (
    ControlType,
)
from rich.progress import track

from typing import List, Tuple, Dict, Optional

from protomotions.envs.mimic.env import Mimic as MimicEnv
from protomotions.agents.ppo.agent import PPO, ExperienceBuffer, discount_values
from protomotions.simulator.base_simulator.simulator import soft_bound

log = logging.getLogger(__name__)

def optimize_act(JtA: torch.Tensor, b: torch.Tensor, tau: torch.Tensor, method: str = "ls", max_iter = 20):
    # Detach inputs to treat them as constants
    JtA = JtA.detach()
    b = b.detach()
    tau_target = tau.detach()

    B, M, D = JtA.shape

    if method == "lbfgs":
        global last_a
        # if last_a is None:
        x = torch.zeros((B, M), device=JtA.device, requires_grad=True)
        optimizer = torch.optim.LBFGS([x], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad()
            a = torch.sigmoid(x)
            tau_pred = torch.einsum("bm,bmd->bd", a, JtA) + b
            loss = (tau_pred - tau_target).pow(2).sum()# + 0.01*(a - last_a).pow(2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            a_final = torch.sigmoid(x)
            # a_final = torch.tanh(x) * 0.5 + 0.5
            tau_final = torch.einsum("bm,bmd->bd", a_final, JtA) + b
            last_a = a_final.detach().clone()

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

class Mimic(PPO):
    env: MimicEnv
    # -----------------------------
    # Motion Mapping and Data Distribution
    # -----------------------------
    def map_motions_to_iterations(self) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int, bool]:
        """
        Maps motion IDs to iterations for distributed processing.
        Returns:
            - motion_map: List of (ids, requires_scene) tuples
            - motions_per_rank: Number of motions this rank handles
            - is_sharded: Boolean, True if each rank has a unique shard of motions
        """
        world_size = self.fabric.world_size
        global_rank = self.fabric.global_rank
        num_motions = self.motion_lib.num_motions()

        # Check if we are in sharded mode (binary files per rank) or global mode (one yaml for all)
        # We assume sharded if the motion_file path is different from rank 0's path.
        local_motion_file = getattr(self.motion_lib, "motion_file", "")
        # Broadcast rank 0's filename to compare
        root_motion_file = self.fabric.broadcast(local_motion_file, src=0)
        
        # If filenames differ, or if num_motions differs (fallback check), we are sharded.
        is_sharded = (local_motion_file != root_motion_file)
        
        # Always run this check to ensure all ranks execute the same number of collectives (all_gather)
        local_count = torch.tensor([num_motions], device=self.device)
        all_counts = self.fabric.all_gather(local_count)
        
        if not is_sharded:
            if isinstance(all_counts, list):
                all_counts = torch.stack(all_counts)
            if not (all_counts == num_motions).all():
                is_sharded = True
                print(f"[Rank {global_rank}] Detected sharded mode via num_motions mismatch.")

        # Handle fixed motion ID case
        if self.env.config.motion_manager.fixed_motion_id is not None:
            motion_ids = torch.tensor(
                [self.env.config.motion_manager.fixed_motion_id], device=self.device
            )
            requires_scene = self.env.get_motion_requires_scene(motion_ids)
            return [(motion_ids, requires_scene)], 1, False

        if is_sharded:
            # Each rank owns its motions fully.
            start_motion = 0
            end_motion = num_motions
            motions_per_rank = num_motions
        else:
            # Global mode: partition the global list
            base_motions_per_rank = num_motions // world_size
            extra_motions = num_motions % world_size
            motions_per_rank = base_motions_per_rank + (1 if global_rank < extra_motions else 0)
            start_motion = base_motions_per_rank * global_rank + min(global_rank, extra_motions)
            end_motion = start_motion + motions_per_rank

        # Create tensor of motion IDs assigned to this rank
        motion_range = torch.arange(start_motion, end_motion, device=self.device)

        # Split motions into batches of size self.num_envs
        motion_map = []
        for i in range(0, len(motion_range), self.num_envs):
            batch_motion_ids = motion_range[i : i + self.num_envs]
            # Sample corresponding scene IDs
            requires_scene = self.env.get_motion_requires_scene(batch_motion_ids)
            motion_map.append((batch_motion_ids, requires_scene))

        return motion_map, motions_per_rank, is_sharded

    # -----------------------------
    # Evaluation and Metrics Collection
    # -----------------------------

    @torch.no_grad()
    def calc_eval_metrics(self) -> Tuple[Dict, Optional[float]]:
        self.eval()
        if self.env.config.motion_manager.fixed_motion_id is not None:
            num_motions = 1
        else:
            num_motions = self.motion_lib.num_motions()

        metrics = {
            # Track which motions are evaluated (within time limit)
            "evaluated": torch.zeros(num_motions, device=self.device, dtype=torch.bool),
        }
        for k in self.config.eval_metric_keys:
            metrics[k] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_max"] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_min"] = 3 * torch.ones(num_motions, device=self.device)

        # Compute how many motions each rank should evaluate
        root_dir = Path(self.fabric.loggers[0].root_dir)
        motion_map, remaining_motions, is_sharded = self.map_motions_to_iterations()
        num_outer_iters = len(motion_map)
        
        # Calculate max_iters robustly by gathering tensors
        local_len = torch.tensor(num_outer_iters, device=self.device)
        all_lens = self.fabric.all_gather(local_len)
        if isinstance(all_lens, list):
            all_lens = torch.stack(all_lens)
        max_iters = all_lens.max().item()

        # Only show progress bar on rank 0 to avoid duplicate output in multi-GPU
        if self.fabric.global_rank == 0:
            print(f"Starting evaluation... Total batches: {max_iters}")
            outer_iter_iterator = track(
                range(max_iters),
                description=f"Evaluating... {remaining_motions} motions remain...",
            )
        else:
            outer_iter_iterator = range(max_iters)

        for outer_iter in outer_iter_iterator:
            if num_outer_iters == 0:
                continue
            motion_pointer = outer_iter % num_outer_iters
            motion_ids, requires_scene = motion_map[motion_pointer]
            num_motions_this_iter = len(motion_ids)
            
            # Important: prevent double-counting metrics if we are just padding execution
            is_valid_iter = outer_iter < num_outer_iters
            
            if is_valid_iter:
                metrics["evaluated"][motion_ids] = True

            # Define the task mapping for each agent.
            self.env.agent_in_scene[:] = False
            self.env.motion_manager.motion_ids[:num_motions_this_iter] = motion_ids
            self.env.agent_in_scene[:num_motions_this_iter] = requires_scene
            # Force respawn on flat terrain to ensure proper motion reconstruction.
            self.env.force_respawn_on_flat = True

            env_ids = torch.arange(
                0, num_motions_this_iter, dtype=torch.long, device=self.device
            )

            dt: float = self.env.dt
            motion_lengths = self.motion_lib.get_motion_length(motion_ids)
            motion_num_frames = (motion_lengths / dt).floor().long()

            max_len = (
                motion_num_frames.max().item()
                if self.config.eval_length is None
                else self.config.eval_length
            )

            for eval_episode in range(self.config.eval_num_episodes):
                # Sample random start time with slight noise for varied initial conditions.
                elapsed_time = (
                    torch.rand_like(self.motion_lib.state.motion_lengths[motion_ids]) * dt
                )
                self.env.motion_manager.motion_times[:num_motions_this_iter] = elapsed_time
                self.env.motion_manager.reset_track_steps.reset_steps(env_ids)
                # Disable automatic reset to maintain consistency in evaluation.
                self.env.disable_reset = True
                self.env.motion_manager.disable_reset_track = True

                obs = self.env.reset(
                    torch.arange(0, self.num_envs, dtype=torch.long, device=self.device)
                )

                for l in range(max_len):
                    actions = self.model.act(obs)
                    obs, rewards, dones, terminated, extras = self.env_step(actions)
                    
                    if is_valid_iter:
                        elapsed_time += dt
                        clip_done = (motion_lengths - dt) < elapsed_time
                        clip_not_done = torch.logical_not(clip_done)
                        for k in self.config.eval_metric_keys:
                            if k in self.env.mimic_info_dict:
                                value = self.env.mimic_info_dict[k].detach()
                            else:
                                raise ValueError(f"Key {k} not found in mimic_info_dict")
                            # Only update metrics for motions that are continuing.
                            metric = value[:num_motions_this_iter]
                            metrics[k][motion_ids[clip_not_done]] += metric[clip_not_done]
                            metrics[f"{k}_max"][motion_ids[clip_not_done]] = torch.maximum(
                                metrics[f"{k}_max"][motion_ids[clip_not_done]],
                                metric[clip_not_done],
                            )
                            metrics[f"{k}_min"][motion_ids[clip_not_done]] = torch.minimum(
                                metrics[f"{k}_min"][motion_ids[clip_not_done]],
                                metric[clip_not_done],
                            )

        # Only print on rank 0 to avoid duplicate output in multi-GPU
        if self.fabric.global_rank == 0:
            print("Evaluation done, now aggregating data.")

        if self.env.config.motion_manager.fixed_motion_id is None:
            motion_lengths = self.motion_lib.state.motion_lengths[:]
            motion_num_frames = (motion_lengths / dt).floor().long()
            
        self.fabric.barrier()

        # Check local evaluation completeness
        if not metrics["evaluated"].all():
             missing_indices = torch.nonzero(~metrics["evaluated"]).flatten().tolist()
             print(f"[Rank {self.fabric.global_rank}] Warning: Missing {len(missing_indices)} evaluations locally.")
        
        # Aggregation Logic
        if not is_sharded:
            # GLOBAL MODE: We must merge disjoint parts from all ranks.
            # Convert bool to float for robust distributed reduction/gathering
            evaluated_float = metrics["evaluated"].float().contiguous()
            metrics["evaluated"] = self.fabric.all_reduce(evaluated_float, reduce_op="max").bool()
            
            assert metrics["evaluated"].all(), f"Not all motions were evaluated. Missing: {(~metrics['evaluated']).sum().item()}"

            # Sync other metrics using all_reduce
            for k in self.config.eval_metric_keys:
                metrics[k] = self.fabric.all_reduce(metrics[k].contiguous(), reduce_op="sum")
                metrics[f"{k}_max"] = self.fabric.all_reduce(metrics[f"{k}_max"].contiguous(), reduce_op="max")
                metrics[f"{k}_min"] = self.fabric.all_reduce(metrics[f"{k}_min"].contiguous(), reduce_op="min")
        else:
            # SHARDED MODE: No global merge of large tensors.
            # Local metrics are complete for the local shard.
            # We will compute logs locally and let `post_epoch_logging` aggregate the scalar averages.
            pass

        self.fabric.barrier()

        to_log = {}
        for k in self.config.eval_metric_keys:
            mean_tracking_errors = metrics[k] / (motion_num_frames * self.config.eval_num_episodes)
            
            if is_sharded and self.fabric.world_size > 1:
                # In sharded mode, we must aggregate the scalar statistics from all ranks
                # to get the global mean.
                local_sum = mean_tracking_errors.sum()
                local_max_sum = metrics[f"{k}_max"].sum()
                local_min_sum = metrics[f"{k}_min"].sum()
                local_count = torch.tensor(mean_tracking_errors.numel(), device=self.device, dtype=torch.float32)

                global_sum = self.fabric.all_reduce(local_sum, reduce_op="sum")
                global_max_sum = self.fabric.all_reduce(local_max_sum, reduce_op="sum")
                global_min_sum = self.fabric.all_reduce(local_min_sum, reduce_op="sum")
                global_count = self.fabric.all_reduce(local_count, reduce_op="sum")

                to_log[f"eval/{k}"] = (global_sum / global_count).item()
                to_log[f"eval/{k}_max"] = (global_max_sum / global_count).item()
                to_log[f"eval/{k}_min"] = (global_min_sum / global_count).item()
            else:
                to_log[f"eval/{k}"] = mean_tracking_errors.detach().mean().item()
                to_log[f"eval/{k}_max"] = metrics[f"{k}_max"].detach().mean().item()
                to_log[f"eval/{k}_min"] = metrics[f"{k}_min"].detach().mean().item()

        if "gt_err" in self.config.eval_metric_keys:
            tracking_failures = (metrics["gt_err_max"] > 0.5).float()
            
            if is_sharded and self.fabric.world_size > 1:
                local_fail_sum = tracking_failures.sum()
                local_count = torch.tensor(tracking_failures.numel(), device=self.device, dtype=torch.float32)
                
                global_fail_sum = self.fabric.all_reduce(local_fail_sum, reduce_op="sum")
                global_count = self.fabric.all_reduce(local_count, reduce_op="sum")
                
                to_log["eval/tracking_success_rate"] = 1.0 - (global_fail_sum / global_count).item()
            else:
                to_log["eval/tracking_success_rate"] = 1.0 - tracking_failures.detach().mean().item()

            failed_motions = torch.nonzero(tracking_failures).flatten().tolist()
            with open(root_dir / f"failed_motions_{self.fabric.global_rank}.txt", "w") as f:
                for motion_id in failed_motions:
                    f.write(f"{motion_id}\n")
                    
            new_weights = torch.ones(self.motion_lib.num_motions(), device=self.device) * 1e-4
            new_weights[failed_motions] = 1.0
            self.env.motion_manager.update_sampling_weights(new_weights)

        stop_early = (
            self.config.training_early_termination.early_terminate_cart_err is not None
            or self.config.training_early_termination.early_terminate_success_rate is not None
        ) 
        
        local_success = torch.tensor(to_log.get("eval/tracking_success_rate", 0.0), device=self.device)
        global_success = self.fabric.all_reduce(local_success, reduce_op="mean")

        if self.config.training_early_termination.early_terminate_success_rate is not None:
             stop_early = stop_early and (global_success >= self.config.training_early_termination.early_terminate_success_rate)
        
        if self.config.training_early_termination.early_terminate_cart_err is not None:
             # Need global cart error too if we use it, but typically success rate is used.
             # Assuming cart error follows similar reduction or just using local rank 0's if global mode.
             # For robust sharded mode, we should average it too.
             local_cart = torch.tensor(to_log.get("eval/cartesian_err", 999.0), device=self.device)
             global_cart = self.fabric.all_reduce(local_cart, reduce_op="mean")
             stop_early = stop_early and (global_cart <= self.config.training_early_termination.early_terminate_cart_err)

        if stop_early and self.fabric.global_rank == 0:
            print("Stopping early! Target error reached")
            self.best_evaluated_score = global_success
            self.save(new_high_score=True)
            self.terminate_early()

        self.env.disable_reset = False
        self.env.motion_manager.disable_reset_track = False
        self.env.force_respawn_on_flat = False

        all_ids = torch.arange(0, self.num_envs, dtype=torch.long, device=self.device)
        self.env.motion_manager.reset_envs(all_ids)
        self.force_full_restart = True

        return to_log, to_log.get("eval/tracking_success_rate", to_log.get("eval/cartesian_err", None))

class MimicDual(Mimic):
    def setup(self):
        super().setup()

        from protomotions.agents.ppo.model import PPOModelDual
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        dual_conf = OmegaConf.create(OmegaConf.to_container(self.config.model, resolve=True))
        self.dual_model = instantiate(dual_conf, _target_=PPOModelDual)

        from torch.optim import Adam
        self.student_model = self.fabric.setup(self.dual_model._student)
        self.student_optimizer = Adam(list(self.student_model.parameters()), lr=getattr(self.config, "student_lr", 0.001))
        self.student_optimizer = self.fabric.setup_optimizers(self.student_optimizer)

        self._train_teacher = self.config.get("train_teacher", False)

        if not self._train_teacher:
            def _dual_get_action_and_value(obs):
                sim = self.env.simulator
                return action, teacher_dist.mean, neglogp, value
            self.model.get_action_and_value = _dual_get_action_and_value
        else:
            def _teacher_get_action_and_value(obs):
                dist = self.model._actor(obs)
                action = dist.sample()
                value = self.model._critic(obs).flatten()
                logstd = self.model._actor.logstd
                std = torch.exp(logstd)
                neglogp = self.model.neglogp(action, dist.mean, std, logstd)
                return action, dist.mean, neglogp, value.flatten()
            self.model.get_action_and_value = _teacher_get_action_and_value

    def build_student_input(self, JtA, teacher_actions):
        JtA_per_dof = JtA.transpose(1, 2)  # (B, 50, 284)
        _, indices = torch.topk(JtA_per_dof.abs(), 3, dim=-1)  # (B, 50, 3)
        stack_JtA = torch.gather(JtA_per_dof, -1, indices)  # (B, 50, 3)
        jta_top3 = stack_JtA.flatten(start_dim=1)  # (B, 150)
        features = torch.cat([jta_top3, teacher_actions], dim=-1)
        return {"features": features.detach()}

    def extra_optimization_steps(self, batch_dict, batch_idx: int):
        if self._train_teacher:
            return {}
        log_dict = {}
        JtA = batch_dict["JtA"].detach()
        b = batch_dict["b"].detach()
        q = batch_dict["q"].detach()
        qd = batch_dict["qd"].detach()

        J = JtA.transpose(1, 2)
        tau_min, tau_max = soft_bound(J, b, self.env.simulator._torque_limits_common)
        actions_target = batch_dict["teacher_actions"].detach()
        torques_target = self.env.simulator._compute_tau_from_actions(actions_target, q, qd, min_tau=tau_min, max_tau=tau_max)

        student_obs = self.build_student_input(JtA, torques_target)
        a_pred = self.student_model(student_obs).mean 

        a_target, _tmp_tau = optimize_act(JtA, b, torques_target)

        torques_pred = self.env.simulator.muscle_ctl.activations_to_joint_torques_dynamic(
            a_pred, JtA, b
        )
        actions_pred = self.env.simulator._compute_actions_from_tau(torques_pred, q, qd)
        act_avg = a_pred.mean()
        act_max = a_pred.max()

        loss_a = F.mse_loss(a_pred, a_target.detach()) * 10
        loss_tau = F.mse_loss(torques_pred, torques_target) 
        loss_a_smooth = F.mse_loss(a_pred[:, 1:] - a_pred[:, :-1], torch.zeros_like(a_pred[:, 1:] - a_pred[:, :-1]))
        loss_tau_smooth = F.mse_loss(torques_pred[:, 1:] - torques_pred[:, :-1], torch.zeros_like(torques_pred[:, 1:] - torques_pred[:, :-1]))
        loss = loss_tau + loss_a +  0.001 * a_pred.pow(2).mean() + 0.01*loss_a_smooth + 0.01*loss_tau_smooth
        self.student_optimizer.zero_grad(set_to_none=True)
        self.fabric.backward(loss)
        self.student_optimizer.step()
        log_dict["muscle/action_match"] = (actions_pred - actions_target).abs().mean().detach()
        log_dict["muscle/tau_match"] = (torques_pred - torques_target).abs().mean().detach()
        log_dict["muscle/tau_pred"] = torques_pred.abs().mean().detach()
        log_dict["muscle/tau_target"] = torques_target.abs().mean().detach()
        log_dict["muscle/activation_avg"] = act_avg.detach()
        log_dict["muscle/activation_max"] = act_max.detach()
        log_dict["losses/student_loss"] = loss.detach()
        return log_dict

    def get_state_dict(self, state_dict):
        d = super().get_state_dict(state_dict)
        if hasattr(self, "student_model"):
            d["student_model"] = self.student_model.state_dict()
        if hasattr(self, "student_optimizer"):
            d["student_optimizer"] = self.student_optimizer.state_dict()
        return d

    def load_parameters(self, state_dict):
        super().load_parameters(state_dict)
        if not self._train_teacher:
            if hasattr(self, "student_model") and ("student_model" in state_dict):
                self.student_model.load_state_dict(state_dict["student_model"], strict=False)
            if hasattr(self, "student_optimizer") and ("student_optimizer" in state_dict):
                self.student_optimizer.load_state_dict(state_dict["student_optimizer"])

    
    @torch.no_grad()
    def calc_eval_metrics(self) -> Tuple[Dict, Optional[float]]:
        return super().calc_eval_metrics()
    def fit(self):
        # Setup experience buffer
        self.experience_buffer = ExperienceBuffer(self.num_envs, self.num_steps).to(
            self.device
        )
        self.experience_buffer.register_key(
            "self_obs", shape=(self.env.config.robot.self_obs_size,)
        )
        self.experience_buffer.register_key(
            "actions", shape=(self.env.config.robot.number_of_actions,)
        )

        # my
        if not self._train_teacher:
            self.experience_buffer.register_key("q", shape=(50,))
            self.experience_buffer.register_key("qd", shape=(50,))
            self.experience_buffer.register_key("JtA", shape=(284,50))
            self.experience_buffer.register_key("b", shape=(50,))
        self.experience_buffer.register_key(
            "teacher_actions", shape=(self.env.config.robot.number_of_actions,)
        )


        self.experience_buffer.register_key("rewards")
        self.experience_buffer.register_key("extra_rewards")
        self.experience_buffer.register_key("total_rewards")
        self.experience_buffer.register_key("dones", dtype=torch.long)
        self.experience_buffer.register_key("values")
        self.experience_buffer.register_key("next_values")
        self.experience_buffer.register_key("returns")
        self.experience_buffer.register_key("advantages")
        self.experience_buffer.register_key("neglogp")
        self.register_extra_experience_buffer_keys()

        if self.config.get("extra_inputs", None) is not None:
            obs = self.env.get_obs()
            for key in self.config.extra_inputs.keys():
                assert (
                    key in obs
                ), f"Key {key} not found in obs returned from env: {obs.keys()}"
                env_tensor = obs[key]
                shape = env_tensor.shape
                dtype = env_tensor.dtype
                self.experience_buffer.register_key(key, shape=shape[1:], dtype=dtype)

        # Force reset on fit start
        done_indices = None
        if self.fit_start_time is None:
            self.fit_start_time = time.time()
        self.fabric.call("on_fit_start", self)

        while self.current_epoch < self.config.max_epochs:
            self.epoch_start_time = time.time()

            # Set networks in eval mode so that normalizers are not updated
            self.eval()
            with torch.no_grad():
                self.fabric.call("before_play_steps", self)

                for step in track(
                    range(self.num_steps),
                    description=f"Epoch {self.current_epoch}, collecting data...",
                ):
                    obs = self.handle_reset(done_indices)
                    self.experience_buffer.update_data("self_obs", step, obs["self_obs"])
                    if self.config.get("extra_inputs", None) is not None:
                        for key in self.config.extra_inputs:
                            self.experience_buffer.update_data(key, step, obs[key])

                    action, teacher_action, neglogp, value = self.model.get_action_and_value(obs)
                    self.experience_buffer.update_data("teacher_actions", step, teacher_action)

                    self.experience_buffer.update_data("actions", step, action)
                    self.experience_buffer.update_data("neglogp", step, neglogp)
                    if self.config.normalize_values:
                        value = self.running_val_norm.normalize(value, un_norm=True)
                    self.experience_buffer.update_data("values", step, value)

                    # Check for NaNs in observations and actions
                    for key in obs.keys():
                        if torch.isnan(obs[key]).any():
                            print(f"NaN in {key}: {obs[key]}")
                            raise ValueError("NaN in obs")
                    if torch.isnan(action).any():
                        raise ValueError(f"NaN in action: {action}")

                    # Step the environment
                    next_obs, rewards, dones, terminated, extras = self.env_step(action)

                    all_done_indices = dones.nonzero(as_tuple=False)
                    done_indices = all_done_indices.squeeze(-1)

                    # Update logging metrics with the environment feedback
                    self.post_train_env_step(rewards, dones, done_indices, extras, step)

                    self.experience_buffer.update_data("rewards", step, rewards)
                    self.experience_buffer.update_data("dones", step, dones)



                    next_value = self.model._critic(next_obs).flatten()
                    if self.config.normalize_values:
                        next_value = self.running_val_norm.normalize(
                            next_value, un_norm=True
                        )
                    next_value = next_value * (1 - terminated.float())
                    self.experience_buffer.update_data("next_values", step, next_value)

                    self.step_count += self.get_step_count_increment()

                # After data collection, compute rewards, advantages, and returns.
                rewards = self.experience_buffer.rewards
                extra_rewards = self.calculate_extra_reward()
                self.experience_buffer.batch_update_data("extra_rewards", extra_rewards)
                total_rewards = rewards + extra_rewards
                self.experience_buffer.batch_update_data("total_rewards", total_rewards)

                advantages = discount_values(
                    self.experience_buffer.dones,
                    self.experience_buffer.values,
                    total_rewards,
                    self.experience_buffer.next_values,
                    self.gamma,
                    self.tau,
                )
                returns = advantages + self.experience_buffer.values
                self.experience_buffer.batch_update_data("returns", returns)

                if self.config.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                self.experience_buffer.batch_update_data("advantages", advantages)

            training_log_dict = self.optimize_model()
            training_log_dict["epoch"] = self.current_epoch
            self.current_epoch += 1
            self.fabric.call("after_train", self)

            # Save model checkpoint at specified intervals before evaluation.
            if self.current_epoch % self.config.manual_save_every == 0:
                self.save()

            if (
                self.config.eval_metrics_every is not None
                and self.current_epoch > 0
                and self.current_epoch % self.config.eval_metrics_every == 0
            ):
                eval_log_dict, evaluated_score = self.calc_eval_metrics()
                evaluated_score = self.fabric.broadcast(evaluated_score, src=0)
                if evaluated_score is not None:
                    if (
                        self.best_evaluated_score is None
                        or evaluated_score >= self.best_evaluated_score
                    ):
                        self.best_evaluated_score = evaluated_score
                        self.save(new_high_score=True)
                training_log_dict.update(eval_log_dict)

            self.post_epoch_logging(training_log_dict)
            self.env.on_epoch_end(self.current_epoch)

            if self.should_stop:
                self.save()
                return

        self.time_report.report()
        self.save()
        self.fabric.call("on_fit_end", self)
    def optimize_model(self) -> Dict:
        if self._train_teacher:
            return super().optimize_model()

        dataset = self.process_dataset(self.experience_buffer.make_dict())
        self.train()
        training_log_dict = {}

        if self.fabric.global_rank == 0:
            batch_iterator = track(
                range(self.max_num_batches()),
                description=f"Epoch {self.current_epoch}, training...",
            )
        else:
            batch_iterator = range(self.max_num_batches())

        for batch_idx in batch_iterator:
            iter_log_dict = {}
            dataset_idx = batch_idx % len(dataset)

            # Reshuffle dataset at the beginning of each mini epoch if configured.
            if dataset_idx == 0 and batch_idx != 0 and dataset.do_shuffle:
                dataset.shuffle()
            batch_dict = dataset[dataset_idx]

            # Check for NaNs in the batch.
            for key in batch_dict.keys():
                if torch.isnan(batch_dict[key]).any():
                    print(f"NaN in {key}: {batch_dict[key]}")
                    raise ValueError("NaN in training")
            # Update actor
            actor_loss, actor_loss_dict = self.actor_step(batch_dict)
            iter_log_dict.update(actor_loss_dict)
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.fabric.backward(actor_loss)
            actor_grad_clip_dict = self.handle_model_grad_clipping(
                self.model._actor, self.actor_optimizer, "actor"
            )
            iter_log_dict.update(actor_grad_clip_dict)
            self.actor_optimizer.step()

            # Update critic
            critic_loss, critic_loss_dict = self.critic_step(batch_dict)
            iter_log_dict.update(critic_loss_dict)
            self.critic_optimizer.zero_grad(set_to_none=True)
            self.fabric.backward(critic_loss)
            critic_grad_clip_dict = self.handle_model_grad_clipping(
                self.model._critic, self.critic_optimizer, "critic"
            )
            iter_log_dict.update(critic_grad_clip_dict)
            self.critic_optimizer.step()

            # Extra optimization steps if needed.
            extra_opt_steps_dict = self.extra_optimization_steps(batch_dict, batch_idx)
            iter_log_dict.update(extra_opt_steps_dict)

            for k, v in iter_log_dict.items():
                if k in training_log_dict:
                    training_log_dict[k][0] += v
                    training_log_dict[k][1] += 1
                else:
                    training_log_dict[k] = [v, 1]

        if self.fabric.world_size > 1:
            self._sync_training_metrics(training_log_dict)

        for k, v in training_log_dict.items():
            training_log_dict[k] = v[0] / v[1]

        self.eval()
        return training_log_dict

    @torch.no_grad()
    def evaluate_policy(self):
        self.eval()
        done_indices = None
        step = 0
        sim = self.env.simulator

        activation_filter_min_cutoff = 3.0
        activation_filter_beta = 0.8
        activation_filter_d_cutoff = 2.0
        activation_filter_min_alpha = 0.2
        activation_filter_state = None
        dt = self.env.config.simulator.config.sim.decimation / self.env.config.simulator.config.sim.fps

        # Warmup phase to stabilize the controller without advancing reference motion
        warmup_steps = 3
        obs = self.handle_reset(done_indices)
        self.env.motion_manager.disable_reset_track = True
        for _ in range(warmup_steps):
            teacher_actions = self.model.act(obs)
            obs, rewards, dones, terminated, extras = self.env_step(teacher_actions)
            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices.squeeze(-1)
        self.env.motion_manager.disable_reset_track = False

        while self.config.max_eval_steps is None or step < self.config.max_eval_steps:
            obs = self.handle_reset(done_indices)
            teacher_actions = self.model.act(obs)
            a = self.env.simulator._update_activation(teacher_actions)
            obs, rewards, dones, terminated, extras = self.env_step(teacher_actions)
            self.env.simulator._last_activations = a

            # student_actions = sim._compute_actions_from_tau(pd_tau_des, q, qd)
            # obs, rewards, dones, terminated, extras = self.env_step(student_actions)
            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices.squeeze(-1)
            step += 1
    def draw_plan(self, df_policy, states_block):
        from isaacsim.util.debug_draw import _debug_draw
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_lines()
        
        world_joints = df_policy.get_world_joints_from_pred(states_block) # (B, T, J, 3)
        B_size, T_size, J_size, _ = world_joints.shape
        
        start_points = []
        end_points = []
        colors = []
        sizes = []
        
        vis_step_stride = 1
        env_limit = min(B_size, 1) 
        
        for b in range(env_limit):
            for j in range(J_size):
                for t in range(0, T_size - 1, vis_step_stride):
                    p1 = world_joints[b, t, j]
                    p2 = world_joints[b, t+1, j]
                    
                    start_points.append(p1.tolist())
                    end_points.append(p2.tolist())
                    
                    # Color gradient: Red (start) -> Blue (end)
                    alpha = t / max(T_size - 1, 1)
                    color = (1.0 - alpha, 0.0, alpha, 1.0) # RGBA tuple
                    
                    colors.append(color)
                    sizes.append(2.0)
        
        if len(start_points) > 0:
            draw.draw_lines(start_points, end_points, colors, sizes)
        
        # --- Visualization End ---
    def _plot_action_timeseries(self, plan_actions, model_actions, out_path: Path, *, title: str):
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception as e:
            print(f"Action visualization skipped (matplotlib unavailable): {e}")
            return None

        plan_np = plan_actions.detach().cpu().float().numpy()
        model_np = model_actions.detach().cpu().float().numpy()

        if plan_np.ndim != 2 or model_np.ndim != 2:
            raise ValueError(f"Expected (T, A) actions. Got plan={plan_np.shape}, model={model_np.shape}")

        T, A = plan_np.shape
        if model_np.shape != (T, A):
            raise ValueError(f"Plan/model actions must match shapes. Got plan={plan_np.shape}, model={model_np.shape}")

        plan_valid = np.isfinite(plan_np).all(axis=1)
        model_valid = np.isfinite(model_np).all(axis=1)

        plan_delta = np.full((T,), np.nan, dtype=np.float32)
        model_delta = np.full((T,), np.nan, dtype=np.float32)
        for t in range(1, T):
            if plan_valid[t] and plan_valid[t - 1]:
                plan_delta[t] = np.linalg.norm(plan_np[t] - plan_np[t - 1])
            if model_valid[t] and model_valid[t - 1]:
                model_delta[t] = np.linalg.norm(model_np[t] - model_np[t - 1])

        plan_model_dist = np.full((T,), np.nan, dtype=np.float32)
        both_valid = plan_valid & model_valid
        if both_valid.any():
            plan_model_dist[both_valid] = np.linalg.norm(plan_np[both_valid] - model_np[both_valid], axis=1)

        fig, axs = plt.subplots(
            3,
            1,
            figsize=(14, 10),
            constrained_layout=True,
            gridspec_kw={"height_ratios": [2.0, 2.0, 1.0]},
        )
        fig.suptitle(title)

        im0 = axs[0].imshow(
            plan_np.T,
            aspect="auto",
            interpolation="nearest",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
        )
        axs[0].set_ylabel("Action Dim")
        axs[0].set_title("Planned Action (executed)")
        fig.colorbar(im0, ax=axs[0], fraction=0.025, pad=0.01)

        im1 = axs[1].imshow(
            model_np.T,
            aspect="auto",
            interpolation="nearest",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
        )
        axs[1].set_ylabel("Action Dim")
        axs[1].set_title("Actor Action self.model.act(obs)")
        fig.colorbar(im1, ax=axs[1], fraction=0.025, pad=0.01)

        axs[2].plot(plan_delta, label="||a_plan[t]-a_plan[t-1]||")
        axs[2].plot(model_delta, label="||a_model[t]-a_model[t-1]||")
        axs[2].plot(plan_model_dist, label="||a_plan[t]-a_model[t]||")
        axs[2].set_xlabel("Time Step")
        axs[2].set_ylabel("L2 Norm")
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(loc="upper right")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return out_path

    def _stack_action_history(self, history, action_dim: int):
        rows = []
        for x in history:
            if x is None:
                rows.append(torch.full((action_dim,), float("nan")))
            else:
                rows.append(x)
        if len(rows) == 0:
            return None
        return torch.stack(rows, dim=0)
    @torch.no_grad()
    def evaluate_policy_diffusion_forcing(self):
        from protomotions.diffusion_forcing.diffusion_forcing import DiffusionForcing, rotation_6d_to_matrix
        
        self.eval()
        done_indices = None
        step = 0
        sim = self.env.simulator
        
        # Checkpoint path
        checkpoint_path = "checkpoints/diffusion_forcing/checkpoint_best.pt"
        if not Path(checkpoint_path).exists():
            print(f"Warning: Checkpoint not found at {checkpoint_path}. Trying fallback.")
            return

        print(f"Initializing Diffusion Forcing Policy from {checkpoint_path}")
        df_policy = DiffusionForcing(
            checkpoint_path,
            device=self.device,
            num_envs=self.env.num_envs,
            history_len=4,
        )
        
        # Reset Env
        obs = self.handle_reset(done_indices)
        
        state_dim = 335
        action_dim = 50
        
        print(f"DF Policy Dims: State={state_dim}, Action={action_dim}")
        df_policy.load(state_dim, action_dim)
        df_policy.reset()

        horizon = 24
        exec_step = 8
        warmup_steps = 2
        viz_enabled = bool(getattr(self.config, "action_viz_enable", True))
        viz_env_index = int(getattr(self.config, "action_viz_env_index", 0))
        viz_max_steps = int(getattr(self.config, "action_viz_max_steps", 2000))
        viz_plot_every = int(getattr(self.config, "action_viz_plot_every", 512))
        viz_dir = Path(getattr(self.config, "action_viz_dir", "outputs/action_viz"))
        plan_action_hist = []
        model_action_hist = []
        viz_action_dim = action_dim

        for _ in range(warmup_steps):
            action = self.model.act(obs) 
            obs, rewards, dones, terminated, extras = self.env_step(action.clone())
                
            next_state_features = df_policy.get_current_state(sim)
            df_policy.record(next_state_features, action)
            
            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices.squeeze(-1)
            
            if len(done_indices) > 0:
                obs = self.handle_reset(done_indices)
                df_policy.reset()
            step += 1

        print(f"Diffusion Forcing: Horizon={horizon}, Exec Step={exec_step}")

        num_envs = self.env.num_envs
        goal_dist = torch.rand(num_envs, device=self.device) * 3.0 + 2.0
        goal_theta = torch.rand(num_envs, device=self.device) * 2 * math.pi
        global_goal_position = torch.stack([
            goal_dist * torch.cos(goal_theta),
            goal_dist * torch.sin(goal_theta),
            torch.zeros(num_envs, device=self.device)
        ], dim=-1)

        while self.config.max_eval_steps is None or step < self.config.max_eval_steps:
          
            def guidance_fn(x_pred):
                # x_pred: (B, T_full, D)
                x_pred_future = x_pred[:, df_policy.context_len:]
                
                state_mean = df_policy.x_mean[df_policy.action_dim:]
                state_std = df_policy.x_std[df_policy.action_dim:]
                
                pred_states = x_pred_future[..., df_policy.action_dim:]
                pred_states = df_policy._unnormalize(pred_states, state_mean, state_std)
                
                world_joints = df_policy.get_world_joints_from_pred(pred_states) # (B, T, J, 3)
                root_pos = world_joints[..., 0, :] # (B, T, 3)
                
                goal_pos_expanded = global_goal_position.unsqueeze(1) # (B, 1, 3)
                # Only use XY for position cost
                pos_cost = (root_pos[..., :2] - goal_pos_expanded[..., :2]).norm(dim=-1).mean()
                
                root_rot_6d = pred_states[..., 3:9]
                root_rot_mat = rotation_6d_to_matrix(root_rot_6d) # (B, T, 3, 3)
                
                forward_dir = root_rot_mat[..., 1] # (B, T, 3)
                forward_dir_xy = forward_dir[..., :2]
                forward_dir_xy = F.normalize(forward_dir_xy, dim=-1)
                
                goal_vec = goal_pos_expanded - root_pos
                goal_dir_xy = goal_vec[..., :2]
                goal_dir_xy = F.normalize(goal_dir_xy, dim=-1)
                
                cos_sim = (forward_dir_xy * goal_dir_xy).sum(dim=-1)
                ori_cost = (1.0 - cos_sim).mean()
                
                loss = -(pos_cost * 2.0 + ori_cost * 1.0)
                
                return loss * 20.0

            actions_block, states_block = df_policy.plan(horizon=horizon)
            
            curr_root_pos = sim.get_root_state().root_pos
            dist_to_goal = (curr_root_pos[:, :2] - global_goal_position[:, :2]).norm(dim=-1)
            
            reached = dist_to_goal < 0.5
            if reached.any():
                idxs = torch.nonzero(reached).squeeze(-1)
                n_reach = len(idxs)
                new_dist = torch.rand(n_reach, device=self.device) * 3.0 + 2.0
                new_theta = torch.rand(n_reach, device=self.device) * 2 * math.pi
                
                global_goal_position[idxs, 0] = curr_root_pos[idxs, 0] + new_dist * torch.cos(new_theta)
                global_goal_position[idxs, 1] = curr_root_pos[idxs, 1] + new_dist * torch.sin(new_theta)
                
                print(f"Envs {idxs.tolist()} reached goal. New goals set.")

            self.draw_plan(df_policy, states_block)

            steps_to_exec = min(horizon, exec_step)
            for h in range(steps_to_exec):
                if self.config.max_eval_steps is not None and step >= self.config.max_eval_steps:
                    break
                action = torch.clamp(actions_block[:, h], -1.0, 1.0)
                # model_action = torch.clamp(self.model.act(obs), -1.0, 1.0)
                # if viz_enabled and len(plan_action_hist) < viz_max_steps:
                #     plan_action_hist.append(action[viz_env_index].detach().cpu())
                #     model_action_hist.append(model_action[viz_env_index].detach().cpu())
                obs, rewards, dones, terminated, extras = self.env_step(action.clone())
                next_state_features = df_policy.get_current_state(sim)
                df_policy.record(next_state_features, action)
                all_done_indices = dones.nonzero(as_tuple=False)
                done_indices = all_done_indices.squeeze(-1)
                if len(done_indices) > 0:
                    # if viz_enabled and (done_indices == viz_env_index).any().item():
                    #     if len(plan_action_hist) < viz_max_steps:
                    #         plan_action_hist.append(None)
                    #         model_action_hist.append(None)
                    obs = self.handle_reset(done_indices)
                    df_policy.reset()
                step += 1
                print(step)
        #         if viz_enabled and viz_plot_every > 0 and step % viz_plot_every == 0 and len(plan_action_hist) > 2:
        #             plan_stack = self._stack_action_history(plan_action_hist, viz_action_dim)
        #             model_stack = self._stack_action_history(model_action_hist, viz_action_dim)
        #             if plan_stack is not None and model_stack is not None:
        #                 out_path = viz_dir / f"action_timeseries_env{viz_env_index}_{step:07d}.png"
        #                 self._plot_action_timeseries(
        #                     plan_stack,
        #                     model_stack,
        #                     out_path,
        #                     title=f"Action Timeseries (env={viz_env_index}) step={step}",
        #                 )
        # if viz_enabled and len(plan_action_hist) > 2:
        #     plan_stack = self._stack_action_history(plan_action_hist, viz_action_dim)
        #     model_stack = self._stack_action_history(model_action_hist, viz_action_dim)
        #     if plan_stack is not None and model_stack is not None:
        #         out_path = viz_dir / f"action_timeseries_env{viz_env_index}_final_{step:07d}.png"
        #         self._plot_action_timeseries(
        #             plan_stack,
        #             model_stack,
        #             out_path,
        #             title=f"Action Timeseries (env={viz_env_index}) final step={step}",
        #         )
