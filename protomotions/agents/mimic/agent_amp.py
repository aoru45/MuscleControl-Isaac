import torch
import logging
from pathlib import Path

from rich.progress import track

from typing import List, Tuple, Dict, Optional

from protomotions.envs.mimic.env import Mimic as MimicEnv
from protomotions.agents.amp.agent import AMP

log = logging.getLogger(__name__)


class Mimic(AMP):
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
        # Note: broadcast returns a list/tensor usually? No, fabric.broadcast handles object pickling for non-tensors.
        is_sharded = (local_motion_file != root_motion_file)
        
        # Always run this check to ensure all ranks execute the same number of collectives (all_gather)
        # preventing desynchronization with downstream gathers.
        local_count = torch.tensor([num_motions], device=self.device)
        all_counts = self.fabric.all_gather(local_count)
        
        if not is_sharded:
            if isinstance(all_counts, list):
                all_counts = torch.stack(all_counts)
            if not (all_counts == num_motions).all():
                is_sharded = True
                print(f"[Rank {global_rank}] Detected sharded mode via num_motions mismatch.")

        if self.env.config.motion_manager.fixed_motion_id is not None:
            motion_ids = torch.tensor(
                [self.env.config.motion_manager.fixed_motion_id], device=self.device
            )
            requires_scene = self.env.get_motion_requires_scene(motion_ids)
            return [(motion_ids, requires_scene)], 1, False # Fixed ID is treated as global-ish

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

        motion_range = torch.arange(start_motion, end_motion, device=self.device)

        motion_map = []
        for i in range(0, len(motion_range), self.num_envs):
            batch_motion_ids = motion_range[i : i + self.num_envs]
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
            "evaluated": torch.zeros(num_motions, device=self.device, dtype=torch.bool)
        }
        for k in self.config.eval_metric_keys:
            metrics[k] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_max"] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_min"] = 3 * torch.ones(num_motions, device=self.device)

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
                description=f"Evaluating... {remaining_motions} motions remain..."
            )
        else:
            outer_iter_iterator = range(max_iters)

        for outer_iter in outer_iter_iterator:
            motion_pointer = outer_iter % num_outer_iters
            motion_ids, requires_scene = motion_map[motion_pointer]
            num_motions_this_iter = len(motion_ids)
            
            # Important: prevent double-counting metrics if we are just padding execution
            is_valid_iter = outer_iter < num_outer_iters
            
            if is_valid_iter:
                metrics["evaluated"][motion_ids] = True

            self.env.agent_in_scene[:] = False
            self.env.motion_manager.motion_ids[:num_motions_this_iter] = motion_ids
            self.env.agent_in_scene[:num_motions_this_iter] = requires_scene
            self.env.force_respawn_on_flat = True

            env_ids = torch.arange(0, num_motions_this_iter, dtype=torch.long, device=self.device)
            dt: float = self.env.dt
            motion_lengths = self.motion_lib.get_motion_length(motion_ids)
            motion_num_frames = (motion_lengths / dt).floor().long()
            max_len = (
                motion_num_frames.max().item()
                if self.config.eval_length is None
                else self.config.eval_length
            )

            for eval_episode in range(self.config.eval_num_episodes):
                elapsed_time = torch.rand_like(self.motion_lib.state.motion_lengths[motion_ids]) * dt
                self.env.motion_manager.motion_times[:num_motions_this_iter] = elapsed_time
                self.env.motion_manager.reset_track_steps.reset_steps(env_ids)
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
                            metric = value[:num_motions_this_iter]
                            metrics[k][motion_ids[clip_not_done]] += metric[clip_not_done]
                            metrics[f"{k}_max"][motion_ids[clip_not_done]] = torch.maximum(
                                metrics[f"{k}_max"][motion_ids[clip_not_done]],
                                metric[clip_not_done]
                            )
                            metrics[f"{k}_min"][motion_ids[clip_not_done]] = torch.minimum(
                                metrics[f"{k}_min"][motion_ids[clip_not_done]],
                                metric[clip_not_done]
                            )

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
            # Save failed motions per rank (always safe)
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
        
        # For early stopping, we need a global consensus.
        # We broadcast the metrics from rank 0, OR we aggregate.
        # Ideally, `post_epoch_logging` handles aggregation. But here we need to decide NOW.
        # Let's aggregate tracking_success_rate for decision making.
        
        local_success = torch.tensor(to_log.get("eval/tracking_success_rate", 0.0), device=self.device)
        global_success = self.fabric.all_reduce(local_success, reduce_op="mean")
        
        if self.config.training_early_termination.early_terminate_success_rate is not None:
             stop_early = stop_early and (global_success >= self.config.training_early_termination.early_terminate_success_rate)
        
        # Only rank 0 controls the final decision flag usually, but all ranks need to know to stop?
        # Typically `terminate_early()` sets a flag.
        
        if stop_early and self.fabric.global_rank == 0:
            print("Stopping early! Target error reached")
            self.best_evaluated_score = global_success
            self.save(new_high_score=True)
            self.terminate_early()
        
        # Sync stop signal? `terminate_early` sets `self._should_stop`.
        # `fit` loop checks `self.should_stop` which broadcasts `_should_stop`. 
        # So we only need Rank 0 to call it.

        self.env.disable_reset = False
        self.env.motion_manager.disable_reset_track = False
        self.env.force_respawn_on_flat = False
        all_ids = torch.arange(0, self.num_envs, dtype=torch.long, device=self.device)
        self.env.motion_manager.reset_envs(all_ids)
        self.force_full_restart = True
        
        # Return local logs. Agent's post_epoch_logging will aggregate them.
        return to_log, to_log.get("eval/tracking_success_rate", to_log.get("eval/cartesian_err", None))
