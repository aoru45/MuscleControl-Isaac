from isaaclab.app import AppLauncher
#0-Eyes_Japan_Dataset_kanno_walk-15-pull-kanno_poses
headless = False
app_launcher = AppLauncher({"headless": headless})
simulation_app = app_launcher.app

import torch
import time
from rich.progress import track
import hydra
from omegaconf import OmegaConf
import protomotions.utils.config_utils
from protomotions.simulator.isaaclab.config import IsaacLabSimulatorConfig, IsaacLabSimParams
from protomotions.simulator.isaaclab.simulator import IsaacLabSimulator
from protomotions.simulator.base_simulator.config import (
    RobotConfig,
    RobotAssetConfig,
    InitState,
    ControlConfig,
    ControlType,
)
from protomotions.envs.base_env.env_utils.terrains.flat_terrain import FlatTerrain
from protomotions.envs.base_env.env_utils.terrains.terrain_config import TerrainConfig

# Create robot configuration
with hydra.initialize(version_base=None, config_path="../../protomotions/config"):
    cfg = hydra.compose(config_name="robot/bio_act")
    robot_config = RobotConfig.from_dict(OmegaConf.to_container(cfg.robot, resolve=True))

# Create simulator configuration
simulator_config = IsaacLabSimulatorConfig(
    sim=IsaacLabSimParams(
        fps=240,
        decimation=4,
    ),
    headless=headless,  # Set to True for headless mode
    robot=robot_config,
    num_envs=1,  # Number of parallel environments
    experiment_name="bio_act_isaaclab_example",
    w_last=False,  # IsaacLab uses wxyz quaternions
)

device = torch.device("cuda")

# Create a flat terrain using the default config
terrain_config = TerrainConfig()
terrain = FlatTerrain(config=terrain_config, num_envs=simulator_config.num_envs, device=device)

# Create and initialize the simulator
simulator = IsaacLabSimulator(config=simulator_config, terrain=terrain, scene_lib=None, visualization_markers=None, device=device, simulation_app=simulation_app)
simulator.on_environment_ready()

# Get robot default state
default_state = simulator.get_default_state()
# Set the robot to a new random position above the ground
root_pos = torch.zeros(simulator_config.num_envs, 3, device=device)
root_pos[:, :2] = terrain.sample_valid_locations(simulator_config.num_envs).view(-1, 2)
root_pos[:, 2] = 1.0
default_state.root_pos[:] = root_pos

# Reset the robots
simulator.reset_envs(default_state, env_ids=torch.arange(simulator_config.num_envs, device=device))

# Run the simulation loop
try:
    for _ in track(range(10000), description="Performing warmup steps"):
        actions = 2*torch.rand(simulator_config.num_envs, simulator_config.robot.number_of_actions, device=device)-1

        simulator._update_activation(actions)
        simulator.step(actions)

        body_state = simulator._get_simulator_bodies_state()
        if torch.isnan(body_state.rigid_body_pos).any():
            print("NaN values detected in body positions")
            exit()
    print("Warmup complete")
    # Run benchmark
    start_time = time.perf_counter()
    num_steps = 1000
    
    for _ in track(range(num_steps), description="Running benchmark"):
        actions = torch.randn(simulator_config.num_envs, simulator_config.robot.number_of_actions, device=device)
        simulator.step(actions)
        
    total_time = time.perf_counter() - start_time
    avg_time_per_step = total_time / num_steps
    
    print(f"\nBenchmark Results:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per step: {avg_time_per_step*1000:.2f} ms")
    simulator.close()
except Exception as e:
    print(f"\nAn error occurred: {e}")
finally:
    simulator.close()
