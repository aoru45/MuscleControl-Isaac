import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext, PhysxCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from easydict import EasyDict
from protomotions.envs.base_env.env_utils.terrains.terrain import Terrain
from protomotions.utils.scene_lib import SceneLib
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from protomotions.simulator.isaaclab.utils.scene import SceneCfg
from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.config import (
    MarkerState,
    ControlType,
    VisualizationMarker,
    SimBodyOrdering,
    SimulatorConfig
)
from protomotions.simulator.base_simulator.robot_state import RobotState
from protomotions.simulator.base_simulator.simulator import soft_bound
from protomotions.utils.muscle_control import MuscleController
# from protomotions.utils.muscle_control_isaaclab import MuscleControllerIsaacLab as MuscleController
from protomotions.simulator.isaaclab.utils.muscle_mesh import build_muscle_mesh, create_muscle_mesh_prim, update_muscle_mesh_points


def optimize_act(JtA: torch.Tensor, b: torch.Tensor, tau: torch.Tensor, method: str = "pgd", max_iter = 50, last_act = None, debug_no_b: bool = False, pgd_steps: int = 1500, pgd_lr: Optional[float] = None, power_iter: int = 12):
    # Detach inputs to treat them as constants
    JtA = JtA.detach()
    b = b.detach()
    tau_target = tau.detach()

    B, M, D = JtA.shape
    if last_act is not None and last_act.ndim != 2:
        if last_act.numel() == B * M:
            last_act = last_act.reshape(B, M)
        else:
            raise RuntimeError(f"last_act shape {tuple(last_act.shape)} incompatible with (B,M)=({B},{M})")

    if method == "lbfgs":
        # if last_act is not None:
        #     x = torch.logit(last_act).clone().detach().requires_grad_(True)
        # else:
        x = torch.zeros((B, M), device=JtA.device, requires_grad=True)
        optimizer = torch.optim.LBFGS([x], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad()
            a = torch.sigmoid(x)
            if debug_no_b:
                tau_pred = torch.einsum("bm,bmd->bd", a, JtA)
            else:
                tau_pred = torch.einsum("bm,bmd->bd", a, JtA) + b
            loss = (tau_pred - tau_target).pow(2).sum()
            if last_act is not None:
                loss += 0.5 * (a - last_act).pow(2).sum()
            loss.backward()
            return loss

        with torch.enable_grad():
            optimizer.step(closure)

        with torch.no_grad():
            a_final = torch.sigmoid(x)
            if debug_no_b:
                tau_final = torch.einsum("bm,bmd->bd", a_final, JtA)
            else:
                tau_final = torch.einsum("bm,bmd->bd", a_final, JtA) + b

        return a_final, tau_final

    elif method == "ls":
        target_torque = (tau_target if debug_no_b else (tau_target - b)).unsqueeze(-1) # (B, D, 1)
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
            if debug_no_b:
                tau_pred = torch.einsum("bm,bmd->bd", a, JtA)
            else:
                tau_pred = torch.einsum("bm,bmd->bd", a, JtA) + b
            loss = (tau_pred - tau_target).pow(2).sum()
            loss.backward()
            return loss

        with torch.enable_grad():
            optimizer.step(closure)

        with torch.no_grad():
            a_final = torch.sigmoid(x)
            if debug_no_b:
                tau_final = torch.einsum("bm,bmd->bd", a_final, JtA)
            else:
                tau_final = torch.einsum("bm,bmd->bd", a_final, JtA) + b
            
        return a_final, tau_final

    elif method in ["pgd", "nnls"]:
        J = JtA.transpose(1, 2)  # (B, D, M)
        target = tau_target if debug_no_b else (tau_target - b)
        target = target.unsqueeze(-1)  # (B, D, 1)
        if pgd_lr is None:
            v = torch.randn((B, J.shape[-1], 1), device=J.device, dtype=J.dtype)
            v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
            for _ in range(power_iter):
                v = torch.bmm(J.transpose(1, 2), torch.bmm(J, v))
                v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
            Jv = torch.bmm(J, v)
            sigma = Jv.norm(dim=1, keepdim=False).clamp(min=1e-6)
            L = 2.0 * sigma * sigma
            pgd_lr = (1.0 / L).clamp(max=1e-2).view(B, 1)
        else:
            pgd_lr = torch.tensor(pgd_lr, device=JtA.device, dtype=JtA.dtype).view(1, 1)
        try:
            a_init = torch.linalg.lstsq(J, target).solution.squeeze(-1)
        except RuntimeError:
            a_init = torch.zeros((B, M), device=JtA.device, dtype=JtA.dtype)
        if last_act is not None:
            a0 = last_act.detach()
        else:
            a0 = a_init.detach()
        if a0.ndim != 2:
            if a0.numel() == B * M:
                a0 = a0.reshape(B, M)
            else:
                raise RuntimeError(f"init activation shape {tuple(a0.shape)} incompatible with (B,M)=({B},{M})")
        if method == "pgd":
            x = torch.clamp(a0, 0.0, 1.0)
        else:
            x = torch.clamp(a0, min=0.0)
        y = x.clone()
        t = 1.0

        with torch.enable_grad():
            for _ in range(pgd_steps):
                y = y.detach().requires_grad_(True)
                if y.ndim != 2:
                    raise RuntimeError(f"PGD activation has shape {tuple(y.shape)}; expected (B,M)=({B},{M})")
                if debug_no_b:
                    tau_pred = torch.einsum("bm,bmd->bd", y, JtA)
                else:
                    tau_pred = torch.einsum("bm,bmd->bd", y, JtA) + b
                loss = (tau_pred - tau_target).pow(2).sum()
                loss.backward()
                with torch.no_grad():
                    x_next = y - pgd_lr * y.grad
                    if method == "pgd":
                        x_next.clamp_(0.0, 1.0)
                    else:
                        x_next.clamp_(min=0.0)
                    t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
                    y = x_next + ((t - 1.0) / t_next) * (x_next - x)
                    x = x_next
                    t = t_next

        with torch.no_grad():
            a_final = x.clamp(0.0, 1.0) if method == "pgd" else x.clamp(min=0.0)
            if debug_no_b:
                tau_final = torch.einsum("bm,bmd->bd", a_final, JtA)
            else:
                tau_final = torch.einsum("bm,bmd->bd", a_final, JtA) + b
        return a_final, tau_final
    else:
        raise ValueError(f"Unknown optimization method: {method}")

class IsaacLabSimulator(Simulator):
    # =====================================================
    # Group 1: Initialization & Configuration
    # =====================================================
    def __init__(
        self,
        config: SimulatorConfig,
        terrain: Terrain,
        device: torch.device,
        simulation_app: Any,
        scene_lib: Optional[SceneLib] = None,
        visualization_markers: Optional[Dict[str, VisualizationMarker]] = None,
    ) -> None:
        """
        Initialize the IsaacLabSimulator.

        Parameters:
            config (SimulatorConfig): The configuration dictionary.
            terrain (Terrain): Terrain data for simulation.
            device (torch.device): Device to use for computation.
            simulation_app (Any): The simulation application instance.
            scene_lib (Optional[SceneLib], optional): The scene library containing scene and object data.
            visualization_markers (Optional[Dict[str, VisualizationMarker]], optional): Configuration for visualization markers.
        """
        super().__init__(
            config=config,
            scene_lib=scene_lib,
            terrain=terrain,
            visualization_markers=visualization_markers,
            device=device,
        )

        sim_cfg = sim_utils.SimulationCfg(
            device=str(device),
            dt=1.0 / self.config.sim.fps,
            render_interval=self.config.sim.decimation,
            physx=PhysxCfg(
                solver_type=self.config.sim.physx.solver_type,
                max_position_iteration_count=self.config.sim.physx.num_position_iterations,
                max_velocity_iteration_count=self.config.sim.physx.num_velocity_iterations,
                bounce_threshold_velocity=self.config.sim.physx.bounce_threshold_velocity,
                gpu_max_rigid_contact_count=self.config.sim.physx.gpu_max_rigid_contact_count,
                gpu_found_lost_pairs_capacity=self.config.sim.physx.gpu_found_lost_pairs_capacity,
                gpu_found_lost_aggregate_pairs_capacity=self.config.sim.physx.gpu_found_lost_aggregate_pairs_capacity,
            ),
        )
        self._simulation_app = simulation_app
        self._sim = SimulationContext(sim_cfg)
        self._sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

        scene_cfg = self._get_scene_cfg()

        self._scene = InteractiveScene(scene_cfg)
        if not self.headless:
            self._setup_keyboard()
        print("[INFO]: Setup complete...")

        self._robot = self._scene["robot"]
        self._contact_sensor = self._scene["contact_sensor"]
        self._object = []
        if self.scene_lib is not None and self.scene_lib.total_spawned_scenes > 0:
            for obj_idx in range(self.scene_lib.num_objects_per_scene):
                self._object.append(self._scene[f"object_{obj_idx}"])
        

        if visualization_markers:
            self._build_markers(visualization_markers)
        # self._setup_foot_contact_visualization()
        self._sim.reset()

        self.muscle_ctl = None
        if self.robot_config.asset.robot_type == 'act_humanoid':
            rig_path = os.path.join('protomotions/data/assets/mjcf/bio.xml')
            muscle_path = os.path.join('protomotions/data/assets/muscle284.xml')
            self.muscle_ctl = MuscleController(muscle_xml_path=muscle_path, device=self.device, rig_path=rig_path)
            # muscle_scale_map = {
            #     "L_Extensor_Digitorum_Longus": 12.00,
            #     "L_Extensor_Digitorum_Longus1": 12.00,
            #     "L_Extensor_Digitorum_Longus2": 12.00,
            #     "L_Extensor_Digitorum_Longus3": 12.00,
            #     "L_Flexor_Digitorum_Longus": 12.00,
            #     "L_Flexor_Digitorum_Longus1": 12.00,
            #     "L_Flexor_Digitorum_Longus2": 12.00,
            #     "L_Flexor_Digitorum_Longus3": 12.00,
            #     "L_Flexor_Digiti_Minimi_Brevis_Foot": 12.00,
            #     "L_Flexor_Hallucis": 8.00,
            #     "L_Flexor_Hallucis1": 8.00,
            #     "L_Extensor_Hallucis_Longus": 8.00,
            #     "L_Tibialis_Anterior": 14.00,
            #     "L_Tibialis_Posterior": 5.00,
            #     "L_Peroneus_Longus": 11.00,
            #     "L_Peroneus_Brevis": 11.00,
            #     "L_Peroneus_Tertius": 12.00,
            #     "L_Peroneus_Tertius1": 12.00,
            #     "L_Plantaris": 12.00,
            #     "L_Soleus": 11.00,
            #     "L_Soleus1": 9.00,
            #     "L_Gastrocnemius_Medial_Head": 3.00,
            #     "L_Gastrocnemius_Lateral_Head": 3.00,
            #     "L_Semimembranosus": 3.00,
            #     "L_Semitendinosus": 2.00,
            #     "L_Psoas_Major1": 6.00,
            #     "L_Psoas_Major2": 4.50,
            #     "L_Psoas_Minor": 3.00,
            #     "L_Pectineus": 4.00,
            #     "L_Gluteus_Medius": 4.00,
            #     "L_Gluteus_Medius1": 4.00,
            #     "L_Gluteus_Medius2": 4.00,
            #     "L_Gluteus_Medius3": 4.00,
            #     "L_Gluteus_Maximus1": 3.00,
            #     "L_Gluteus_Maximus2": 3.00,
            #     "L_Gluteus_Maximus3": 3.00,
            #     "L_Gluteus_Maximus4": 3.00,
            #     "L_Quadratus_Lumborum1": 2.50,

            #     "R_Extensor_Digitorum_Longus": 14.00,
            #     "R_Extensor_Digitorum_Longus1": 14.00,
            #     "R_Extensor_Digitorum_Longus2": 14.00,
            #     "R_Extensor_Digitorum_Longus3": 16.00,
            #     "R_Flexor_Digitorum_Longus": 20.00,
            #     "R_Flexor_Digitorum_Longus1": 20.00,
            #     "R_Flexor_Digitorum_Longus2": 16.00,
            #     "R_Flexor_Digitorum_Longus3": 16.00,
            #     "R_Flexor_Digiti_Minimi_Brevis_Foot": 12.00,
            #     "R_Flexor_Hallucis": 8.00,
            #     "R_Flexor_Hallucis1": 8.00,
            #     "R_Extensor_Hallucis_Longus": 8.00,
            #     "R_Tibialis_Anterior": 20.00,
            #     "R_Tibialis_Posterior": 5.00,
            #     "R_Peroneus_Longus": 12.00,
            #     "R_Peroneus_Brevis": 12.00,
            #     "R_Peroneus_Tertius": 12.00,
            #     "R_Peroneus_Tertius1": 12.00,
            #     "R_Plantaris": 12.00,
            #     "R_Soleus": 8.00,
            #     "R_Soleus1": 6.00,
            #     "R_Gastrocnemius_Medial_Head": 3.00,
            #     "R_Gastrocnemius_Lateral_Head": 3.00,
            #     "R_Semimembranosus": 3.00,
            #     "R_Semitendinosus": 2.00,
            #     "R_Psoas_Major1": 6.00,
            #     "R_Psoas_Major2": 4.50,
            #     "R_Psoas_Minor": 3.00,
            #     "R_Pectineus": 4.00,
            #     "R_Gluteus_Medius": 4.00,
            #     "R_Gluteus_Medius1": 4.00,
            #     "R_Gluteus_Medius2": 4.00,
            #     "R_Gluteus_Medius3": 4.00,
            #     "R_Gluteus_Maximus1": 3.00,
            #     "R_Gluteus_Maximus2": 3.00,
            #     "R_Gluteus_Maximus3": 3.00,
            #     "R_Gluteus_Maximus4": 3.00,
            #     "R_Latissimus_Dorsi3": 3.00,
            #     "R_Serratus_Posterior_Inferior": 3.00,
            # }
            # self.muscle_ctl.set_muscle_scales(muscle_scale_map, max_scale=20.0)
            # self.muscle_ctl.set_global_scale(1.5)
            # self._muscle_scale_suggest = {}
            # self._muscle_scale_applied = True
            # self._muscle_scale_max = 20.0
            self.muscle_ctl.prepare(self.robot_config.body_names, self.robot_config.dof_names)
            self._muscle_mesh_skin = None

    def _setup_foot_contact_visualization(self) -> None:
        self._foot_contact_markers = None
        if self.headless:
            return
        left_foot_names = ["TalusL", "FootThumbL", "FootPinkyL"]
        right_foot_names = ["TalusR", "FootThumbR", "FootPinkyR"]
        left_foot_ids = [self.robot_config.body_names.index(name) for name in left_foot_names]
        right_foot_ids = [self.robot_config.body_names.index(name) for name in right_foot_names]

        self._foot_contact_body_ids_common = torch.tensor(
            left_foot_ids + right_foot_ids, device=self.device, dtype=torch.long
        )

        marker_geom_cfg = sim_utils.SphereCfg(
            radius=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(174/255, 104/255, 231/255),
                opacity=0.85,
            ),
        )
        self._foot_contact_base_scale = torch.tensor(
            [0.15, 0.15, 0.003], device=self.device, dtype=torch.float
        )

        marker_obj_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/FootContact",
            markers={"contact": marker_geom_cfg},
        )
        self._foot_contact_markers = VisualizationMarkers(marker_obj_cfg)

        if self.config.w_last:
            self._foot_contact_identity_quat = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float
            )
        else:
            self._foot_contact_identity_quat = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float
            )

    def _draw_foot_contacts(self) -> None:
        """Update foot contact markers each render step."""
        if self.headless or self._foot_contact_markers is None:
            return

        bodies_state = self.get_bodies_state()
        contact_forces = self.get_bodies_contact_buf()

        foot_pos_world_left = bodies_state.rigid_body_pos[:, self._foot_contact_body_ids_common[:3], :].clone()
        foot_pos_world_right = bodies_state.rigid_body_pos[:, self._foot_contact_body_ids_common[3:], :].clone()

        foot_pos_world_left[..., -1] = 0.
        foot_pos_world_right[..., -1] = 0.

        foot_forces = contact_forces[:, self._foot_contact_body_ids_common, :] # (num_envs, 6, 3)

        foot_force_norm = torch.norm(foot_forces, dim=-1) # (num_envs, 6)
        foot_force_norm_left = foot_force_norm[:, :3].max(1).values # (num_envs,)
        foot_force_norm_right = foot_force_norm[:, 3:].max(1).values # (num_envs,)

        foot_in_contact_left = foot_force_norm_left > 1.0
        foot_in_contact_right = foot_force_norm_right > 1.0
        foot_pos_world = torch.stack([foot_pos_world_left.mean(1), foot_pos_world_right.mean(1)], dim=1) # (num_envs, 2, 3)

        foot_in_contact = torch.cat([foot_in_contact_left, foot_in_contact_right])

        translations = foot_pos_world.reshape(-1, 3)
        orientations = (
            self._foot_contact_identity_quat.view(1, 1, 4)
            .expand(self.num_envs, 2, 4)
            .reshape(-1, 4)
        )
        scales = (
            self._foot_contact_base_scale.view(1, 1, 3)
            .expand(self.num_envs, 2, 3)
            .reshape(-1, 3)
            .clone()
        )
        scales[~foot_in_contact.reshape(-1)] = 0.0

        self._foot_contact_markers.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
        )

    def _get_scene_cfg(self) -> SceneCfg:
        """
        Construct and return the scene configuration from the current config, scene library, and terrain.

        Returns:
            SceneCfg: The constructed scene configuration.
        """
        scene_cfgs = None
        if self.scene_lib is not None and self.scene_lib.total_spawned_scenes > 0:
            scene_cfgs, self._initial_scene_pos = self._preprocess_object_playground()

        scene_cfg = SceneCfg(
            config=self.config,
            robot_config=self.robot_config,
            num_envs=self.config.num_envs,
            env_spacing=2.0,
            scene_cfgs=scene_cfgs,
            terrain=self.terrain,
        )
        return scene_cfg

    def _preprocess_object_playground(self) -> Tuple[List[Any], torch.Tensor]:
        """
        Process and build the object playground from the scene library.

        Returns:
            Tuple[List[Any], torch.Tensor]: A tuple containing the object configurations and the initial object positions.
        """
        print("=========== Building object playground")

        objects_cfgs = []
        for _ in range(self.scene_lib.num_objects_per_scene):
            objects_cfgs.append([])
        initial_obj_pos = torch.zeros(
            (self.num_envs, self.scene_lib.num_objects_per_scene, 7),
            device=self.device,
            dtype=torch.float,
        )

        for scene_idx, scene_spawn_info in enumerate(self.scene_lib.scenes):
            scene_offset = self.scene_lib.scene_offsets[scene_idx]

            height_at_scene_origin = self.terrain.get_ground_heights(
                torch.tensor(
                    [[scene_offset[0], scene_offset[1]]],
                    device=self.device,
                    dtype=torch.float,
                )
            ).item()
            self._scene_position.append(
                torch.tensor(
                    [scene_offset[0], scene_offset[1], height_at_scene_origin],
                    device=self.device,
                    dtype=torch.float,
                )
            )
            self._object_dims.append([])

            for obj_idx, obj in enumerate(scene_spawn_info.objects):
                # Get the spawn info for this object which contains the correct ID
                object_spawn_info = next(
                    info
                    for info in self.scene_lib.object_spawn_list
                    if info.object_path == obj.object_path
                    and (info.is_first_instance or info.first_instance_id == info.id)
                )

                file_extension = object_spawn_info.object_path.split("/")[-1].split(
                    "."
                )[-1]

                assert file_extension in [
                    "usd",
                    "usda",
                    "urdf",
                ], f"Object asset [{obj.object_path}] must be a USD/URDF file"

                # Calculate the global position of the object
                global_object_position = torch.tensor(
                    [
                        scene_offset[0] + obj.translation[0],
                        scene_offset[1] + obj.translation[1],
                        0 + obj.translation[2],
                    ],
                    device=self.device,
                    dtype=torch.float,
                )

                initial_obj_pos[scene_idx, obj_idx, :3] = global_object_position
                initial_obj_pos[scene_idx, obj_idx, 3:] = torch.tensor(
                    [
                        obj.rotation[3],
                        obj.rotation[0],
                        obj.rotation[1],
                        obj.rotation[2],
                    ],
                    device=self.device,
                    dtype=torch.float,
                )  # Convert xyzw to wxyz

                main_dir_path = (
                    f"{os.path.dirname(os.path.abspath(__file__))}/../../../"
                )
                asset_path = Path(
                    os.path.join(main_dir_path, obj.object_path)
                ).resolve()

                # Common properties based on object options
                rigid_props = sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=obj.options.fix_base_link
                )
                collision_props = sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002,
                    rest_offset=0.0,
                )

                if file_extension == "urdf":
                    # Parse the URDF file
                    tree = ET.parse(asset_path)
                    root = tree.getroot()

                    # Get the box dimensions from the collision geometry
                    link = root.find("link")
                    collision = link.find("collision")
                    geometry = collision.find("geometry")
                    box = geometry.find("box")
                    size = box.get("size").split(" ")

                    spawn_cfg = sim_utils.CuboidCfg(
                        size=(
                            float(size[0]),
                            float(size[1]),
                            float(size[2]),
                        ),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.6, 0.2), metallic=0.2
                        ),
                        rigid_props=rigid_props,
                        mass_props=sim_utils.MassPropertiesCfg(
                            mass=1.0, density=obj.options.density
                        ),
                        collision_props=collision_props,
                    )
                else:
                    spawn_cfg = sim_utils.UsdFileCfg(
                        activate_contact_sensors=True,
                        usd_path=str(asset_path),
                        rigid_props=rigid_props,
                        collision_props=collision_props,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.2, 0.7, 0.3), metallic=0.2
                        ),
                    )
                objects_cfgs[obj_idx].append(spawn_cfg)

                object_dims = torch.tensor(
                    object_spawn_info.object_dims, device=self.device, dtype=torch.float
                )
                self._object_dims[-1].append(object_dims)
            self._object_dims[-1] = torch.stack(self._object_dims[-1]).reshape(
                self._num_objects_per_scene, -1
            )

        return objects_cfgs, initial_obj_pos

    def _setup_keyboard(self) -> None:
        """
        Set up keyboard callbacks for control using the Se2Keyboard interface.
        """
        from isaaclab.devices.keyboard.se2_keyboard import Se2Keyboard, Se2KeyboardCfg

        self.keyboard_interface = Se2Keyboard(Se2KeyboardCfg(sim_device=str(self.device)))
        self.keyboard_interface.add_callback("R", self._requested_reset)
        self.keyboard_interface.add_callback("U", self._update_inference_parameters)
        self.keyboard_interface.add_callback("L", self._toggle_video_record)
        self.keyboard_interface.add_callback(";", self._cancel_video_record)
        self.keyboard_interface.add_callback("Q", self.close)
        self.keyboard_interface.add_callback("O", self._toggle_camera_target)
        self.keyboard_interface.add_callback("J", self._push_robot)

    # =====================================================
    # Group 2: Environment Setup & Configuration
    # =====================================================
    def on_environment_ready(self) -> None:
        """
        Configure initial environment settings when the simulation is ready.
        This includes setting up joint limits and initializing state tensors.
        """
        self._isaaclab_default_state = RobotState(
            root_pos=self._robot.data.root_pos_w.clone(),
            root_rot=self._robot.data.root_quat_w.clone(),
            root_vel=torch.zeros(
                (len(self._robot.data.root_pos_w), 3), device=self.device
            ),
            root_ang_vel=torch.zeros(
                (len(self._robot.data.root_pos_w), 3), device=self.device
            ),
            dof_pos=self._robot.data.joint_pos.clone(),
            dof_vel=self._robot.data.joint_vel.clone(),
            rigid_body_pos=self._robot.data.body_pos_w.clone(),
            rigid_body_rot=self._robot.data.body_quat_w.clone(),
            rigid_body_vel=self._robot.data.body_lin_vel_w.clone(),
            rigid_body_ang_vel=self._robot.data.body_ang_vel_w.clone(),
        )

        dof_limits = self._robot.data.joint_limits.clone()
        self._dof_limits_lower_sim = dof_limits[0, :, 0].to(self.device)
        self._dof_limits_upper_sim = dof_limits[0, :, 1].to(self.device)

        super().on_environment_ready()

        # Update initial object positions
        if self.scene_lib is not None and self.scene_lib.total_spawned_scenes > 0:
            objects_start_pos = torch.zeros(
                (self.num_envs, 13), device=self.device, dtype=torch.float
            )
            for obj_idx, object in enumerate(self._object):
                objects_start_pos[:, :7] = self._initial_scene_pos[:, obj_idx, :]
                object.write_root_state_to_sim(objects_start_pos)

    def _activation_color(self, a: torch.Tensor):
        """
        Map muscle activation values to colors using a high-quality anatomical colormap.
        Gradient: Neutral Gray -> Pale Pink -> Vibrant Red -> Deep Crimson
        Inspired by medical anatomical visualization standards (e.g. Zygote Body, OpenAnatomy).
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, device=self.device, dtype=torch.float32)
        
        # Clamp activation to [0, 1]
        t = torch.clamp(a, 0.0, 1.0)
        
        # Define colormap control points (Stop positions, RGB values)
        # 0.0: Neutral Gray (Bone/Tendon)
        # 0.3: Pale Pinkish (Onset)
        # 0.7: Vibrant Muscle Red (Active)
        # 1.0: Deep Crimson (Max Contraction)
        stops = torch.tensor([0.0, 0.3, 0.7, 1.0], device=t.device)
        colors = torch.tensor([
            [0.75, 0.75, 0.75],  # Gray
            [0.85, 0.55, 0.55],  # Pale Pink
            [0.80, 0.10, 0.10],  # Vibrant Red
            [0.50, 0.00, 0.00]   # Deep Crimson
        ], device=t.device)
        
        # Perform piece-wise linear interpolation
        # Find indices of the segment each value falls into
        # Since we have 4 stops, we have 3 segments
        
        # This implementation is vectorized for speed
        out_colors = torch.zeros((t.shape[0], 3), device=t.device, dtype=torch.float32)
        
        # Segment 0: 0.0 -> 0.3
        mask0 = (t < stops[1])
        if mask0.any():
            t0 = (t[mask0] - stops[0]) / (stops[1] - stops[0])
            out_colors[mask0] = torch.lerp(colors[0], colors[1], t0.unsqueeze(1))
            
        # Segment 1: 0.3 -> 0.7
        mask1 = (t >= stops[1]) & (t < stops[2])
        if mask1.any():
            t1 = (t[mask1] - stops[1]) / (stops[2] - stops[1])
            out_colors[mask1] = torch.lerp(colors[1], colors[2], t1.unsqueeze(1))
            
        # Segment 2: 0.7 -> 1.0
        mask2 = (t >= stops[2])
        if mask2.any():
            t2 = (t[mask2] - stops[2]) / (stops[3] - stops[2])
            out_colors[mask2] = torch.lerp(colors[2], colors[3], t2.unsqueeze(1))
            
        return out_colors

    def _draw_muscles(self):
        self._update_muscle_mesh_visualization()

    def _update_muscle_mesh_visualization(self):
        if self.robot_config.asset.robot_type != 'act_humanoid':
            return
        if self._muscle_mesh_skin is None:
            body_names = self.robot_config.body_names
            body_pos0 = self._robot.data.body_pos_w[0][self.data_conversion.body_convert_to_common].clone()
            body_rot0 = self._robot.data.body_quat_w[0][self.data_conversion.body_convert_to_common].clone()
            mesh, skin = build_muscle_mesh(self.muscle_ctl, body_pos0, body_rot0, body_names)
            create_muscle_mesh_prim("/Visuals/MuscleMesh", mesh, visual_material=None)
            self._muscle_mesh_prim_path = "/Visuals/MuscleMesh"
            self._muscle_mesh_skin = skin
        body_pos = self._robot.data.body_pos_w[0][self.data_conversion.body_convert_to_common].clone()
        body_rot = self._robot.data.body_quat_w[0][self.data_conversion.body_convert_to_common].clone()
        
        muscle_colors = None
        if self.muscle_ctl is not None:
            if hasattr(self, "_last_activations") and self._last_activations is not None:
                a = self._last_activations
                if a.dim() > 1:
                    a = a[0]
            else:
                # Fallback to random if no activations available
                M = self.muscle_ctl.n_muscles
                #a = torch.rand(M, device=self.device)
                a = torch.zeros(M, device=self.device)
            
            # Calculate colors
            # Move to CPU because seg_map is numpy array
            a = a.detach().cpu()
            colors = self._activation_color(a)
            
            # Map to segments
            seg_map = self._muscle_mesh_skin["segment_to_muscle_map"]
            muscle_colors = colors[seg_map]

        update_muscle_mesh_points(self._muscle_mesh_prim_path, self._muscle_mesh_skin, body_pos, body_rot, muscle_colors=muscle_colors)

    def _get_body_jacobians_common(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        view = self._robot.root_physx_view
        jac = view.get_jacobians()
        if jac.shape[-1] != self._num_dof:
            if jac.shape[-1] > self._num_dof:
                jac = jac[..., -self._num_dof:]
            else:
                return None, None
        com = self._robot.data.body_com_pos_w.clone()
        body_idx = self.data_conversion.body_convert_to_common.to(jac.device)
        dof_idx = self.data_conversion.dof_convert_to_common.to(jac.device)
        jac = jac[:, body_idx, :, :]
        jac = jac[:, :, :, dof_idx]
        com = com[:, body_idx]
        return jac, com

    def update_muscle_features(self) -> Optional[Dict[str, torch.Tensor]]:
        bodies = self.get_bodies_state()
        if self.last_bodies is not None and self.last_bodies == bodies:
            return self.cache_feats
        self.last_bodies = bodies
        if self.muscle_ctl is None:
            return None
        jac, com = self._get_body_jacobians_common()
        JtA, b = self.muscle_ctl.update_muscle_features(
            bodies.rigid_body_pos,
            bodies.rigid_body_rot,
            com,
            jac,
        )
        feats = {"JtA": JtA, "b": b}
        self.cache_feats = feats
        return feats

    # =====================================================
    # Group 3: Simulation Steps & State Management
    # =====================================================
    def _physics_step(self) -> None:
        """
        Advance the simulation by stepping for a number of iterations equal to the decimation factor.
        Applies PD control or motor forces as required.
        """
        for idx in range(self.decimation):
            if self.control_type == ControlType.BUILT_IN_PD:
                self._apply_pd_control()
            else:
                self._apply_motor_forces()
            self._scene.write_data_to_sim()
            self._sim.step(render=False)
            if (idx + 1) % self.decimation == 0 and not self.headless:
                self._draw_muscles()
                # self._draw_foot_contacts()
                self._sim.render()
            self._scene.update(dt=self._sim.get_physics_dt())

    def _apply_pd_control(self) -> None:
        """
        Apply PD control by converting actions into PD targets and updating joint targets accordingly.
        """
        common_pd_tar = self._action_to_pd_targets(self._common_actions)
        isaaclab_pd_tar = common_pd_tar[:, self.data_conversion.dof_convert_to_sim]
        self._robot.set_joint_position_target(isaaclab_pd_tar, joint_ids=None)

    def _apply_motor_forces(self) -> None:
        """
        Apply motor forces to the robot.

        Raises:
            NotImplementedError: Not supported yet.
        """
        common_torques = self._compute_torques(self._common_actions)

        isaaclab_torques = common_torques[:, self.data_conversion.dof_convert_to_sim]
        self._robot.set_joint_effort_target(isaaclab_torques, joint_ids=None)

    def _set_simulator_env_state(
        self, new_states: RobotState, env_ids: Optional[torch.Tensor]
    ) -> None:
        """
        Apply the provided state to the simulation by writing root and joint states.

        Parameters:
            new_states (RobotState): The new simulation state.
            env_ids (Optional[torch.Tensor]): Specific environment IDs to update.
        """
        init_root_state = torch.cat(
            [
                new_states.root_pos,
                new_states.root_rot,
                new_states.root_vel,
                new_states.root_ang_vel,
            ],
            dim=-1,
        )
        self._robot.write_root_state_to_sim(init_root_state, env_ids)
        self._robot.write_joint_state_to_sim(
            new_states.dof_pos, new_states.dof_vel, None, env_ids
        )

    # =====================================================
    # Group 4: State Getters
    # =====================================================
    def _get_simulator_default_state(self) -> RobotState:
        """
        Retrieve the default simulation state based on the initialized values.

        Returns:
            RobotState: The default state containing positions, orientations, velocities, etc.
        """
        return self._isaaclab_default_state

    def _get_sim_body_ordering(self) -> SimBodyOrdering:
        """
        Obtain the ordering of body, degree-of-freedom, and contact sensor names.

        Returns:
            SimBodyOrdering: An object containing the body names, DOF names, and contact sensor body names.
        """
        return SimBodyOrdering(
            body_names=self._robot.data.body_names,
            dof_names=self._robot.data.joint_names,
            contact_sensor_body_names=self._contact_sensor.body_names,
        )

    def _get_simulator_bodies_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the state (positions, rotations, velocities) of all simulation bodies.

        Parameters:
            env_ids (Optional[torch.Tensor]): Restrict state retrieval to specific environments if provided.

        Returns:
            RobotState: The state of the bodies.
        """
        isaacsim_bodies_positions = self._robot.data.body_pos_w.clone()
        isaacsim_bodies_rotations = self._robot.data.body_quat_w.clone()
        isaacsim_bodies_velocities = self._robot.data.body_lin_vel_w.clone()
        isaacsim_bodies_ang_velocities = self._robot.data.body_ang_vel_w.clone()

        isaacsim_bodies_positions = isaacsim_bodies_positions.view(
            self.num_envs, self._num_bodies, 3
        )
        isaacsim_bodies_rotations = isaacsim_bodies_rotations.view(
            self.num_envs, self._num_bodies, 4
        )
        isaacsim_bodies_velocities = isaacsim_bodies_velocities.view(
            self.num_envs, self._num_bodies, 3
        )
        isaacsim_bodies_ang_velocities = isaacsim_bodies_ang_velocities.view(
            self.num_envs, self._num_bodies, 3
        )
        if env_ids is not None:
            isaacsim_bodies_positions = isaacsim_bodies_positions[env_ids]
            isaacsim_bodies_rotations = isaacsim_bodies_rotations[env_ids]
            isaacsim_bodies_velocities = isaacsim_bodies_velocities[env_ids]
            isaacsim_bodies_ang_velocities = isaacsim_bodies_ang_velocities[env_ids]
        return RobotState(
            rigid_body_pos=isaacsim_bodies_positions,
            rigid_body_rot=isaacsim_bodies_rotations,
            rigid_body_vel=isaacsim_bodies_velocities,
            rigid_body_ang_vel=isaacsim_bodies_ang_velocities,
        )

    def _get_simulator_dof_forces(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Retrieve applied torque forces for the robot's degrees of freedom.

        Parameters:
            env_ids (Optional[torch.Tensor]): Restrict query to specific environments if provided.

        Returns:
            torch.Tensor: The DOF forces.
        """
        isaacsim_dof_forces = self._robot.data.applied_torque.clone()
        if env_ids is not None:
            isaacsim_dof_forces = isaacsim_dof_forces[env_ids]
        return isaacsim_dof_forces

    def _get_simulator_dof_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the state (positions and velocities) of the robot's DOFs.

        Parameters:
            env_ids (Optional[torch.Tensor]): Restrict state retrieval to specific environments if provided.

        Returns:
            RobotState: The DOF state.
        """
        isaacsim_dof_pos = self._robot.data.joint_pos.clone()
        isaacsim_dof_vel = self._robot.data.joint_vel.clone()
        if env_ids is not None:
            isaacsim_dof_pos = isaacsim_dof_pos[env_ids]
            isaacsim_dof_vel = isaacsim_dof_vel[env_ids]
        return RobotState(
            dof_pos=isaacsim_dof_pos,
            dof_vel=isaacsim_dof_vel,
        )

    def _get_simulator_bodies_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Retrieve the contact force buffer for simulation bodies.

        Parameters:
            env_ids (Optional[torch.Tensor]): Specific environments to query.

        Returns:
            torch.Tensor: Tensor containing the contact forces.
        """
        if self._contact_sensor.data.force_matrix_w is not None:
            isaacsim_rb_contacts = (
                self._contact_sensor.data.force_matrix_w.clone().view(
                    self.num_envs, self._num_bodies, -1, 3
                )
            )
            isaacsim_rb_contacts = isaacsim_rb_contacts.sum(dim=2)
        else:
            isaacsim_rb_contacts = self._contact_sensor.data.net_forces_w.clone().view(
                self.num_envs, self._num_bodies, 3
            )
        if env_ids is not None:
            isaacsim_rb_contacts = isaacsim_rb_contacts[env_ids]
        return isaacsim_rb_contacts

    def _get_simulator_object_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Retrieve the contact buffer for simulation objects.

        Parameters:
            env_ids (Optional[torch.Tensor]): Specific environments to query.

        Returns:
            torch.Tensor: The object contact buffer.
        """
        if self.scene_lib is not None and self.scene_lib.total_spawned_scenes > 0:
            object_forces = []
            for obj_idx in range(self.scene_lib.num_objects_per_scene):
                object_forces.append(
                    self._object[obj_idx].data.net_contact_forces_w.clone()
                )
            if env_ids is not None:
                object_forces = object_forces[env_ids]
            return torch.stack(object_forces, dim=1)
        else:
            return_tensor = torch.zeros(
                self.num_envs, 1, 3, device=self.device, dtype=torch.float
            )
            if env_ids is not None:
                return_tensor = return_tensor[env_ids]
            return return_tensor

    def _get_simulator_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the root state (position, rotation, velocity) of the robot.

        Parameters:
            env_ids (Optional[torch.Tensor]): Specific environments to query.

        Returns:
            RobotState: The robot's root state.
        """
        isaacsim_root_pos = self._robot.data.root_pos_w.clone()
        isaacsim_root_rot = self._robot.data.root_quat_w.clone()
        isaacsim_root_vel = self._robot.data.root_lin_vel_w.clone()
        isaacsim_root_ang_vel = self._robot.data.root_ang_vel_w.clone()
        if env_ids is not None:
            isaacsim_root_pos = isaacsim_root_pos[env_ids]
            isaacsim_root_rot = isaacsim_root_rot[env_ids]
            isaacsim_root_vel = isaacsim_root_vel[env_ids]
            isaacsim_root_ang_vel = isaacsim_root_ang_vel[env_ids]
        return RobotState(
            root_pos=isaacsim_root_pos,
            root_rot=isaacsim_root_rot,
            root_vel=isaacsim_root_vel,
            root_ang_vel=isaacsim_root_ang_vel,
        )

    def _get_simulator_object_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the combined root state for all simulation objects.

        Parameters:
            env_ids (Optional[torch.Tensor]): Specific environments to query.

        Returns:
            RobotState: The objects' root state.
        """
        isaacsim_root_pos = []
        isaacsim_root_rot = []
        isaacsim_root_vel = []
        isaacsim_root_ang_vel = []
        for obj_idx in range(self.scene_lib.num_objects_per_scene):
            isaacsim_root_pos.append(self._object[obj_idx].data.root_pos_w.clone())
            isaacsim_root_rot.append(self._object[obj_idx].data.root_quat_w.clone())
            isaacsim_root_vel.append(self._object[obj_idx].data.root_lin_vel_w.clone())
            isaacsim_root_ang_vel.append(
                self._object[obj_idx].data.root_ang_vel_w.clone()
            )
        isaacsim_root_pos = torch.stack(isaacsim_root_pos, dim=1)
        isaacsim_root_rot = torch.stack(isaacsim_root_rot, dim=1)
        isaacsim_root_vel = torch.stack(isaacsim_root_vel, dim=1)
        isaacsim_root_ang_vel = torch.stack(isaacsim_root_ang_vel, dim=1)
        if env_ids is not None:
            isaacsim_root_pos = isaacsim_root_pos[env_ids]
            isaacsim_root_rot = isaacsim_root_rot[env_ids]
            isaacsim_root_vel = isaacsim_root_vel[env_ids]
            isaacsim_root_ang_vel = isaacsim_root_ang_vel[env_ids]
        return RobotState(
            root_pos=isaacsim_root_pos,
            root_rot=isaacsim_root_rot,
            root_vel=isaacsim_root_vel,
            root_ang_vel=isaacsim_root_ang_vel,
        )

    def get_num_actors_per_env(self) -> int:
        """
        Compute and return the number of actor instances per environment.

        Returns:
            int: Number of actors per environment.
        """
        root_pos = self._robot.data.root_pos_w
        return root_pos.shape[0] // self.num_envs

    # =====================================================
    # Group 5: Control & Computation Methods
    # =====================================================

    def _push_robot(self):
        vel_w = self._robot.data.root_vel_w
        self._robot.write_root_velocity_to_sim(
            vel_w + torch.ones_like(vel_w),
            env_ids=torch.arange(self.num_envs, device=self.device),
        )

    # =====================================================
    # Group 6: Rendering & Visualization
    # =====================================================
    def render(self) -> None:
        """
        Render the simulation view. Initializes or updates the camera if the simulator is not in headless mode.
        """
        if not self.headless:
            if not hasattr(self, "_perspective_view"):
                from protomotions.simulator.isaaclab.utils.perspective_viewer import (
                    PerspectiveViewer,
                )

                self._perspective_view = PerspectiveViewer()
                self._init_camera()
            else:
                self._update_camera()
        super().render()

    def _init_camera(self) -> None:
        """
        Initialize the camera view based on the current simulation root state.
        """
        self._cam_prev_char_pos = (
            self._get_simulator_root_state(0).root_pos.cpu().numpy()
        )
        pos = self._cam_prev_char_pos + np.array([0, 5, 1])
        self._perspective_view.set_camera_view(
            pos, self._cam_prev_char_pos + np.array([0, 0, 0.2])
        )

    def _update_camera(self) -> None:
        """
        Update the camera view based on the target's position and current camera movement.
        """
        if self._camera_target["element"] == 0:
            char_root_pos = (
                self._get_simulator_root_state(self._camera_target["env"])
                .root_pos.cpu()
                .numpy()
            )
            height_offset = 0.2
        else:
            in_scene_object_id = self._camera_target["element"] - 1
            char_root_pos = (
                self._get_simulator_object_root_state(self._camera_target["env"])
                .root_pos[in_scene_object_id]
                .cpu()
                .numpy()
            )
            height_offset = 0

        cam_pos = np.array(self._perspective_view.get_camera_state())
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = np.array(
            [char_root_pos[0], char_root_pos[1], char_root_pos[2] + height_offset]
        )
        new_cam_pos = np.array(
            [
                char_root_pos[0] + cam_delta[0],
                char_root_pos[1] + cam_delta[1],
                char_root_pos[2] + cam_delta[2],
            ]
        )
        self._perspective_view.set_camera_view(new_cam_pos, new_cam_target)
        self._cam_prev_char_pos[:] = char_root_pos

    def _write_viewport_to_file(self, file_name: str) -> None:
        """
        Capture the current viewport and save it to the specified file.

        Parameters:
            file_name (str): The filename for the saved image.
        """
        from omni.kit.viewport.utility import (
            get_active_viewport,
            capture_viewport_to_file,
        )

        vp_api = get_active_viewport()
        capture_viewport_to_file(vp_api, file_name)

    def close(self) -> None:
        """
        Close the simulation application and perform cleanup.
        """
        self._simulation_app.close()

    def _build_markers(
        self, visualization_markers: Dict[str, VisualizationMarker]
    ) -> None:
        """Build and configure visualization markers.

        Args:
            visualization_markers (Dict[str, VisualizationMarker]): Dictionary mapping marker names to their configurations
        """
        self._visualization_markers = {}
        if visualization_markers is None:
            return

        for marker_name, markers_cfg in visualization_markers.items():
            if markers_cfg.type == "sphere":
                marker_obj_cfg = VisualizationMarkersCfg(
                    prim_path=f"/Visuals/{marker_name}",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=1,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(
                                    markers_cfg.color[0],
                                    markers_cfg.color[1],
                                    markers_cfg.color[2],
                                )
                            ),
                        ),
                    },
                )
            elif markers_cfg.type == "arrow":
                marker_obj_cfg = VisualizationMarkersCfg(
                    prim_path=f"/Visuals/{marker_name}",
                    markers={
                        "arrow_x": sim_utils.UsdFileCfg(
                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                            scale=(1.0, 1.0, 1.0),
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(
                                    markers_cfg.color[0],
                                    markers_cfg.color[1],
                                    markers_cfg.color[2],
                                ),
                                opacity=0.5,
                            ),
                        ),
                    },
                )
            else:
                raise ValueError(f"Marker type {markers_cfg.type} not supported")

            marker_scale = []
            for i, marker in enumerate(markers_cfg.markers):
                if markers_cfg.type == "sphere":
                    if marker.size == "tiny":
                        scale = 0.007
                    elif marker.size == "small":
                        scale = 0.01
                    else:
                        scale = 0.05
                    marker_scale.append([scale, scale, scale])
                elif markers_cfg.type == "arrow":
                    if marker.size == "small":
                        scale = 0.1
                    else:
                        scale = 0.5
                    marker_scale.append([scale, 0.2 * scale, 0.2 * scale])

            self._visualization_markers[marker_name] = EasyDict(
                {
                    "marker": VisualizationMarkers(marker_obj_cfg),
                    "scale": torch.tensor(marker_scale, device=self.device).repeat(
                        self.num_envs, 1
                    ),
                }
            )

    def _update_simulator_markers(
        self, markers_state: Optional[Dict[str, MarkerState]] = None
    ) -> None:
        """Update the visualization markers with new state information.

        Args:
            markers_state (Dict[str, MarkerState]): Dictionary mapping marker names to their state (translation and orientation)
        """
        if markers_state is None:
            return

        for marker_name, markers_state_item in markers_state.items():
            if marker_name not in self._visualization_markers:
                continue
            marker_dict = self._visualization_markers[marker_name]
            marker_dict.marker.visualize(
                translations=markers_state_item.translation.view(-1, 3),
                orientations=markers_state_item.orientation.view(-1, 4),
                scales=marker_dict.scale,
            )

    def _update_activation(self, teacher_actions: torch.Tensor):
        feats = self.update_muscle_features()
        JtA, b = feats["JtA"], feats["b"]

        common_dof_state = self.get_dof_state()
        q = common_dof_state.dof_pos
        qd = common_dof_state.dof_vel

        J = JtA.transpose(1, 2)
        tau_teacher = self._compute_tau_from_actions(teacher_actions, q, qd)
        if hasattr(self, "_last_activations"):
            last_act = self._last_activations
        else:
            last_act = None
        a, pd_tau_des = optimize_act(JtA, b, tau_teacher, last_act=last_act, method="lbfgs")
      
        print("diff", torch.norm(pd_tau_des - tau_teacher,dim=-1))
        # s = torch.linalg.svdvals(J)
        # print("min sv", s.min(dim=1).values, "max sv", s.max(dim=1).values)
        # tau_cap = torch.sum(torch.clamp(JtA, min=0.0), dim=1) + b
        # print("cap", tau_cap.abs().max(dim=1).values, "target", tau_teacher.abs().max(dim=1).values)
        # num = (pd_tau_des * tau_teacher).sum(dim=1)
        # den = (pd_tau_des.norm(dim=1)*tau_teacher.norm(dim=1)+1e-6)
        # print("cos", num/den)
        return a
