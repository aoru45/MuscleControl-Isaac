import torch
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict
from isaac_utils.rotations import quaternion_to_matrix

class Muscle:
    def __init__(self, name, waypoints, attrs=None):
        self.name = name
        self.waypoints = waypoints
        self.attrs = attrs or {}

class MuscleControllerIsaacLab:
    def __init__(
        self, 
        muscle_xml_path: str, 
        device: torch.device,
        rig_path: str,
        *args,
        **kwargs
    ):
        self.device = device
        self._parse_muscles(muscle_xml_path, rig_path)
        
        # Hill-type constants
        self.gamma = torch.tensor(0.5, dtype=torch.float, device=device)
        self.k_pe = torch.tensor(4.0, dtype=torch.float, device=device)
        self.e_mo = torch.tensor(0.6, dtype=torch.float, device=device)
        self.f_toe = torch.tensor(0.33, dtype=torch.float, device=device)
        self.k_toe = torch.tensor(3.0, dtype=torch.float, device=device)
        self.k_lin = torch.tensor(51.878788, dtype=torch.float, device=device)
        self.e_toe = torch.tensor(0.02, dtype=torch.float, device=device)
        self.e_t0 = torch.tensor(0.033, dtype=torch.float, device=device)

        self._backend_idx_per_muscle = None
        self._local_points_per_muscle = None
        self._backend_name_to_idx = None
        self._mj_bodyid_to_backend_idx = None
        self._padded_backend_idx = None
        self._padded_local_pts = None
        self._maxW = 0
        self._l_mt0 = None

        # Cache placeholders
        self._cached_J_p = None             # Point Jacobian (B, M, W, 3, D_sim)
        self._cached_u_vecs = None          # Unit vectors of segments (B, M, W-1, 3)
        self._cached_force_active_scale = None # Hill active term scalar
        self._cached_force_passive = None      # Hill passive term scalar
        self._cached_mask_wp = None            # Mask for valid waypoints
        self._cached_l_mt = None            # Muscle lengths

        self._batch_padded_idx = None
        self._batch_padded_pts = None

        self._dof_prepared = False

    def _parse_muscles(self, path, rig_path):
        import mujoco
        if 1:
            self.m = mujoco.MjModel.from_xml_path(rig_path)
            self.d = mujoco.MjData(self.m)
            mujoco.mj_forward(self.m, self.d)
            self.body_name_to_id = {}
            for i in range(self.m.nbody):
                name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i)
                if name:
                    self.body_name_to_id[name] = i
            self.joint_name_to_id = {}
            for j in range(self.m.njnt):
                name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)
                if name:
                    self.joint_name_to_id[name] = j
            self.body_id_to_name = {v: k for k, v in self.body_name_to_id.items()}


            self.joint_pos = np.array(self.m.jnt_pos, dtype=np.float64)
            self.joint_axis = np.array(self.m.jnt_axis, dtype=np.float64)
            self.body_parent = np.array(self.m.body_parentid, dtype=np.int32)

        tree = ET.parse(path)
        root = tree.getroot()
        muscles = []
        names = []
        f0 = []
        lm0 = []
        lt0 = []
        lmax = []
        for unit in root.findall('Unit'):
            name = unit.attrib.get('name', '')
            names.append(name)
            f0_val = float(unit.attrib.get('f0', '1000'))
            lm0_val = float(unit.attrib.get('lm', '1.0'))
            lt0_val = float(unit.attrib.get('lt', '0.2'))
            lmax_val = float(unit.attrib.get('lmax', '-0.1'))
            
            # Heuristic for invalid lmax
            if lmax_val <= 0:
                lmax_val = lt0_val + 1.6 * lm0_val

            f0.append(f0_val)
            lm0.append(lm0_val)
            lt0.append(lt0_val)
            lmax.append(lmax_val)
            
            wps = []
            attrs = {
                'f0': f0_val,
                'lm': lm0_val,
                'lt': lt0_val,
                'lmax': lmax_val,
            }
            for wp in unit.findall('Waypoint'):
                body = wp.attrib['body']
                p = [float(x) for x in wp.attrib['p'].strip().split()]
                pw = self._rx90(p)
                bid = self.body_name_to_id.get(body, -1)
                if bid >= 0:
                    x = np.array(self.d.xpos[bid], dtype=np.float64)
                    R = np.array(self.d.xmat[bid], dtype=np.float64).reshape(3, 3)
                    pl = R.T @ (pw - x)
                else:
                    pl = pw
                wps.append((body, pl))
            if len(wps) >= 2:
                muscles.append(Muscle(name, wps, attrs))
        self.muscle_names = names
        self.f0 = torch.tensor(f0, device=self.device, dtype=torch.float)
        self.l_m0 = torch.tensor(lm0, device=self.device, dtype=torch.float)
        self.l_t0 = torch.tensor(lt0, device=self.device, dtype=torch.float)
        self.lmax = torch.tensor(lmax, device=self.device, dtype=torch.float)
        self.muscles = muscles

    def prepare_mapping(self, backend_body_names: list):
        if self._backend_idx_per_muscle is not None:
            return
        name_to_idx = {n: i for i, n in enumerate(backend_body_names)}
        self._backend_name_to_idx = name_to_idx
        nbody = int(self.m.nbody)
        mj_to_backend = np.full((nbody,), -1, dtype=np.int64)
        for bid in range(nbody):
            nm = self.body_id_to_name.get(bid, None)
            if nm is not None and nm in name_to_idx:
                mj_to_backend[bid] = name_to_idx[nm]
        self._mj_bodyid_to_backend_idx = mj_to_backend
        idxs = []
        locals_pts = []
        for mus in self.muscles:
            mus_indices = []
            mus_points = []
            for body, p_local in mus.waypoints:
                if body in name_to_idx:
                    mus_indices.append(name_to_idx[body])
                    mus_points.append(p_local)
            idxs.append(np.array(mus_indices, dtype=np.int64))
            if len(mus_points) > 0:
                locals_pts.append(np.stack(mus_points, axis=0))
            else:
                locals_pts.append(np.zeros((0, 3), dtype=np.float64))
        self._backend_idx_per_muscle = idxs
        self._local_points_per_muscle = locals_pts
        M = len(idxs)
        W_list = [len(w) for w in idxs]
        maxW = max(W_list) if M > 0 else 0
        padded_idxs = np.full((M, maxW), -1, dtype=np.int64)
        padded_local = np.zeros((M, maxW, 3), dtype=np.float64)
        for m, W in enumerate(W_list):
            if W > 0:
                padded_idxs[m, :W] = idxs[m]
                padded_local[m, :W] = locals_pts[m]
                wp_mj = []
                for body, _ in self.muscles[m].waypoints:
                    wp_mj.append(self.body_name_to_id.get(body, -1))
        self._padded_backend_idx = padded_idxs
        self._padded_local_pts = padded_local
        self._maxW = maxW
        l0 = np.zeros((M,), dtype=np.float64)
        for m in range(M):
            W = len(self.muscles[m].waypoints)
            if W < 2:
                l0[m] = 1.0 # Avoid division by zero
                continue
            total = 0.0
            for w in range(W - 1):
                b0, p0_local = self.muscles[m].waypoints[w]
                b1, p1_local = self.muscles[m].waypoints[w + 1]
                bid0 = self.body_name_to_id.get(b0, -1)
                bid1 = self.body_name_to_id.get(b1, -1)
                if bid0 >= 0:
                    x0 = np.array(self.d.xpos[bid0], dtype=np.float64)
                    R0 = np.array(self.d.xmat[bid0], dtype=np.float64).reshape(3, 3)
                    p0 = x0 + R0 @ p0_local
                else:
                    p0 = np.array(p0_local, dtype=np.float64)
                if bid1 >= 0:
                    x1 = np.array(self.d.xpos[bid1], dtype=np.float64)
                    R1 = np.array(self.d.xmat[bid1], dtype=np.float64).reshape(3, 3)
                    p1 = x1 + R1 @ p1_local
                else:
                    p1 = np.array(p1_local, dtype=np.float64)
                total += np.linalg.norm(p1 - p0)
            l0[m] = total
        self._l_mt0 = torch.tensor(l0, dtype=torch.float32, device=self.device)

    def prepare_dof_geometry(self, dof_names: list):
        if self._dof_prepared:
            return
        dof_bodyid = []
        dof_jpos = []
        dof_jaxis = []
        for nm in dof_names:
            jid = self.joint_name_to_id.get(nm, None)
            if jid is None:
                dof_bodyid.append(0)
                dof_jpos.append(np.zeros(3, dtype=np.float64))
                dof_jaxis.append(np.array([1.0, 0.0, 0.0], dtype=np.float64))
            else:
                dof_bodyid.append(int(self.m.jnt_bodyid[jid]))
                jpos = self.joint_pos[jid] if self.joint_pos is not None else np.zeros(3, dtype=np.float64)
                jaxis = self.joint_axis[jid] if self.joint_axis is not None else np.array([1.0, 0.0, 0.0], dtype=np.float64)
                dof_jpos.append(jpos)
                dof_jaxis.append(jaxis)
        self._dof_bodyid = np.array(dof_bodyid, dtype=np.int64)
        self._dof_joint_pos_local = np.stack(dof_jpos, axis=0)
        self._dof_joint_axis_local = np.stack(dof_jaxis, axis=0)
        nbody = int(self.m.nbody)
        anc = np.zeros((nbody, nbody), dtype=bool)
        for c in range(nbody):
            b = c
            visited = set()
            while True:
                if b in visited:
                    break
                visited.add(b)
                anc[b, c] = True
                nb = int(self.body_parent[b])
                if nb < 0 or nb == b:
                    break
                b = nb
        self._ancestor_matrix = anc
        self._dof_prepared = True
    
    def g_al(self, x: torch.Tensor):
        return torch.exp(-((x - 1.0) * (x - 1.0)) / self.gamma)

    def g_pl(self, x: torch.Tensor):
        f_pl = (torch.exp(self.k_pe * (x - 1.0) / self.e_mo) - 1.0) / (torch.exp(self.k_pe) - 1.0)
        return torch.where(x < 1.0, torch.zeros_like(f_pl), f_pl)

    def g_t(self, e_t: torch.Tensor):
        toe = self.f_toe / (torch.exp(self.k_toe) - 1.0) * (torch.exp(self.k_toe * e_t / self.e_toe) - 1.0)
        lin = self.k_lin * (e_t - self.e_toe) + self.f_toe
        return torch.where(e_t <= self.e_t0, toe, lin)

    def forward_batched_points(self, body_pos: torch.Tensor, body_rot: torch.Tensor, backend_body_names: list):
        names = backend_body_names
        name_to_idx = {n: i for i, n in enumerate(names)}
        M = len(self.muscles)
        W_list = [len(m.waypoints) for m in self.muscles]
        maxW = max(W_list) if M > 0 else 0
        if self._batch_padded_idx is None:
            self._batch_padded_idx = torch.full((M, maxW), -1, dtype=torch.long)
            self._batch_padded_pts = torch.zeros((M, maxW, 3), dtype=body_pos.dtype)
            for m_idx, mus in enumerate(self.muscles):
                for w_idx, (body, p_local) in enumerate(mus.waypoints):
                    self._batch_padded_idx[m_idx, w_idx] = name_to_idx.get(body, -1)
                    self._batch_padded_pts[m_idx, w_idx] = torch.tensor(p_local, dtype=body_pos.dtype)
            self._batch_padded_idx = self._batch_padded_idx.to(body_pos.device)
            self._batch_padded_pts = self._batch_padded_pts.to(body_pos.device)
        
        rotmats = quaternion_to_matrix(body_rot, w_last=True)
        safe_idx = self._batch_padded_idx.clamp(min=0)
        flat_idx = safe_idx.view(-1)
        pos_sel = body_pos[flat_idx].view(1, M, maxW, 3)
        rot_sel = rotmats[flat_idx].view(1, M, maxW, 3, 3)
        world_pts = pos_sel + torch.matmul(rot_sel, self._batch_padded_pts.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        return world_pts, W_list
   
    # ----------------------------------------------------------------------
    # Core Logic
    # ----------------------------------------------------------------------

    def update_muscle_state(self, body_pos: torch.Tensor, body_rot: torch.Tensor, jacobians: torch.Tensor):
        """
        Updates cached physical quantities (Jacobians, Directions, Hill terms).
        
        Args:
            body_pos: (B, num_bodies_sim, 3) World positions
            body_rot: (B, num_bodies_sim, 4) World quaternions (x, y, z, w)
            jacobians: (B, num_bodies_sim, 6, num_dof) Jacobians
        """
        B = body_pos.shape[0]
        M, maxW = self._padded_backend_idx.shape
        device = self.device
        dtype = body_pos.dtype

        # 1. Kinematics: Calculate World Positions of Waypoints
        R = quaternion_to_matrix(body_rot, w_last=True)
        # -----------------------------------------------------
        # Gather body transforms for each waypoint
        padded_backend_idx = torch.as_tensor(self._padded_backend_idx, dtype=torch.long, device=device)
        flat_idx = padded_backend_idx.view(-1)
        # Handle static/world bodies (-1) by clamping to 0 and masking later
        safe_idx = flat_idx.clamp(min=0)
        
        # (B, M*maxW, 3)
        pos_body_wp = body_pos[:, safe_idx, :]
        # (B, M*maxW, 3, 3)
        rot_body_wp = R[:, safe_idx, :, :]
        
        local_pts = torch.as_tensor(self._padded_local_pts, dtype=dtype, device=device).reshape(1,-1,3)
        
        # r_wp: vector from body origin to waypoint in world frame
        r_wp = torch.matmul(rot_body_wp, local_pts.reshape(1, -1, 3, 1)).squeeze(-1)
        
        # p_world: absolute world position of waypoints
        p_world_flat = pos_body_wp + r_wp

        is_static = (flat_idx == -1).unsqueeze(0).unsqueeze(-1) # (1, Total, 1)
        p_world_flat = torch.where(is_static, local_pts.expand_as(p_world_flat), p_world_flat)
        
        p_world = p_world_flat.view(B, M, maxW, 3)

        # 2. Geometry: Segment Directions and Lengths
        # -------------------------------------------
        seg = p_world[:, :, 1:, :] - p_world[:, :, :-1, :] # (B, M, maxW-1, 3)
        seg_len = torch.linalg.norm(seg, dim=-1, keepdim=True) + 1e-8
        u_vecs = seg / seg_len # Unit vectors (B, M, maxW-1, 3)

        W_list = torch.tensor([len(m.waypoints) for m in self.muscles], dtype=torch.long, device=device).view(1, M, 1, 1)
        idx_grid = torch.arange(maxW - 1, device=device).view(1, 1, -1, 1)
        seg_mask = (idx_grid < (W_list - 1)).float() # (1, M, maxW-1, 1)

        l_mt = (seg_len * seg_mask).sum(dim=2).squeeze(-1) # (B, M)
        # MASS uses normalized lengths for muscle dynamics.
        l_mt0 = self._l_mt0.view(1, -1).to(dtype=dtype, device=device)
        l_mt_norm = l_mt / (l_mt0 + 1e-8)


        # 3. Jacobian: Point Jacobian (d_pos / d_theta)
        # ---------------------------------------------
        jac_joints = jacobians
        
        # J_wp_body: (B, M*maxW, 6, N_dof_sim)
        J_wp_body = jac_joints[:, safe_idx, :, :]
        
        # If static (-1), Jacobian is zero
        J_wp_body = torch.where(is_static.unsqueeze(-1), torch.zeros_like(J_wp_body), J_wp_body)
        
        J_v = J_wp_body[:, :, 0:3, :]
        J_w = J_wp_body[:, :, 3:6, :]
        
        # Point Jacobian: J_p = J_v - r x J_w
        # r_wp is (B, Total, 3)
        r_exp = r_wp.unsqueeze(-1)
        rxJw = torch.linalg.cross(r_exp.expand_as(J_w), J_w, dim=-2)
        J_p_flat = J_v - rxJw
        J_p = J_p_flat.view(B, M, maxW, 3, -1) # (B, M, W, 3, D_sim)

        # 4. Hill-Type Terms Pre-calculation
        # ----------------------------------
        l_m = l_mt_norm - self.l_t0.view(1, -1)
        x = l_m / self.l_m0.view(1, -1)
        e_t = (l_mt_norm - l_m - self.l_t0.view(1, -1)) / self.l_t0.view(1, -1)
        
        f_active_scale = self.f0.view(1, -1) * self.g_al(x)
        f_passive = self.f0.view(1, -1) * (self.g_pl(x) )

        # 5. Caching
        # ----------
        self._cached_J_p = J_p
        self._cached_u_vecs = u_vecs
        self._cached_seg_mask = seg_mask
        self._cached_force_active_scale = f_active_scale
        self._cached_force_passive = f_passive
        self._cached_l_mt = l_mt_norm

    def _rx90(self, p):
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        return np.array([x, -z, y], dtype=np.float64)

    def compute_torque(self, activations: torch.Tensor) -> torch.Tensor:
        if self._cached_J_p is None:
            raise RuntimeError("Muscle state not updated. Call update_muscle_state() first.")
        B, M = activations.shape
        W = self._cached_J_p.shape[2]
        tension = self._cached_force_active_scale * activations + self._cached_force_passive # (B, M)
        T_vecs = tension.view(B, M, 1, 1) * self._cached_u_vecs
        T_vecs = T_vecs * self._cached_seg_mask
        F_wp = torch.zeros((B, M, W, 3), device=self.device, dtype=activations.dtype)
        F_wp[:, :, :-1, :] += T_vecs
        F_wp[:, :, 1:, :] -= T_vecs
        tau_com = torch.einsum('bmwxd,bmwx->bd', self._cached_J_p, F_wp)
        return tau_com

    def compute_activation_pd(self, action: torch.Tensor, dof_vel: torch.Tensor) -> torch.Tensor:
        """
        Computes muscle activations using a PD control law on muscle length, matching 'Kinesis' implementation.
        
        Args:
            action (torch.Tensor): Controls [-1, 1], mapped to target length [0, lmax].
            dof_vel (torch.Tensor): DOF velocities (B, D).
            
        Returns:
            torch.Tensor: Muscle activations [0, 1] (B, M).
        """
        if self._cached_J_p is None:
            raise RuntimeError("Muscle state not updated. Call update_muscle_state() first.")
            
        B, M = action.shape
        
        # 1. Map action to target length
        # Kinesis: range [0, hi] where hi is max length
        lo = 0.0
        hi = self.lmax.view(1, M)
        target_len = (action + 1.0) / 2.0 * (hi - lo) + lo
        
        # 2. Compute muscle velocity
        # v_wp = J_p * dof_vel
        # J_p: (B, M, W, 3, D), dof_vel: (B, D)
        # v_wp: (B, M, W, 3)
        v_wp = torch.einsum('bmwjd,bd->bmwj', self._cached_J_p, dof_vel)
        
        # v_seg_rel = v_{i+1} - v_i
        v_seg = v_wp[:, :, 1:] - v_wp[:, :, :-1] # (B, M, W-1, 3)
        
        # v_muscle = sum(v_seg . u_vec) for all segments
        # dot product along dim 3 (xyz)
        v_shortening = (v_seg * self._cached_u_vecs).sum(dim=-1) # (B, M, W-1)
        muscle_vel = (v_shortening * self._cached_seg_mask.squeeze(-1)).sum(dim=-1) # (B, M)
        
        # 3. PD Control for Desired Force
        # Kinesis: F = Kp(l_target - l_current) - Kd * v_current
        # Kinesis uses conventions where Force < 0 is pulling (shortening).
        # Our `compute_torque` assumes Tension > 0 is pulling.
        # So Tension_des = -F_kinesis
        # Tension_des = Kp(l_current - l_target) + Kd * v_current
        
        # Constants from Kinesis / MuJoCo defaults
        # Kp = scale * F0. Kinesis uses 5 * F0?
        # In `target_length_to_force` (Kinesis): kp = 5 * peak_force
        kp_scale = 5.0
        kp = kp_scale * self.f0.view(1, M)
        kd = 0.1 * kp
        
        l_current = self._cached_l_mt
        
        # If v_current > 0 (lengthening), we want to resist -> Pull harder -> Tension increases.
        # + Kd * v works: v>0 -> adds tension.
        
        # If l_current > l_target (too long), we want to shorten -> Pull harder.
        # + Kp * (l_current - l_target) works.
        
        tension_des = kp * (l_current - target_len) + kd * muscle_vel
        
        # Clip tension to [0, F0] (Kinesis clips force to [-F0, 0])
        tension_des = torch.clamp(tension_des, min=0.0)
        max_f = self.f0.view(1, M)
        tension_des = torch.min(tension_des, max_f)
        
        # 4. Invert Hill Model
        # Tension = F_act * a + F_pass
        # a = (Tension - F_pass) / F_act
        
        f_act = self._cached_force_active_scale
        f_pass = self._cached_force_passive
        
        # Avoid division by zero
        f_act = f_act + 1e-6
        
        activations = (tension_des - f_pass) / f_act
        activations = torch.clamp(activations, 0.0, 1.0)
        
        return activations
