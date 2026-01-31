import xml.etree.ElementTree as ET
from typing import Dict, List
import torch
import numpy as np
from isaac_utils.rotations import quaternion_to_matrix


class Muscle:
    def __init__(self, name, waypoints, attrs=None):
        self.name = name
        self.waypoints = waypoints
        self.attrs = attrs or {}

class Waypoint:
    def __init__(self, bodies, local_positions, weights):
        self.bodies = bodies
        self.local_positions = local_positions
        self.weights = weights

class MuscleController:
    def __init__(self, muscle_xml_path: str, device: torch.device, mujoco=None, rig_path: str = None):
        self.device = device
        tree = ET.parse(muscle_xml_path)
        root = tree.getroot()
        names = []
        f0 = []
        lm0 = []
        lt0 = []
        lmax = []
        for unit in root.findall('Unit'):
            names.append(unit.attrib['name'])
            f0.append(float(unit.attrib.get('f0', '1000')))
            lm0.append(float(unit.attrib.get('lm', '1.0')))
            lt0.append(float(unit.attrib.get('lt', '0.2')))
            lmax.append(float(unit.attrib.get('lmax', '-0.1')))
        self.muscle_names = names
        self.f0 = torch.tensor(f0, device=device, dtype=torch.float)
        self._f0_base = self.f0.clone()
        self._f0_scale = torch.ones_like(self.f0)
        self._global_scale = 1.0
        self.f0 = self._f0_base * self._f0_scale * self._global_scale
        self._muscle_name_to_idx = {name: i for i, name in enumerate(self.muscle_names)}
        self.l_m0 = torch.tensor(lm0, device=device, dtype=torch.float)
        self.l_t0 = torch.tensor(lt0, device=device, dtype=torch.float)
        self.lmax = torch.tensor(lmax, device=device, dtype=torch.float)
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
        self._padded_wp_bodies_mj = None
        self._padded_weights = None
        self._maxW = 0
        self._maxK = 0
        self._l_mt0 = None
        self._dof_prepared = False
        self._dof_bodyid = None
        self._dof_joint_pos_local = None
        self._dof_joint_axis_local = None
        self._ancestor_matrix = None
        self.m = None
        self.d = None
        self.body_name_to_id = None
        self.body_id_to_name = None
        self.joint_name_to_id = None
        if (mujoco is not None) and (rig_path is not None):
            self.m = mujoco.MjModel.from_xml_path(rig_path)
            self.d = mujoco.MjData(self.m)
            mujoco.mj_forward(self.m, self.d)
            self.body_name_to_id = {}
            for i in range(self.m.nbody):
                name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i)
                if name:
                    self.body_name_to_id[name] = i
            self.body_id_to_name = {v: k for k, v in self.body_name_to_id.items()}
            self.joint_name_to_id = {}
            for j in range(self.m.njnt):
                name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)
                if name:
                    self.joint_name_to_id[name] = j
            self.joint_pos = np.array(self.m.jnt_pos, dtype=np.float64)
            self.joint_axis = np.array(self.m.jnt_axis, dtype=np.float64)
            self.body_parent = np.array(self.m.body_parentid, dtype=np.int32)
            self.body_jntadr = np.array(self.m.body_jntadr, dtype=np.int32)
            self.body_jntnum = np.array(self.m.body_jntnum, dtype=np.int32)
        self.muscles = self._parse_muscles(muscle_xml_path, mujoco)

   
    def g_al(self, x: torch.Tensor):
        return torch.exp(-((x - 1.0) * (x - 1.0)) / self.gamma)

    def g_pl(self, x: torch.Tensor):
        f_pl = (torch.exp(self.k_pe * (x - 1.0) / self.e_mo) - 1.0) / (torch.exp(self.k_pe) - 1.0)
        return torch.where(x < 1.0, torch.zeros_like(f_pl), f_pl)

    def g_t(self, e_t: torch.Tensor):
        toe = self.f_toe / (torch.exp(self.k_toe) - 1.0) * (torch.exp(self.k_toe * e_t / self.e_toe) - 1.0)
        lin = self.k_lin * (e_t - self.e_toe) + self.f_toe
        return torch.where(e_t <= self.e_t0, toe, lin)

    def g(self, l_m: torch.Tensor, l_mt: torch.Tensor, activation: torch.Tensor):
        x = l_m / self.l_m0.view(1, -1)
        e_t = (l_mt - l_m - self.l_t0.view(1, -1)) / self.l_t0.view(1, -1)
        return self.g_t(e_t) - (self.g_pl(x) + activation * self.g_al(x))

    def update_muscle_features(
        self,
        body_pos: torch.Tensor,
        body_rot: torch.Tensor,
        body_com_pos: torch.Tensor,
        jacobians: torch.Tensor,
        dof_names: list,
        backend_body_names: list = None,
    ) -> (torch.Tensor, torch.Tensor):
        self.prepare_dof_geometry(dof_names)
        if self._backend_idx_per_muscle is None:
            if backend_body_names is None:
                backend_body_names = [str(i) for i in range(body_pos.shape[1])]
            self.prepare_mapping(backend_body_names)
       
        pos = body_pos
        rot = body_rot
        com = body_com_pos
        jac = jacobians

        B, N, _ = pos.shape
        device = pos.device
        dtype = pos.dtype
        R = quaternion_to_matrix(rot, w_last=True)

        padded_backend_idx = torch.as_tensor(self._padded_backend_idx, dtype=torch.long, device=device)
        local_pts = torch.as_tensor(self._padded_local_pts, dtype=dtype, device=device)
        weights = torch.as_tensor(self._padded_weights, dtype=dtype, device=device)
        padded_wp_mj = torch.as_tensor(self._padded_wp_bodies_mj, dtype=torch.long, device=device)
        M, maxW, _ = padded_backend_idx.shape

        safe_idx = padded_backend_idx.clamp(min=0)
        x_wp = pos[:, safe_idx, :]
        R_wp = R[:, safe_idx, :, :]
        p_world_infl = x_wp + torch.matmul(R_wp, local_pts.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        p_world = (p_world_infl * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=3)
        wp_mask = (weights.sum(dim=2) > 0)

        seg = p_world[:, :, 1:, :] - p_world[:, :, :-1, :]
        seg_len = torch.linalg.norm(seg, dim=-1, keepdim=True) + 1e-8
        seg_mask = (wp_mask[:, 1:] & wp_mask[:, :-1]).to(dtype)
        u = seg / seg_len
        u = u * seg_mask.unsqueeze(0).unsqueeze(-1)
        force_dir = torch.zeros((B, M, maxW, 3), dtype=dtype, device=device)
        force_dir[:, :, :-1, :] += u
        force_dir[:, :, 1:, :] += -u
        force_dir = force_dir * wp_mask.unsqueeze(0).unsqueeze(-1)

        l_mt0 = torch.as_tensor(self._l_mt0, dtype=dtype, device=device).clamp(min=1e-8)
        l_mt = (seg_len.squeeze(-1) * seg_mask.unsqueeze(0)).sum(dim=-1) / l_mt0.view(1, M)
        l_m = l_mt - self.l_t0.view(1, -1)
        x = l_m / self.l_m0.view(1, -1)
        f_active = self.f0.view(1, -1) * self.g_al(x)
        f_passive = self.f0.view(1, -1) * self.g_pl(x)

        primary_idx = padded_backend_idx[:, :, 0].clamp(min=0)
        jac_primary = jac[:, primary_idx, :, :]
        body_com_primary = com[:, primary_idx, :]
        r_wp = p_world - body_com_primary
        J_v = jac_primary[..., 0:3, :]
        J_w = jac_primary[..., 3:6, :]
        r_exp = r_wp.unsqueeze(-1)
        rxJw = torch.linalg.cross(r_exp.expand_as(J_w), J_w, dim=-2)
        J_p = J_v - rxJw

        contrib = (J_p * force_dir.unsqueeze(-1)).sum(dim=-2)
        Gsum = contrib.sum(dim=2)

        JtA = f_active.unsqueeze(-1) * Gsum
        b = (f_passive.unsqueeze(-1) * Gsum).sum(dim=1)
        return JtA, b

    def _rx90(self, p):
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        return np.array([x, -z, y], dtype=np.float64)

    def _parse_muscles(self, path, mujoco):
        tree = ET.parse(path)
        root = tree.getroot()
        muscles = []
        for unit in root.findall('Unit'):
            name = unit.attrib.get('name', '')
            wps = []
            attrs = {
                'f0': float(unit.attrib.get('f0', '1000')),
                'lm': float(unit.attrib.get('lm', '1.0')),
                'lt': float(unit.attrib.get('lt', '0.2')),
                'lmax': float(unit.attrib.get('lmax', '-0.1')),
            }
            waypoints = list(unit.findall('Waypoint'))
            num_waypoints = len(waypoints)
            for w_idx, wp in enumerate(waypoints):
                body = wp.attrib['body']
                p = [float(x) for x in wp.attrib['p'].strip().split()]
                pw = self._rx90(p)
                if (self.m is None) or (self.d is None):
                    wps.append(Waypoint([body], [pw], [1.0]))
                    continue
                if w_idx == 0 or w_idx == num_waypoints - 1:
                    wps.append(self._make_single_waypoint(body, pw))
                else:
                    wps.append(self._make_lbs_waypoint(body, pw))
            if len(wps) >= 2:
                muscles.append(Muscle(name, wps, attrs))
        return muscles

    def _get_body_pose(self, bid: int):
        x = np.array(self.d.xpos[bid], dtype=np.float64)
        R = np.array(self.d.xmat[bid], dtype=np.float64).reshape(3, 3)
        return x, R

    def _make_single_waypoint(self, body: str, pw: np.ndarray) -> Waypoint:
        bid = self.body_name_to_id.get(body, -1)
        if bid >= 0:
            x, R = self._get_body_pose(bid)
            pl = R.T @ (pw - x)
        else:
            pl = pw
        return Waypoint([body], [pl], [1.0])

    def _make_lbs_waypoint(self, body: str, pw: np.ndarray) -> Waypoint:
        nbody = int(self.m.nbody)
        distances = np.zeros((nbody,), dtype=np.float64)
        local_positions = [None] * nbody
        distances[0] = np.inf
        local_positions[0] = np.zeros(3, dtype=np.float64)
        for bid in range(1, nbody):
            x, R = self._get_body_pose(bid)
            local_positions[bid] = R.T @ (pw - x)
            if self.body_jntnum[bid] > 0:
                jid = int(self.body_jntadr[bid])
                jpos_local = self.joint_pos[jid]
                joint_world = x + R @ jpos_local
            else:
                joint_world = x
            distances[bid] = np.linalg.norm(pw - joint_world)
        nearest = int(np.argmin(distances))
        if distances[nearest] >= 0.08:
            return self._make_single_waypoint(body, pw)
        bodies = []
        locals_pts = []
        weights = []
        nearest_name = self.body_id_to_name.get(nearest, None)
        if nearest_name is not None:
            bodies.append(nearest_name)
            locals_pts.append(local_positions[nearest])
            weights.append(1.0 / np.sqrt(max(distances[nearest], 1e-8)))
        parent = int(self.body_parent[nearest])
        if parent > 0 and parent != nearest:
            parent_name = self.body_id_to_name.get(parent, None)
            if parent_name is not None:
                bodies.append(parent_name)
                locals_pts.append(local_positions[parent])
                weights.append(1.0 / np.sqrt(max(distances[parent], 1e-8)))
        if len(bodies) == 0:
            return self._make_single_waypoint(body, pw)
        wsum = float(np.sum(weights))
        if wsum > 0:
            weights = [w / wsum for w in weights]
        else:
            weights = [1.0 for _ in weights]
        return Waypoint(bodies, locals_pts, weights)

    def forward_with_backend(self, body_pos: torch.Tensor, body_rot: torch.Tensor, backend_body_names: list = None):
        names = backend_body_names if backend_body_names is not None else list(self.body_name_to_id.keys())
        if body_pos.ndim == 2:
            pos = body_pos[None]
            rot = body_rot[None]
            B, N = 1, body_pos.shape[0]
        else:
            pos = body_pos
            rot = body_rot
            B, N = body_pos.shape[0], body_pos.shape[1]
        name_to_idx = {n: i for i, n in enumerate(names)}
        M = len(self.muscles)
        W_list = [len(m.waypoints) for m in self.muscles]
        K_list = [max(len(wp.bodies) for wp in m.waypoints) if len(m.waypoints) > 0 else 0 for m in self.muscles]
        maxW = max(W_list) if M > 0 else 0
        maxK = max(K_list) if M > 0 else 0
        padded_idx = torch.full((M, maxW, maxK), -1, dtype=torch.long)
        padded_pts = torch.zeros((M, maxW, maxK, 3), dtype=pos.dtype)
        padded_w = torch.zeros((M, maxW, maxK), dtype=pos.dtype)
        for m_idx, mus in enumerate(self.muscles):
            for w_idx, wp in enumerate(mus.waypoints):
                for k_idx, (body, p_local, w) in enumerate(zip(wp.bodies, wp.local_positions, wp.weights)):
                    idx = name_to_idx.get(body, -1)
                    padded_idx[m_idx, w_idx, k_idx] = idx
                    if idx >= 0:
                        padded_pts[m_idx, w_idx, k_idx] = torch.tensor(p_local, dtype=pos.dtype)
                        padded_w[m_idx, w_idx, k_idx] = float(w)
        padded_idx = padded_idx.to(pos.device)
        padded_pts = padded_pts.to(pos.device)
        padded_w = padded_w.to(pos.device)
        xq = rot[..., 0]; yq = rot[..., 1]; zq = rot[..., 2]; wq = rot[..., 3]
        rotmats = torch.empty((B, N, 3, 3), dtype=pos.dtype, device=pos.device)
        rotmats[..., 0, 0] = 1 - 2 * (yq * yq + zq * zq)
        rotmats[..., 0, 1] = 2 * (xq * yq - zq * wq)
        rotmats[..., 0, 2] = 2 * (xq * zq + yq * wq)
        rotmats[..., 1, 0] = 2 * (xq * yq + zq * wq)
        rotmats[..., 1, 1] = 1 - 2 * (xq * xq + zq * zq)
        rotmats[..., 1, 2] = 2 * (yq * zq - xq * wq)
        rotmats[..., 2, 0] = 2 * (xq * zq - yq * wq)
        rotmats[..., 2, 1] = 2 * (yq * zq + xq * wq)
        rotmats[..., 2, 2] = 1 - 2 * (xq * xq + yq * yq)
        safe_idx = padded_idx.clamp(min=0)
        flat_idx = safe_idx.reshape(-1)
        flat_index_pos = flat_idx.view(1, -1, 1).expand(B, -1, 3)
        pos_sel = torch.gather(pos, 1, flat_index_pos).view(B, M, maxW, maxK, 3)
        flat_index_rot = flat_idx.view(1, -1, 1, 1).expand(B, -1, 3, 3)
        rot_sel = torch.gather(rotmats, 1, flat_index_rot).view(B, M, maxW, maxK, 3, 3)
        world_pts_infl = pos_sel + torch.matmul(rot_sel, padded_pts.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        world_pts = (world_pts_infl * padded_w.unsqueeze(0).unsqueeze(-1)).sum(dim=3)
        out = []
        for m in range(M):
            W = W_list[m]
            pts_env = []
            for b in range(B):
                pts = [tuple(world_pts[b, m, w].detach().cpu().numpy()) for w in range(W)]
                pts_env.append(pts)
            out.append((self.muscles[m].name, pts_env if B > 1 else pts_env[0]))
        return out

    def forward_batched_points(self, body_pos: torch.Tensor, body_rot: torch.Tensor, backend_body_names: list):
        names = backend_body_names
        name_to_idx = {n: i for i, n in enumerate(names)}
        M = len(self.muscles)
        W_list = [len(m.waypoints) for m in self.muscles]
        K_list = [max(len(wp.bodies) for wp in m.waypoints) if len(m.waypoints) > 0 else 0 for m in self.muscles]
        maxW = max(W_list) if M > 0 else 0
        maxK = max(K_list) if M > 0 else 0
        padded_idx = torch.full((M, maxW, maxK), -1, dtype=torch.long)
        padded_pts = torch.zeros((M, maxW, maxK, 3), dtype=body_pos.dtype)
        padded_w = torch.zeros((M, maxW, maxK), dtype=body_pos.dtype)
        for m_idx, mus in enumerate(self.muscles):
            for w_idx, wp in enumerate(mus.waypoints):
                for k_idx, (body, p_local, w) in enumerate(zip(wp.bodies, wp.local_positions, wp.weights)):
                    idx = name_to_idx.get(body, -1)
                    padded_idx[m_idx, w_idx, k_idx] = idx
                    if idx >= 0:
                        padded_pts[m_idx, w_idx, k_idx] = torch.tensor(p_local, dtype=body_pos.dtype)
                        padded_w[m_idx, w_idx, k_idx] = float(w)
        padded_idx = padded_idx.to(body_pos.device)
        padded_pts = padded_pts.to(body_pos.device)
        padded_w = padded_w.to(body_pos.device)
        xq = body_rot[..., 0]; yq = body_rot[..., 1]; zq = body_rot[..., 2]; wq = body_rot[..., 3]
        rotmats = torch.empty((1, body_pos.shape[0], 3, 3), dtype=body_pos.dtype, device=body_pos.device)
        rotmats[..., 0, 0] = 1 - 2 * (yq * yq + zq * zq)
        rotmats[..., 0, 1] = 2 * (xq * yq - zq * wq)
        rotmats[..., 0, 2] = 2 * (xq * zq + yq * wq)
        rotmats[..., 1, 0] = 2 * (xq * yq + zq * wq)
        rotmats[..., 1, 1] = 1 - 2 * (xq * xq + zq * zq)
        rotmats[..., 1, 2] = 2 * (yq * zq - xq * wq)
        rotmats[..., 2, 0] = 2 * (xq * zq - yq * wq)
        rotmats[..., 2, 1] = 2 * (yq * zq + xq * wq)
        rotmats[..., 2, 2] = 1 - 2 * (xq * xq + yq * yq)
        safe_idx = padded_idx.clamp(min=0)
        flat_idx = safe_idx.view(-1)
        pos_sel = body_pos[flat_idx].view(1, M, maxW, maxK, 3)
        rot_sel = rotmats[:, flat_idx].view(1, M, maxW, maxK, 3, 3)
        world_pts_infl = pos_sel + torch.matmul(rot_sel, padded_pts.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        world_pts = (world_pts_infl * padded_w.unsqueeze(0).unsqueeze(-1)).sum(dim=3)
        return world_pts, W_list

    def forward(self):
        out = []
        for mus in self.muscles:
            pts = []
            for wp in mus.waypoints:
                pw = np.zeros(3, dtype=np.float64)
                for body, p_local, w in zip(wp.bodies, wp.local_positions, wp.weights):
                    bid = self.body_name_to_id.get(body, -1)
                    if bid < 0:
                        pw += w * np.array(p_local, dtype=np.float64)
                        continue
                    x = np.array(self.d.xpos[bid], dtype=np.float64)
                    R = np.array(self.d.xmat[bid], dtype=np.float64).reshape(3, 3)
                    pw += w * (x + R @ p_local)
                pts.append(pw)
            out.append((mus.name, pts))
        return out

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
        weights = []
        wp_mj_ids = []
        for mus in self.muscles:
            mus_indices = []
            mus_points = []
            mus_weights = []
            mus_wp_mj = []
            for wp in mus.waypoints:
                wp_indices = []
                wp_points = []
                wp_weights = []
                wp_mj = []
                for body, p_local, w in zip(wp.bodies, wp.local_positions, wp.weights):
                    idx = name_to_idx.get(body, -1)
                    wp_indices.append(idx)
                    wp_points.append(p_local if idx >= 0 else np.zeros(3, dtype=np.float64))
                    wp_weights.append(w if idx >= 0 else 0.0)
                    wp_mj.append(self.body_name_to_id.get(body, -1) if idx >= 0 else -1)
                mus_indices.append(wp_indices)
                mus_points.append(wp_points)
                mus_weights.append(wp_weights)
                mus_wp_mj.append(wp_mj)
            idxs.append(mus_indices)
            locals_pts.append(mus_points)
            weights.append(mus_weights)
            wp_mj_ids.append(mus_wp_mj)
        self._backend_idx_per_muscle = idxs
        self._local_points_per_muscle = locals_pts
        M = len(idxs)
        W_list = [len(w) for w in idxs]
        K_list = [max((len(k) for k in w), default=0) for w in idxs]
        maxW = max(W_list) if M > 0 else 0
        maxK = max(K_list) if M > 0 else 0
        padded_idxs = np.full((M, maxW, maxK), -1, dtype=np.int64)
        padded_local = np.zeros((M, maxW, maxK, 3), dtype=np.float64)
        padded_weights = np.zeros((M, maxW, maxK), dtype=np.float64)
        padded_wp_mj = np.full((M, maxW, maxK), -1, dtype=np.int64)
        for m, W in enumerate(W_list):
            for w_idx in range(W):
                k_len = len(idxs[m][w_idx])
                if k_len == 0:
                    continue
                padded_idxs[m, w_idx, :k_len] = np.array(idxs[m][w_idx], dtype=np.int64)
                padded_local[m, w_idx, :k_len] = np.stack(locals_pts[m][w_idx], axis=0)
                padded_weights[m, w_idx, :k_len] = np.array(weights[m][w_idx], dtype=np.float64)
                padded_wp_mj[m, w_idx, :k_len] = np.array(wp_mj_ids[m][w_idx], dtype=np.int64)
        self._padded_backend_idx = padded_idxs
        self._padded_local_pts = padded_local
        self._padded_weights = padded_weights
        self._padded_wp_bodies_mj = padded_wp_mj
        self._maxW = maxW
        self._maxK = maxK
        l0 = np.zeros((M,), dtype=np.float64)
        for m in range(M):
            W = len(self.muscles[m].waypoints)
            if W < 2:
                l0[m] = 0.0
                continue
            total = 0.0
            for w in range(W - 1):
                p0 = np.zeros(3, dtype=np.float64)
                p1 = np.zeros(3, dtype=np.float64)
                wp0 = self.muscles[m].waypoints[w]
                wp1 = self.muscles[m].waypoints[w + 1]
                for body, p_local, wgt in zip(wp0.bodies, wp0.local_positions, wp0.weights):
                    bid0 = self.body_name_to_id.get(body, -1)
                    if bid0 >= 0:
                        x0 = np.array(self.d.xpos[bid0], dtype=np.float64)
                        R0 = np.array(self.d.xmat[bid0], dtype=np.float64).reshape(3, 3)
                        p0 += wgt * (x0 + R0 @ p_local)
                    else:
                        p0 += wgt * np.array(p_local, dtype=np.float64)
                for body, p_local, wgt in zip(wp1.bodies, wp1.local_positions, wp1.weights):
                    bid1 = self.body_name_to_id.get(body, -1)
                    if bid1 >= 0:
                        x1 = np.array(self.d.xpos[bid1], dtype=np.float64)
                        R1 = np.array(self.d.xmat[bid1], dtype=np.float64).reshape(3, 3)
                        p1 += wgt * (x1 + R1 @ p_local)
                    else:
                        p1 += wgt * np.array(p_local, dtype=np.float64)
                total += np.linalg.norm(p1 - p0)
            l0[m] = total
        self._l_mt0 = l0

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

    def _compute_torque(
        self,
        activations: torch.Tensor,
        JtA: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        torques_active = torch.einsum('bm,bmd->bd', activations, JtA)
        return torques_active + b
