import torch
from isaac_utils.rotations import quaternion_to_matrix
from .muscle_parser import CharactorMuscle

class MuscleController:
    def __init__(self, muscle_xml_path: str, rig_path: str, device: torch.device):
        self.muscle_char = CharactorMuscle(muscle_xml_path, rig_path, device)
        self.n_muscles = len(self.muscle_char.muscles)

    def prepare(self, body_names: list, dof_names: list):
        self.muscle_char.prepare_mapping(body_names, dof_names)
   
    def g_al(self, x: torch.Tensor):
        return torch.exp(-((x - 1.0) * (x - 1.0)) / self.muscle_char.gamma)

    def g_pl(self, x: torch.Tensor):
        f_pl = (torch.exp(self.muscle_char.k_pe * (x - 1.0) / self.muscle_char.e_mo) - 1.0) / (torch.exp(self.muscle_char.k_pe) - 1.0)
        return torch.where(x < 1.0, torch.zeros_like(f_pl), f_pl)

    def g_t(self, e_t: torch.Tensor):
        toe = self.muscle_char.f_toe / (torch.exp(self.muscle_char.k_toe) - 1.0) * (torch.exp(self.muscle_char.k_toe * e_t / self.muscle_char.e_toe) - 1.0)
        lin = self.muscle_char.k_lin * (e_t - self.muscle_char.e_toe) + self.muscle_char.f_toe
        return torch.where(e_t <= self.muscle_char.e_t0, toe, lin)

    def g(self, l_m: torch.Tensor, l_mt: torch.Tensor, activation: torch.Tensor):
        x = l_m / self.muscle_char.l_m0.view(1, -1)
        e_t = (l_mt - l_m - self.muscle_char.l_t0.view(1, -1)) / self.muscle_char.l_t0.view(1, -1)
        return self.g_t(e_t) - (self.g_pl(x) + activation * self.g_al(x))

    def update_muscle_features(
        self,
        body_pos: torch.Tensor,
        body_rot: torch.Tensor,
        body_com_pos: torch.Tensor,
        jacobians: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor):
        
        pos = body_pos
        rot = body_rot
        com = body_com_pos
        jac = jacobians

        B, N, _ = pos.shape
        device = pos.device
        dtype = pos.dtype
        R = quaternion_to_matrix(rot, w_last=True)

        padded_backend_idx = torch.as_tensor(self.muscle_char._padded_backend_idx, dtype=torch.long, device=device)
        local_pts = torch.as_tensor(self.muscle_char._padded_local_pts, dtype=dtype, device=device)
        weights = torch.as_tensor(self.muscle_char._padded_weights, dtype=dtype, device=device)
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

        l_mt0 = torch.as_tensor(self.muscle_char._l_mt0, dtype=dtype, device=device).clamp(min=1e-8)
        l_mt = (seg_len.squeeze(-1) * seg_mask.unsqueeze(0)).sum(dim=-1) / l_mt0.view(1, M)
        l_m = l_mt - self.muscle_char.l_t0.view(1, -1)
        x = l_m / self.muscle_char.l_m0.view(1, -1)
        f_active = self.muscle_char.f0.view(1, -1) * self.g_al(x)
        f_passive = self.muscle_char.f0.view(1, -1) * self.g_pl(x)

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
    
    def forward_batched_points(self, body_pos: torch.Tensor, body_rot: torch.Tensor, backend_body_names: list):
        names = backend_body_names
        name_to_idx = {n: i for i, n in enumerate(names)}
        M = self.n_muscles
        W_list = [len(m.waypoints) for m in self.muscle_char.muscles]
        K_list = [max(len(wp.bodies) for wp in m.waypoints) if len(m.waypoints) > 0 else 0 for m in self.muscle_char.muscles]
        maxW = max(W_list) if M > 0 else 0
        maxK = max(K_list) if M > 0 else 0
        padded_idx = torch.full((M, maxW, maxK), -1, dtype=torch.long)
        padded_pts = torch.zeros((M, maxW, maxK, 3), dtype=body_pos.dtype)
        padded_w = torch.zeros((M, maxW, maxK), dtype=body_pos.dtype)
        for m_idx, mus in enumerate(self.muscle_char.muscles):
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

    def _compute_torque(
        self,
        activations: torch.Tensor,
        JtA: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        torques_active = torch.einsum('bm,bmd->bd', activations, JtA)
        return torques_active + b
