import xml.etree.ElementTree as ET
import numpy as np
import torch
from typing import Dict, List
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

class CharactorMuscle:
    def __init__(self, muscle_xml_path, rig_path, device):
        self.device = device
        self.prepare_mujoco(rig_path)
        self.muscles = self._parse_muscles(muscle_xml_path)
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
        self._padded_backend_idx = None
        self._padded_local_pts = None
        self._padded_weights = None
        self._maxW = 0
        self._maxK = 0
        self._l_mt0 = None
        self._dof_prepared = False
        self._dof_bodyid = None
        self._dof_joint_pos_local = None
        self._dof_joint_axis_local = None
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
 
        self._dof_prepared = True
    def prepare_body_geometry(self, body_names: list):
        if self._backend_idx_per_muscle is not None:
            return
        name_to_idx = {n: i for i, n in enumerate(body_names)}
        self._backend_name_to_idx = name_to_idx
        nbody = int(len(body_names))
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
    def prepare_mujoco(self, rig_path):
        import mujoco
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
    def prepare_mapping(self, body_names: list, dof_names: list):
        self.prepare_body_geometry(body_names)
        self.prepare_dof_geometry(dof_names)
    def _parse_muscles(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        muscles = []
        names, f0, lm0, lt0, lmax = [], [], [], [], []
        for unit in root.findall('Unit'):
            name = unit.attrib.get('name', '')
            names.append(name)
            f0.append(float(unit.attrib.get('f0', '1000')))
            lm0.append(float(unit.attrib.get('lm', '1.0')))
            lt0.append(float(unit.attrib.get('lt', '0.2')))
            lmax.append(float(unit.attrib.get('lmax', '-0.1')))
            wps = []
            attrs = {
                'f0': f0[-1],
                'lm': lm0[-1],
                'lt': lt0[-1],
                'lmax': lmax[-1],
            }
            waypoints = list(unit.findall('Waypoint'))
            num_waypoints = len(waypoints)
            for w_idx, wp in enumerate(waypoints):
                body = wp.attrib['body']
                p = [float(x) for x in wp.attrib['p'].strip().split()]
                pw = self._rx90(p)
                if w_idx == 0 or w_idx == num_waypoints - 1:
                    wps.append(self._make_single_waypoint(body, pw))
                else:
                    wps.append(self._make_lbs_waypoint(body, pw))
            if len(wps) >= 2:
                muscles.append(Muscle(name, wps, attrs))
        self.f0 = torch.tensor(f0, device=self.device, dtype=torch.float)
        self.l_m0 = torch.tensor(lm0, device=self.device, dtype=torch.float)
        self.l_t0 = torch.tensor(lt0, device=self.device, dtype=torch.float)
        self.lmax = torch.tensor(lmax, device=self.device, dtype=torch.float)
        return muscles
    def _make_single_waypoint(self, body: str, pw: np.ndarray) -> Waypoint:
        bid = self.body_name_to_id.get(body, -1)
        if bid >= 0:
            x, R = self._get_body_pose(bid)
            pl = R.T @ (pw - x)
        else:
            pl = pw
        return Waypoint([body], [pl], [1.0])
    def _rx90(self, p):
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        return np.array([x, -z, y], dtype=np.float64)
    def _get_body_pose(self, bid: int):
        x = np.array(self.d.xpos[bid], dtype=np.float64)
        R = np.array(self.d.xmat[bid], dtype=np.float64).reshape(3, 3)
        return x, R
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