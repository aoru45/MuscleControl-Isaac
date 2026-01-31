import torch
import numpy as np
import trimesh
from isaaclab.terrains.utils import create_prim_from_mesh
from pxr import UsdGeom, Gf, Usd, Vt, UsdShade, Sdf, UsdPhysics
import omni.usd
from typing import Dict, Any, Tuple

def _quat_to_rot_xyzw(q: torch.Tensor):
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    w = q[..., 3]
    R = torch.empty((*q.shape[:-1], 3, 3), dtype=q.dtype, device=q.device)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def _compute_tangents_and_frames(points):
    # points: (N, 3)
    N = len(points)
    tangents = np.zeros_like(points)
    
    # Compute tangents
    if N > 1:
        tangents[:-1] = points[1:] - points[:-1]
        tangents[-1] = tangents[-2] # Repeat last tangent
        
        # Smoother tangents for internal points
        if N > 2:
            tangents[1:-1] = 0.5 * (points[2:] - points[:-2])
            
    # Normalize
    norms = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8
    tangents = tangents / norms
    
    # Compute Frames (Normal and Binormal) using parallel transport approximation
    normals = np.zeros_like(points)
    binormals = np.zeros_like(points)
    
    # Initial frame
    t0 = tangents[0]
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(t0, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
        
    n0 = np.cross(t0, ref)
    n0 /= np.linalg.norm(n0) + 1e-8
    b0 = np.cross(t0, n0)
    
    normals[0] = n0
    binormals[0] = b0
    
    for i in range(N - 1):
        t_cur = tangents[i]
        t_next = tangents[i+1]
        n_cur = normals[i]
        
        # Project n_cur onto plane of t_next
        n_next = n_cur - np.dot(n_cur, t_next) * t_next
        n_next_norm = np.linalg.norm(n_next)
        
        if n_next_norm < 1e-6:
             ref = np.array([0.0, 0.0, 1.0])
             if abs(np.dot(t_next, ref)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
             n_next = np.cross(t_next, ref)
        
        n_next /= (np.linalg.norm(n_next) + 1e-8)
        b_next = np.cross(t_next, n_next)
        
        normals[i+1] = n_next
        binormals[i+1] = b_next
        
    return tangents, normals, binormals

def build_muscle_mesh(
    muscle_ctl,
    body_pos: torch.Tensor,
    body_rot_wxyz: torch.Tensor,
    body_names,
    radius=0.01,
    radial_segments=12,
):
    rot_xyzw = torch.stack(
        (body_rot_wxyz[:, 1], body_rot_wxyz[:, 2],
         body_rot_wxyz[:, 3], body_rot_wxyz[:, 0]), -1)

    world_pts, W_list = muscle_ctl.forward_batched_points(
        body_pos, rot_xyzw, list(body_names))

    pts = world_pts[0].cpu().numpy()
    R = _quat_to_rot_xyzw(rot_xyzw).cpu().numpy()
    X = body_pos.cpu().numpy()
    name_to_idx = {n: i for i, n in enumerate(body_names)}

    all_verts, all_faces = [], []
    jointA, jointB, wA, wB, pLocalA, pLocalB = [], [], [], [], [], []
    segment_id_per_face = []
    segment_to_muscle_map = []
    segment_id = 0
    base_idx = 0

    waypoint_world = []

    for m_idx, W in enumerate(W_list):
        if W < 2:
            continue
            
        points = pts[m_idx, :W].astype(np.float32)
        wp_influences = []
        for wp in muscle_ctl.muscle_char.muscles[m_idx].waypoints[:W]:
            idxs = []
            ws = []
            for body, w in zip(wp.bodies, wp.weights):
                idx = name_to_idx.get(body, -1)
                if idx >= 0 and w > 0:
                    idxs.append(idx)
                    ws.append(float(w))
            if len(idxs) == 0:
                idxs = [0, 0]
                ws = [1.0, 0.0]
            elif len(idxs) == 1:
                idxs = [idxs[0], idxs[0]]
                ws = [1.0, 0.0]
            else:
                order = np.argsort(ws)[::-1]
                idxs = [idxs[order[0]], idxs[order[1]]]
                ws = [ws[order[0]], ws[order[1]]]
                wsum = ws[0] + ws[1]
                if wsum > 0:
                    ws = [ws[0] / wsum, ws[1] / wsum]
                else:
                    ws = [1.0, 0.0]
            wp_influences.append((idxs[0], idxs[1], ws[0], ws[1]))
        
        for w in range(W):
            waypoint_world.append(points[w])
            
        tangents, normals, binormals = _compute_tangents_and_frames(points)
        
        # Fusiform profile
        dists = np.linalg.norm(points[1:] - points[:-1], axis=1)
        cum_dists = np.concatenate(([0], np.cumsum(dists)))
        total_len = cum_dists[-1]
        if total_len < 1e-6:
            us = np.zeros(W)
        else:
            us = cum_dists / total_len
            
        radii = radius * (0.6 + 1.4 * np.sin(np.pi * us))
        
        muscle_verts = []
        for i in range(W):
            p = points[i]
            n = normals[i]
            b = binormals[i]
            r = radii[i]
            
            for k in range(radial_segments):
                angle = 2.0 * np.pi * k / radial_segments
                c = np.cos(angle)
                s = np.sin(angle)
                
                v_pos = p + r * (c * n + s * b)
                muscle_verts.append(v_pos)
                
                jA, jB, w_a, w_b = wp_influences[i]
                jointA.append(jA)
                jointB.append(jB)
                wA.append(w_a)
                wB.append(w_b)

                v_local_a = R[jA].T @ (v_pos - X[jA])
                v_local_b = R[jB].T @ (v_pos - X[jB])
                pLocalA.append(v_local_a)
                pLocalB.append(v_local_b)
        
        muscle_verts = np.array(muscle_verts, dtype=np.float32)
        all_verts.append(muscle_verts)
        
        muscle_faces = []
        for i in range(W - 1):
            for k in range(radial_segments):
                k_next = (k + 1) % radial_segments
                
                idx0 = i * radial_segments + k
                idx1 = i * radial_segments + k_next
                idx2 = (i + 1) * radial_segments + k
                idx3 = (i + 1) * radial_segments + k_next
                
                a = idx0 + base_idx
                b = idx1 + base_idx
                c = idx2 + base_idx
                d = idx3 + base_idx
                
                muscle_faces.append([a, c, b])
                muscle_faces.append([b, c, d])
                
        muscle_faces = np.array(muscle_faces, dtype=np.int32)
        all_faces.append(muscle_faces)
        
        num_faces = muscle_faces.shape[0]
        segment_id_per_face.append(np.full((num_faces,), segment_id, np.int32))
        segment_to_muscle_map.append(m_idx)
        
        base_idx += len(muscle_verts)
        segment_id += 1

    if len(all_verts) == 0:
        return trimesh.Trimesh(), {}

    V = np.concatenate(all_verts, axis=0)
    F = np.concatenate(all_faces, axis=0)

    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)

    skin = dict(
        jointA=np.array(jointA, np.int32),
        jointB=np.array(jointB, np.int32),
        wA=np.array(wA, np.float32),
        wB=np.array(wB, np.float32),
        pLocalA=np.array(pLocalA, np.float32),
        pLocalB=np.array(pLocalB, np.float32),
        segment_id_per_face=np.concatenate(segment_id_per_face),
        segment_to_muscle_map=np.array(segment_to_muscle_map, np.int32),
        num_segments=segment_id,
        waypoint_world=np.array(waypoint_world, np.float32),
    )
    return mesh, skin

def create_muscle_mesh_prim(prim_path: str, mesh: trimesh.Trimesh, visual_material=None):
    # We ignore visual_material here and create a custom PBR material for muscles
    create_prim_from_mesh(prim_path, mesh, visual_material=None, physics_material=None)
    
    stage = omni.usd.get_context().get_stage()
    # create_prim_from_mesh usually puts the mesh at {prim_path}/mesh
    mesh_prim_path = f"{prim_path}/mesh"
    mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
    
    if not mesh_prim.IsValid():
        # Fallback: maybe it's at prim_path directly
        mesh_prim = stage.GetPrimAtPath(prim_path)
        if not mesh_prim.IsValid():
            print(f"[Warning] Could not find muscle mesh prim at {mesh_prim_path} or {prim_path}")
            return

    # Disable collision for muscle mesh to prevent it from affecting the simulation
    collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
    collision_api.CreateCollisionEnabledAttr().Set(False)

    # Define custom muscle material
    mat_path = f"{prim_path}/MuscleMaterial"
    material = UsdShade.Material.Define(stage, mat_path)
    pbr_shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
    pbr_shader.CreateIdAttr("UsdPreviewSurface")
    
    # Muscle properties: Natural tissue look
    # Higher roughness for less plastic/wet look, but still some sheen
    pbr_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    pbr_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    pbr_shader.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.4)
    # Remove clearcoat to reduce "slimy" artificial look
    pbr_shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.0)
    pbr_shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.0)
    
    # Create Primvar Reader to read displayColor from mesh
    primvar_reader = UsdShade.Shader.Define(stage, f"{mat_path}/primvarReader")
    primvar_reader.CreateIdAttr("UsdPrimvarReader_float3")
    primvar_reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("displayColor")
    # Fallback to Neutral Gray
    primvar_reader.CreateInput("fallback", Sdf.ValueTypeNames.Color3f).Set((0.75, 0.75, 0.75))
    
    # Connect Primvar Reader to Diffuse Color
    pbr_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(primvar_reader.ConnectableAPI(), "result")
    
    # Connect shader to material
    material.CreateSurfaceOutput().ConnectToSource(pbr_shader.ConnectableAPI(), "surface")
    
    # Bind material to mesh
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)

def update_muscle_mesh_points(prim_path: str, skin: Dict[str, Any], body_pos: torch.Tensor, body_rot_wxyz: torch.Tensor, muscle_colors=None, skeletal_color=(0.45, 0.02, 0.02)):
    rot_xyzw = torch.stack((body_rot_wxyz[:, 1], body_rot_wxyz[:, 2], body_rot_wxyz[:, 3], body_rot_wxyz[:, 0]), dim=-1)
    R = _quat_to_rot_xyzw(rot_xyzw)
    X = body_pos
    jA = torch.from_numpy(skin["jointA"]).to(body_pos.device)
    jB = torch.from_numpy(skin["jointB"]).to(body_pos.device)
    wA = torch.from_numpy(skin["wA"]).to(body_pos.device)
    wB = torch.from_numpy(skin["wB"]).to(body_pos.device)
    pA = torch.from_numpy(skin["pLocalA"]).to(body_pos.device)
    pB = torch.from_numpy(skin["pLocalB"]).to(body_pos.device)
    XA = X[jA]
    XB = X[jB]
    RA = R[jA]
    RB = R[jB]
    PA = XA + torch.matmul(RA, pA.unsqueeze(-1)).squeeze(-1)
    PB = XB + torch.matmul(RB, pB.unsqueeze(-1)).squeeze(-1)
    P = wA.view(-1, 1) * PA + wB.view(-1, 1) * PB
    arr = P.detach().cpu().numpy().astype(np.float32)
    stage = omni.usd.get_context().get_stage()
    target_path = f"{prim_path}/mesh"
    prim = stage.GetPrimAtPath(target_path)
    mesh = UsdGeom.Mesh(prim)
    vs = Vt.Vec3fArray.FromNumpy(arr)
    mesh.CreatePointsAttr().Set(vs)
    
    if muscle_colors is not None:
        if isinstance(muscle_colors, torch.Tensor):
            muscle_colors = muscle_colors.detach().cpu().numpy()
        else:
            muscle_colors = np.array(muscle_colors)
            
        # Append skeletal color for spheres
        # We assume muscle_colors has shape (num_muscle_segments, 3)
        # We append one row for skeletal color
        sk_color = np.array(skeletal_color, dtype=np.float32)
        all_colors = np.vstack([muscle_colors, sk_color[None, :]])
        
        segment_ids = skin["segment_id_per_face"]
        # segment_ids has values 0..num_segments-1 for muscles, and num_segments for spheres
        # So we can directly use it as index
        
        face_colors = all_colors[segment_ids] # (num_faces, 3)
        
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(Vt.Vec3fArray.FromNumpy(face_colors))
