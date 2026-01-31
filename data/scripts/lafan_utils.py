import re, os, ntpath
import numpy as np
def length(x, axis=-1, keepdims=True):
    """
    Computes vector norm along a tensor axis(axes)

    :param x: tensor
    :param axis: axis(axes) along which to compute the norm
    :param keepdims: indicates if the dimension(s) on axis should be kept
    :return: The length or vector of lengths.
    """
    lgth = np.sqrt(np.sum(x * x, axis=axis, keepdims=keepdims))
    return lgth


def normalize(x, axis=-1, eps=1e-8):
    """
    Normalizes a tensor over some axis (axes)

    :param x: data tensor
    :param axis: axis(axes) along which to compute the norm
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized tensor
    """
    res = x / (length(x, axis=axis) + eps)
    return res


def quat_normalize(x, eps=1e-8):
    """
    Normalizes a quaternion tensor

    :param x: data tensor
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized quaternions tensor
    """
    res = normalize(x, eps=eps)
    return res


def angle_axis_to_quat(angle, axis):
    """
    Converts from and angle-axis representation to a quaternion representation

    :param angle: angles tensor
    :param axis: axis tensor
    :return: quaternion tensor
    """
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    q = np.concatenate([c, s * axis], axis=-1)
    return q


def euler_to_quat(e, order='zyx'):
    """

    Converts from an euler representation to a quaternion representation

    :param e: euler tensor
    :param order: order of euler rotations
    :return: quaternion tensor
    """
    axis = {
        'x': np.asarray([1, 0, 0], dtype=np.float32),
        'y': np.asarray([0, 1, 0], dtype=np.float32),
        'z': np.asarray([0, 0, 1], dtype=np.float32)}

    q0 = angle_axis_to_quat(e[..., 0], axis[order[0]])
    q1 = angle_axis_to_quat(e[..., 1], axis[order[1]])
    q2 = angle_axis_to_quat(e[..., 2], axis[order[2]])

    return quat_mul(q0, quat_mul(q1, q2))


def quat_inv(q):
    """
    Inverts a tensor of quaternions

    :param q: quaternion tensor
    :return: tensor of inverted quaternions
    """
    res = np.asarray([1, -1, -1, -1], dtype=np.float32) * q
    return res


def quat_fk(lrot, lpos, parents):
    """
    Performs Forward Kinematics (FK) on local quaternions and local positions to retrieve global representations

    :param lrot: tensor of local quaternions with shape (..., Nb of joints, 4)
    :param lpos: tensor of local positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of global quaternion, global positions
    """
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(quat_mul_vec(gr[parents[i]], lpos[..., i:i+1, :]) + gp[parents[i]])
        gr.append(quat_mul    (gr[parents[i]], lrot[..., i:i+1, :]))

    res = np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)
    return res


def quat_ik(grot, gpos, parents):
    """
    Performs Inverse Kinematics (IK) on global quaternions and global positions to retrieve local representations

    :param grot: tensor of global quaternions with shape (..., Nb of joints, 4)
    :param gpos: tensor of global positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of local quaternion, local positions
    """
    res = [
        np.concatenate([
            grot[..., :1, :],
            quat_mul(quat_inv(grot[..., parents[1:], :]), grot[..., 1:, :]),
        ], axis=-2),
        np.concatenate([
            gpos[..., :1, :],
            quat_mul_vec(
                quat_inv(grot[..., parents[1:], :]),
                gpos[..., 1:, :] - gpos[..., parents[1:], :]),
        ], axis=-2)
    ]

    return res


def quat_mul(x, y):
    """
    Performs quaternion multiplication on arrays of quaternions

    :param x: tensor of quaternions of shape (..., Nb of joints, 4)
    :param y: tensor of quaternions of shape (..., Nb of joints, 4)
    :return: The resulting quaternions
    """
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)

    return res


def quat_mul_vec(q, x):
    """
    Performs multiplication of an array of 3D vectors by an array of quaternions (rotation).

    :param q: tensor of quaternions of shape (..., Nb of joints, 4)
    :param x: tensor of vectors of shape (..., Nb of joints, 3)
    :return: the resulting array of rotated vectors
    """
    t = 2.0 * np.cross(q[..., 1:], x)
    res = x + q[..., 0][..., np.newaxis] * t + np.cross(q[..., 1:], t)

    return res


def quat_slerp(x, y, a):
    """
    Performs spherical linear interpolation (SLERP) between x and y, with proportion a

    :param x: quaternion tensor
    :param y: quaternion tensor
    :param a: indicator (between 0 and 1) of completion of the interpolation.
    :return: tensor of interpolation results
    """
    len = np.sum(x * y, axis=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    y[neg] = -y[neg]

    a = np.zeros_like(x[..., 0]) + a
    amount0 = np.zeros(a.shape)
    amount1 = np.zeros(a.shape)

    linear = (1.0 - len) < 0.01
    omegas = np.arccos(len[~linear])
    sinoms = np.sin(omegas)

    amount0[linear] = 1.0 - a[linear]
    amount0[~linear] = np.sin((1.0 - a[~linear]) * omegas) / sinoms

    amount1[linear] = a[linear]
    amount1[~linear] = np.sin(a[~linear] * omegas) / sinoms
    res = amount0[..., np.newaxis] * x + amount1[..., np.newaxis] * y

    return res


def quat_between(x, y):
    """
    Quaternion rotations between two 3D-vector arrays

    :param x: tensor of 3D vectors
    :param y: tensor of 3D vetcors
    :return: tensor of quaternions
    """
    res = np.concatenate([
        np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))[..., np.newaxis] +
        np.sum(x * y, axis=-1)[..., np.newaxis],
        np.cross(x, y)], axis=-1)
    return res


def interpolate_local(lcl_r_mb, lcl_q_mb, n_past, n_future):
    """
    Performs interpolation between 2 frames of an animation sequence.

    The 2 frames are indirectly specified through n_past and n_future.
    SLERP is performed on the quaternions
    LERP is performed on the root's positions.

    :param lcl_r_mb:  Local/Global root positions (B, T, 1, 3)
    :param lcl_q_mb:  Local quaternions (B, T, J, 4)
    :param n_past:    Number of frames of past context
    :param n_future:  Number of frames of future context
    :return: Interpolated root and quats
    """
    # Extract last past frame and target frame
    start_lcl_r_mb = lcl_r_mb[:, n_past - 1, :, :][:, None, :, :]  # (B, 1, J, 3)
    end_lcl_r_mb = lcl_r_mb[:, -n_future, :, :][:, None, :, :]

    start_lcl_q_mb = lcl_q_mb[:, n_past - 1, :, :]
    end_lcl_q_mb = lcl_q_mb[:, -n_future, :, :]

    # LERP Local Positions:
    n_trans = lcl_r_mb.shape[1] - (n_past + n_future)
    interp_ws = np.linspace(0.0, 1.0, num=n_trans + 2, dtype=np.float32)
    offset = end_lcl_r_mb - start_lcl_r_mb

    const_trans    = np.tile(start_lcl_r_mb, [1, n_trans + 2, 1, 1])
    inter_lcl_r_mb = const_trans + (interp_ws)[None, :, None, None] * offset

    # SLERP Local Quats:
    interp_ws = np.linspace(0.0, 1.0, num=n_trans + 2, dtype=np.float32)
    inter_lcl_q_mb = np.stack(
        [(quat_normalize(quat_slerp(quat_normalize(start_lcl_q_mb), quat_normalize(end_lcl_q_mb), w))) for w in
         interp_ws], axis=1)

    return inter_lcl_r_mb, inter_lcl_q_mb


def remove_quat_discontinuities(rotations):
    """

    Removing quat discontinuities on the time dimension (removing flips)

    :param rotations: Array of quaternions of shape (T, J, 4)
    :return: The processed array without quaternion inversion.
    """
    rots_inv = -rotations

    for i in range(1, rotations.shape[0]):
        # Compare dot products
        replace_mask = np.sum(rotations[i - 1: i] * rotations[i: i + 1], axis=-1) < np.sum(
            rotations[i - 1: i] * rots_inv[i: i + 1], axis=-1)
        replace_mask = replace_mask[..., np.newaxis]
        rotations[i] = replace_mask * rots_inv[i] + (1.0 - replace_mask) * rotations[i]

    return rotations


# Orient the data according to the las past keframe
def rotate_at_frame(X, Q, parents, n_past=10):
    """
    Re-orients the animation data according to the last frame of past context.

    :param X: tensor of local positions of shape (Batchsize, Timesteps, Joints, 3)
    :param Q: tensor of local quaternions (Batchsize, Timesteps, Joints, 4)
    :param parents: list of parents' indices
    :param n_past: number of frames in the past context
    :return: The rotated positions X and quaternions Q
    """
    # Get global quats and global poses (FK)
    global_q, global_x = quat_fk(Q, X, parents)

    key_glob_Q = global_q[:, n_past - 1: n_past, 0:1, :]  # (B, 1, 1, 4)
    forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] \
                 * quat_mul_vec(key_glob_Q, np.array([0, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :])
    forward = normalize(forward)
    yrot = quat_normalize(quat_between(np.array([1, 0, 0]), forward))
    new_glob_Q = quat_mul(quat_inv(yrot), global_q)
    new_glob_X = quat_mul_vec(quat_inv(yrot), global_x)

    # back to local quat-pos
    Q, X = quat_ik(new_glob_Q, new_glob_X, parents)

    return X, Q


def extract_feet_contacts(pos, lfoot_idx, rfoot_idx, velfactor=0.02):
    """
    Extracts binary tensors of feet contacts

    :param pos: tensor of global positions of shape (Timesteps, Joints, 3)
    :param lfoot_idx: indices list of left foot joints
    :param rfoot_idx: indices list of right foot joints
    :param velfactor: velocity threshold to consider a joint moving or not
    :return: binary tensors of left foot contacts and right foot contacts
    """
    lfoot_xyz = (pos[1:, lfoot_idx, :] - pos[:-1, lfoot_idx, :]) ** 2
    contacts_l = (np.sum(lfoot_xyz, axis=-1) < velfactor)

    rfoot_xyz = (pos[1:, rfoot_idx, :] - pos[:-1, rfoot_idx, :]) ** 2
    contacts_r = (np.sum(rfoot_xyz, axis=-1) < velfactor)

    # Duplicate the last frame for shape consistency
    contacts_l = np.concatenate([contacts_l, contacts_l[-1:]], axis=0)
    contacts_r = np.concatenate([contacts_r, contacts_r[-1:]], axis=0)

    return contacts_l, contacts_r

channelmap = {
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z'
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x': 0,
    'y': 1,
    'z': 2,
}


class Anim(object):
    """
    A very basic animation object
    """
    def __init__(self, quats, pos, offsets, parents, bones, fps=30.0):
        """
        :param quats: local quaternions tensor
        :param pos: local positions tensor
        :param offsets: local joint offsets
        :param parents: bone hierarchy
        :param bones: bone names
        :param fps: frames per second
        """
        self.quats = quats
        self.pos = pos
        self.offsets = offsets
        self.parents = parents
        self.bones = bones
        self.fps = fps


def read_bvh(filename, start=None, end=None, order=None):
    """
    Reads a BVH file and extracts animation information.

    :param filename: BVh filename
    :param start: start frame
    :param end: end frame
    :param order: order of euler rotations
    :return: A simple Anim object conatining the extracted information.
    """

    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []
    orients = np.array([]).reshape((0, 4))
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)
    frametime = 0.033333

    # Parse the  file, line by line
    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "{" in line: continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start) - 1
            else:
                fnum = int(fmatch.group(1))
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue

        dmatch = line.strip().split(' ')
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - start if start else i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()

    rotations = euler_to_quat(np.radians(rotations), order=order)
    rotations = remove_quat_discontinuities(rotations)
    
    fps = 1.0 / frametime if frametime > 0 else 30.0

    return Anim(rotations, positions, offsets, parents, names, fps)


def get_lafan1_set(bvh_path, actors, window=50, offset=20):
    """
    Extract the same test set as in the article, given the location of the BVH files.

    :param bvh_path: Path to the dataset BVH files
    :param list: actor prefixes to use in set
    :param window: width  of the sliding windows (in timesteps)
    :param offset: offset between windows (in timesteps)
    :return: tuple:
        X: local positions
        Q: local quaternions
        parents: list of parent indices defining the bone hierarchy
        contacts_l: binary tensor of left-foot contacts of shape (Batchsize, Timesteps, 2)
        contacts_r: binary tensor of right-foot contacts of shape (Batchsize, Timesteps, 2)
    """
    npast = 10
    subjects = []
    seq_names = []
    X = []
    Q = []
    contacts_l = []
    contacts_r = []

    # Extract
    bvh_files = os.listdir(bvh_path)

    for file in bvh_files:
        if file.endswith('.bvh'):
            seq_name, subject = ntpath.basename(file[:-4]).split('_')

            if subject in actors:
                print('Processing file {}'.format(file))
                seq_path = os.path.join(bvh_path, file)
                anim = read_bvh(seq_path)

                # Sliding windows
                i = 0
                while i+window < anim.pos.shape[0]:
                    q, x = quat_fk(anim.quats[i: i+window], anim.pos[i: i+window], anim.parents)
                    # Extract contacts
                    c_l, c_r = extract_feet_contacts(x, [3, 4], [7, 8], velfactor=0.02)
                    X.append(anim.pos[i: i+window])
                    Q.append(anim.quats[i: i+window])
                    seq_names.append(seq_name)
                    subjects.append(subjects)
                    contacts_l.append(c_l)
                    contacts_r.append(c_r)

                    i += offset

    X = np.asarray(X)
    Q = np.asarray(Q)
    contacts_l = np.asarray(contacts_l)
    contacts_r = np.asarray(contacts_r)

    # Sequences around XZ = 0
    xzs = np.mean(X[:, :, 0, ::2], axis=1, keepdims=True)
    X[:, :, 0, 0] = X[:, :, 0, 0] - xzs[..., 0]
    X[:, :, 0, 2] = X[:, :, 0, 2] - xzs[..., 1]

    # Unify facing on last seed frame
    X, Q = rotate_at_frame(X, Q, anim.parents, n_past=npast)

    return X, Q, anim.parents, contacts_l, contacts_r


def get_train_stats(bvh_folder, train_set):
    """
    Extract the same training set as in the paper in order to compute the normalizing statistics
    :return: Tuple of (local position mean vector, local position standard deviation vector, local joint offsets tensor)
    """
    print('Building the train set...')
    xtrain, qtrain, parents, _, _ = get_lafan1_set(bvh_folder, train_set, window=50, offset=20)

    print('Computing stats...\n')
    # Joint offsets : are constant, so just take the first frame:
    offsets = xtrain[0:1, 0:1, 1:, :]  # Shape : (1, 1, J, 3)

    # Global representation:
    q_glbl, x_glbl = quat_fk(qtrain, xtrain, parents)

    # Global positions stats:
    x_mean = np.mean(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
    x_std = np.std(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)

    return x_mean, x_std, offsets