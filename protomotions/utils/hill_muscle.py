import xml.etree.ElementTree as ET
import numpy as np

class HillMuscleModel:
    def __init__(self, muscles):
        self.muscles = muscles
        self.vmax = 10.0
        self.fl_width = 0.45
        self.fv_k = 0.25
        self.passive_k = 3.0
        self.passive_shift = 1.0
    @staticmethod
    def from_xml(path):
        tree = ET.parse(path)
        root = tree.getroot()
        muscles = {}
        for unit in root.findall('Unit'):
            name = unit.attrib['name']
            f0 = float(unit.attrib.get('f0', '1000'))
            lm = float(unit.attrib.get('lm', '1.0'))
            lt = float(unit.attrib.get('lt', '0.2'))
            pen = float(unit.attrib.get('pen_angle', '0.0'))
            lmax = float(unit.attrib.get('lmax', '-0.1'))
            muscles[name] = {'f0': f0, 'l_opt': lm, 'l_tendon': lt, 'pennation': pen, 'lmax': lmax}
        return HillMuscleModel(muscles)
    def force_length_active(self, l, p):
        lopt = p['l_opt']
        x = l / lopt
        w = self.fl_width
        return np.exp(-((x - 1.0) / w) ** 2)
    def force_velocity(self, v, p):
        lopt = p['l_opt']
        vmax = self.vmax * lopt
        y = v / vmax
        num = 1.0 - y
        den = 1.0 + self.fv_k * y
        return np.maximum(0.0, num / den)
    def force_length_passive(self, l, p):
        lopt = p['l_opt']
        x = l / lopt
        if x <= self.passive_shift:
            return 0.0
        return self.passive_k * (x - self.passive_shift) ** 2
    def fiber_to_tendon(self, ff, p):
        pen = p['pennation']
        return ff * np.cos(np.deg2rad(pen))
    def muscle_force(self, name, activation, length, velocity):
        p = self.muscles[name]
        a = np.clip(activation, 0.0, 1.0)
        fl = self.force_length_active(length, p)
        fv = self.force_velocity(velocity, p)
        fp = self.force_length_passive(length, p)
        ff = p['f0'] * (a * fl * fv + fp)
        return self.fiber_to_tendon(ff, p)
    def activations_to_torque(self, activations, lengths, velocities, moment_arms, joint_names=None):
        tau = {}
        for mname, a in activations.items():
            l = lengths[mname]
            v = velocities[mname]
            F = self.muscle_force(mname, a, l, v)
            for jname, r in moment_arms.get(mname, {}).items():
                tau[jname] = tau.get(jname, 0.0) + F * r
        if joint_names is None:
            return tau
        return {j: tau.get(j, 0.0) for j in joint_names}