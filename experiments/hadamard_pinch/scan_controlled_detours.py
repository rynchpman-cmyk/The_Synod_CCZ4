"""Screen thermodynamic detours around the reference pinch coordinates."""

from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals

from audit_transit_corridor import EntropyPencil, normal
from validate_pinch import PARAMETERS
from causal_repair.nulloth import NullothMasterFunction


def classify(master, background, phi, v):
    pencil = EntropyPencil(master, background, phi, v)
    collision = pencil.collision((0,89.9))
    if collision['defective']:
        return 'oblique defect'
    native = pencil.native[0]
    if min(native.legendre_minimum_eigenvalue, native.thermodynamic_minimum_eigenvalue) <= PARAMETERS.convexity_floor:
        return 'constitutive convexity'
    for angle in np.linspace(0,90,46):
        a0, a = pencil.material(normal(angle))
        roots = eigvals(a,a0)
        if max(abs(roots.imag)) > 1e-8:
            return 'complex characteristics'
        if max(abs(roots)) > 1+1e-8:
            return 'causality'
        _,rank,condition,_ = master._clustered_eigenbasis(np.linalg.solve(a0,a))
        if rank < 9 or condition >= PARAMETERS.characteristic_condition_limit:
            return 'eigenbasis conditioning'
    return 'screened'


def main():
    folder = Path(__file__).resolve().parent
    phi0,v0,_ = json.loads((folder/'hadamard_pinch_summary.json').read_text())['pinch_phi_v_speed']
    master = NullothMasterFunction(PARAMETERS)
    rows = []
    for background in product((.8,1.,1.2),(.2,.25,.3),(.2,.3,.45)):
        for phi in (phi0-.004,phi0,phi0+.004,.02):
            tests = [(v,classify(master, background,phi,v)) for v in np.r_[np.linspace(.18,.70,9),v0-.01,v0,v0+.01]]
            valid = [v for v,outcome in tests if outcome=='screened']
            rows.append({'background_n_d_eps':background,'phi':phi,'tests':tests,
                         'highest_screened_v':max(valid,default=None)})
    target = [(row['phi'],v,row['background_n_d_eps']) for row in rows for v,status in row['tests']
              if status=='screened' and abs(v-v0)<=.011]
    data = {'scope':'collinear homogeneous constitutive detour screen','slices':rows,
            'screened_at_reference_velocities':target}
    (folder/'hadamard_pinch_thermodynamic_detours.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(len(rows),'slices',sum(len(row['tests']) for row in rows),'states',len(target),'screened at reference velocities',flush=True)


if __name__=='__main__':
    main()
