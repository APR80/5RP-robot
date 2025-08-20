import numpy as np
from kinematics.forward_kinematics import forward

def differential_kinematics(q: np.ndarray, qdot: np.ndarray):
    fk = forward(q)
    R_rel = fk.R_rel
    P_rel = fk.P_rel

    z = np.array([0, 0, 1])
    w = np.zeros((7, 3))
    v = np.zeros((7, 3))

    for i in range(6):
        R_prev = R_rel[i].T
        w[i+1] = R_prev @ w[i] + qdot[i] * z
        v[i+1] = R_prev @ (v[i] + np.cross(w[i], P_rel[i]))

    # transform into base frame
    R06    = fk.Ts_abs[-1][:3, :3]
    v_base = R06 @ v[-1]
    w_base = R06 @ w[-1]
    return v_base, w_base

if __name__ == "__main__":
    q     = np.array([np.pi/2, 0, 0, 0, 0, 0.05])
    qdot  = np.array([1, -1, 0, 0, 0, 0.01])
    v_lin, w_ang = differential_kinematics(q, qdot)
    print("Linear velocity (base frame): ", v_lin)
    print("Angular velocity (base frame):", w_ang)
