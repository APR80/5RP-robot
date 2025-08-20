import numpy as np
from kinematics.forward_kinematics import forward
from params import DH

# Precompute constants
JOINT_TYPES = [link[4] for link in DH]
N_JOINTS    = len(DH)

def jacobian(q):
    """
    6×n Jacobian for q (length n).
    """
    q = np.asarray(q, float)
    if q.size != N_JOINTS:
        raise ValueError(f"Expected {N_JOINTS} joints, got {q.size}")

    fk  = forward(q)
    Ts  = fk.Ts_abs
    o_n = Ts[-1][:3, 3]

    J = np.zeros((6, N_JOINTS), float)
    for i, (T_i, jt) in enumerate(zip(Ts, JOINT_TYPES)):
        z, o = T_i[:3,2], T_i[:3,3]
        if jt == 'R':
            J[:3, i] = np.cross(z, o_n - o)
            J[3:, i] = z
        else:
            J[:3, i] = z

    return J

if __name__ == "__main__":
    q    = [np.pi/2, 0, 0, 0, 0, 0.05]
    qdot = np.array([1, -1, 0, 0, 0, 0.01])
    J    = jacobian(q)
    x    = J @ qdot

    print("[vx,vy,vz,wx,wy,wz]:", np.round(x,6))
