"""
In this module I compute the Jacobian matrices for each link’s center of mass.
These CoM Jacobians will be used in the Newton–Euler dynamics formulation.
"""

import numpy as np
from kinematics.forward_kinematics import forward
from params import DH

JOINT_TYPES = [link[4] for link in DH]
n = len(JOINT_TYPES)

def compute_com_jacobians(q, Pc_local, joint_types):
    J_list = []
    fk = forward(q)
    T_links = fk.Ts_abs
    zs = [T[:3, 2] for T in T_links]
    ps = [T[:3, 3] for T in T_links]

    for i in range(n):

        T_i = T_links[i]

        # Position of the link's COM in world coordinates
        x_com_world = T_i[:3, :3] @ Pc_local[i] + T_i[:3, 3]

        J_i = np.zeros((6, n))

        for j in range(n):
            if j <= i:
                # Get axis (z_j) and origin (p_j) for joint j+1
                z_j_axis = zs[j]
                p_j_origin = ps[j]

                if joint_types[j] == 'R':
                    J_i[3:, j] = z_j_axis

                    J_i[:3, j] = np.cross(z_j_axis, x_com_world - p_j_origin)
                else:
                    J_i[3:, j] = 0

                    J_i[:3, j] = z_j_axis

        J_list.append(J_i)
    return J_list