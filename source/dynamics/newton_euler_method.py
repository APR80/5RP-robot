import numpy as np
from params import DH, m, Ic, Pc
from kinematics.forward_kinematics import forward

class NewtonEuler:
    """
    Calculates the inverse dynamics of a robot manipulator using the
    recursive Newton-Euler algorithm.
    """

    def __init__(self, dh_params, masses, inertias, com_positions, gravity=np.array([0, 0, -9.81])):

        self.n = len(dh_params)
        self.m = masses
        self.Ic = inertias
        self.Pc = com_positions
        self.joint_types = [p[4] for p in dh_params]
        self.gravity = gravity
        self.k = np.array([0., 0., 1.])

    def calculate_torques(self, q: np.ndarray, qdot: np.ndarray, qdotdot: np.ndarray, use_gravity=True) -> np.ndarray:
        """
        the required joint torques/forces(tau = M @ qdotdot + C + G).
        """
        fk_result = forward(q)
        R_rel = fk_result.R_rel
        P_rel = fk_result.P_rel

        w, wdot, vdot, vdotc = self._forward_propagation(qdot, qdotdot, R_rel, P_rel, use_gravity)
        tau = self._backward_propagation(w, wdot, vdotc, R_rel, P_rel)

        return tau

    def calculate_mass_matrix(self, q: np.ndarray) -> np.ndarray:

        M = np.zeros((self.n, self.n))
        # Set velocity and gravity to zero
        qdot_zeros = np.zeros(self.n)
        for i in range(self.n):
            qdotdot_unit = np.zeros(self.n)
            qdotdot_unit[i] = 1.0
            # Calculate the i-th column of M
            M[:, i] = self.calculate_torques(q, qdot_zeros, qdotdot_unit, use_gravity=False)
        return M

    def calculate_coriolis_vector(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:

        # Set acceleration and gravity to zero
        qdotdot_zeros = np.zeros(self.n)
        return self.calculate_torques(q, qdot, qdotdot_zeros, use_gravity=False)

    def calculate_gravity_vector(self, q: np.ndarray) -> np.ndarray:

        # Set velocities and accelerations to zero
        qdot_zeros = np.zeros(self.n)
        qdotdot_zeros = np.zeros(self.n)
        return self.calculate_torques(q, qdot_zeros, qdotdot_zeros, use_gravity=True)

    def _forward_propagation(self, qdot, qdotdot, R, P, use_gravity):
        """Calculates accelerations of CoM of links from base to end-effector."""
        w = [np.zeros(3) for _ in range(self.n + 1)]
        wdot = [np.zeros(3) for _ in range(self.n + 1)]
        vdot = [np.zeros(3) for _ in range(self.n + 1)]
        vdotc = [np.zeros(3) for _ in range(self.n + 1)]

        #account for gravity
        if use_gravity:
            vdot[0] = -self.gravity

        for i in range(self.n):
            R_i_T = R[i].T
            w[i + 1] = R_i_T @ w[i]
            wdot[i + 1] = R_i_T @ wdot[i]
            vdot[i + 1] = R_i_T @ (np.cross(wdot[i], P[i]) + np.cross(w[i], np.cross(w[i], P[i])) + vdot[i])

            if self.joint_types[i] == 'R':
                w[i + 1] += qdot[i] * self.k
                wdot[i + 1] += np.cross(R_i_T @ w[i], qdot[i] * self.k) + qdotdot[i] * self.k
            else:
                vdot[i + 1] += 2 * np.cross(w[i + 1], qdot[i] * self.k) + qdotdot[i] * self.k

            vdotc[i] = np.cross(wdot[i + 1], self.Pc[i]) + \
                       np.cross(w[i + 1], np.cross(w[i + 1], self.Pc[i])) + vdot[i + 1]

        return w, wdot, vdot, vdotc

    def _backward_propagation(self, w, wdot, vdotc, R, P):
        """Calculates forces and torques from end-effector to base."""
        f = [np.zeros(3) for _ in range(self.n + 1)]
        n = [np.zeros(3) for _ in range(self.n + 1)]
        tau = np.zeros(self.n)

        for i in range(self.n - 1, -1, -1):
            Fi = self.m[i] * vdotc[i]
            Ni = self.Ic[i] @ wdot[i + 1] + np.cross(w[i + 1], self.Ic[i] @ w[i + 1])

            f[i] = R[i + 1] @ f[i + 1] + Fi if i < self.n - 1 else Fi

            n_prop = R[i + 1] @ n[i + 1] if i < self.n - 1 else np.zeros(3)
            p_prop = R[i + 1] @ f[i + 1] if i < self.n - 1 else np.zeros(3)

            n[i] = Ni + n_prop + np.cross(self.Pc[i], Fi) + np.cross(P[i + 1], p_prop) if i < self.n - 1 \
                else Ni + np.cross(self.Pc[i], Fi)

            if self.joint_types[i] == 'R':
                tau[i] = n[i].T @ self.k
            else:
                tau[i] = f[i].T @ self.k

        return tau


if __name__ == '__main__':
    #example usage
    inv_dyn = NewtonEuler(dh_params=DH, masses=m, inertias=Ic, com_positions=Pc)

    q = np.zeros(6)
    qdot = np.zeros(6)
    qdotdot = np.array([  1.13564582,  31.83833122, -42.20881005 ,  9.80303457 ,  1.81806351,
  10.21510774])

    #Dynamics terms
    M = inv_dyn.calculate_mass_matrix(q)
    C = inv_dyn.calculate_coriolis_vector(q, qdot)
    G = inv_dyn.calculate_gravity_vector(q)

    # the required joint torques
    tau = inv_dyn.calculate_torques(q, qdot, qdotdot)

    np.set_printoptions(precision=4, suppress=True)
    print("Dynamics Components Calculation Example")
    print("-" * 40)
    print('The Newton-Euler method')

    print("Mass Matrix M(q):")
    print(M)
    print("-" * 40)

    print("Coriolis Vector C(q, q_dot)q_dot:")
    print(C)
    print("-" * 40)

    print("Gravity Vector G(q):")
    print(G)
    print("-" * 40)

    print("Calculated Joint Torques/Forces (tau):")
    print(tau)