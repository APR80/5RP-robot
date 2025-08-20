import numpy as np
from params import DH, m, Ic, Pc
from kinematics.forward_kinematics import forward
from kinematics.com_jacobians import compute_com_jacobians

class Lagrangian:
    """
    Calculates the dynamic terms of a robot manipulator using the Lagrangian formulation.
    """
    def __init__(self, dh_params, masses, inertias, com_positions, gravity=np.array([0, 0, -9.81])):
        self.n = len(dh_params)
        self.m = masses
        self.Ic = inertias
        self.Pc = com_positions
        # The joint types are derived from the DH parameters(5th element).
        self.joint_types = [p[4] for p in dh_params]
        self.gravity = gravity

    def calculate_mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        M = Σ [ m_i * Jv_i^T * Jv_i + Jw_i^T * R_i * I_i * R_i^T * Jw_i ]
        """
        q = np.asarray(q, float)
        if q.size != self.n:
            raise ValueError(f"Expected {self.n} joints, got {q.size}")

        fk_result = forward(q)
        R_abs = fk_result.R_abs
        all_jacobians = compute_com_jacobians(q, self.Pc, self.joint_types)

        M = np.zeros((self.n, self.n))
        for i in range(self.n):
            m_i = self.m[i]
            I_local_i = self.Ic[i]
            J_i = all_jacobians[i]
            Jv_i = J_i[:3, :]
            Jw_i = J_i[3:, :]

            # Rotate inertia tensor from local frame to world frame
            I_world_i = R_abs[i] @ I_local_i @ R_abs[i].T
            M += m_i * (Jv_i.T @ Jv_i) + (Jw_i.T @ I_world_i @ Jw_i)
        return M

    def calculate_coriolis_vector(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """
        c_k = Σ_{i,j} Γ_kij * q̇_i * q̇_j
        """
        q = np.asarray(q, float)
        qdot = np.asarray(qdot, float)
        if q.size != self.n or qdot.size != self.n:
            raise ValueError(f"Expected {self.n} joints for q and qdot.")

        # finite differences to approximate the derivatives of the mass matrix
        h = 1e-6  # Perturbation
        M = self.calculate_mass_matrix(q)
        dM_dq = np.zeros((self.n, self.n, self.n))

        for i in range(self.n):
            q_perturbed = q.copy()
            q_perturbed[i] += h
            M_perturbed = self.calculate_mass_matrix(q_perturbed)
            dM_dq[i, :, :] = (M_perturbed - M) / h

        # Christoffel symbols of the first kind
        christoffel = np.zeros((self.n, self.n, self.n))
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    # Γ_kij = 0.5 * (∂M_kj/∂q_i + ∂M_ki/∂q_j - ∂M_ij/∂q_k)
                    dMkj_dqi = dM_dq[i, k, j]
                    dMki_dqj = dM_dq[j, k, i]
                    dMij_dqk = dM_dq[k, i, j]
                    christoffel[k, i, j] = 0.5 * (dMkj_dqi + dMki_dqj - dMij_dqk)

        c = np.zeros(self.n)
        for k in range(self.n):
            c[k] = qdot.T @ christoffel[k, :, :] @ qdot

        return c

    def calculate_gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """
        G = ∂U/∂q = - Σ [ m_i * Jv_i^T * g ]
        """
        q = np.asarray(q, float)
        if q.size != self.n:
            raise ValueError(f"Expected {self.n} joints, got {q.size}")

        all_jacobians = compute_com_jacobians(q, self.Pc, self.joint_types)

        G = np.zeros(self.n)
        for i in range(self.n):
            m_i = self.m[i]
            Jv_i = all_jacobians[i][:3, :]
            G -= m_i * Jv_i.T @ self.gravity
        return G


if __name__ == '__main__':
    #example usage
    dynamics = Lagrangian(dh_params=DH, masses=m, inertias=Ic, com_positions=Pc)

    q = np.zeros(6)
    qdot = np.zeros(6)
    qdotdot = np.array([1.13564582, 31.83833122, -42.20881005, 9.80303457, 1.81806351,
                        10.21510774])

    #Dynamics terms
    M = dynamics.calculate_mass_matrix(q)
    C = dynamics.calculate_coriolis_vector(q, qdot)
    G = dynamics.calculate_gravity_vector(q)

    # the required joint torques
    tau = M @ qdotdot + C + G

    np.set_printoptions(precision=4, suppress=True)
    print("Dynamics Components Calculation Example")
    print("-" * 40)
    print('The Lagrangian method')

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