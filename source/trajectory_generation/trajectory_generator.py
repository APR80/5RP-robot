import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


class TrajectoryGeneration:
    """
    A collection of classes for joint-space trajectory generation.
    it acts as a namespace for different trajectory generation methods described in the book.
    """

    class cubic_polynomial:

        def __init__(self, q0: np.ndarray, qf: np.ndarray, dq0: np.ndarray, dqf: np.ndarray, tf: float):
            self.tf = tf
            q0 = np.asarray(q0)
            qf = np.asarray(qf)
            dq0 = np.asarray(dq0)
            dqf = np.asarray(dqf)

            # Coefficients
            a0 = q0
            a1 = dq0
            a2 = (3 * (qf - q0) / tf ** 2) - (2 * dq0 + dqf) / tf
            a3 = (-2 * (qf - q0) / tf ** 3) + (dq0 + dqf) / tf ** 2

            self.coefficients = np.vstack([a0, a1, a2, a3])

        def sample(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Samples at given time points.
            """
            t = np.clip(t, 0, self.tf)

            # Time basis vectors
            T = np.vstack([np.ones_like(t), t, t ** 2, t ** 3]).T
            Td = np.vstack([np.zeros_like(t), np.ones_like(t), 2 * t, 3 * t ** 2]).T
            Tdd = np.vstack([np.zeros_like(t), np.zeros_like(t), 2 * np.ones_like(t), 6 * t]).T

            # position, velocity, and acceleration
            q = T @ self.coefficients
            dq = Td @ self.coefficients
            ddq = Tdd @ self.coefficients
            return q, dq, ddq

    class quintic_polynomial:

        def __init__(self, q0: np.ndarray, qf: np.ndarray, dq0: np.ndarray, dqf: np.ndarray, ddq0: np.ndarray,
                     ddqf: np.ndarray, tf: float):
            self.tf = tf

            # System of equations matrix for quintic polynomial constraints
            M = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [1, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
                [0, 1, 2 * tf, 3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
                [0, 0, 2, 6 * tf, 12 * tf ** 2, 20 * tf ** 3]
            ])

            # Boundary conditions vector
            B = np.vstack([q0, dq0, ddq0, qf, dqf, ddqf])

            # Solve for coefficients(A = M^-1 * B)
            self.coefficients = np.linalg.solve(M, B)

        def sample(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            t = np.clip(t, 0, self.tf)

            T = np.vstack([np.ones_like(t), t, t ** 2, t ** 3, t ** 4, t ** 5]).T
            Td = np.vstack([np.zeros_like(t), np.ones_like(t), 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4]).T
            Tdd = np.vstack(
                [np.zeros_like(t), np.zeros_like(t), 2 * np.ones_like(t), 6 * t, 12 * t ** 2, 20 * t ** 3]).T

            q = T @ self.coefficients
            dq = Td @ self.coefficients
            ddq = Tdd @ self.coefficients
            return q, dq, ddq

    class LSPB:
        """
        Generates a Linear Segment with Parabolic Blends trajectory.
        """

        def __init__(self, q0: np.ndarray, qf: np.ndarray, ddqb: np.ndarray, tf: float):

            self.tf = tf
            self.q0 = np.asarray(q0)
            self.qf = np.asarray(qf)
            ddqb = np.asarray(ddqb)

            if not (self.q0.shape == self.qf.shape and self.q0.shape == ddqb.shape):
                raise ValueError("Inputs 'q0', 'qf', and 'ddqb' must have the same shape.")
            if tf <= 0:
                raise ValueError("Total time 'tf' must be positive.")
            if np.any(ddqb <= 0):
                raise ValueError("Blend acceleration 'ddqb' must be positive for all joints.")

            delta_q = self.qf - self.q0

            # Handle joints that do not move
            self.static_mask = np.isclose(delta_q, 0)

            # Need signed acceleration
            a_blend = np.sign(delta_q) * ddqb

            # Feasibility Check
            min_accel = (4 * np.abs(delta_q)) / tf ** 2
            if np.any(ddqb < min_accel):
                raise ValueError(
                    f"ddqb is too small for one or more joints. "
                    f"It must be >= {min_accel} for the given displacement and time."
                )

            # Blend time 'tb' and constant velocity
            sqrt_term = np.sqrt(tf ** 2 * a_blend ** 2 - 4 * delta_q * a_blend)
            self.tb = (tf * a_blend - sqrt_term) / (2 * a_blend)
            self.v_const = delta_q / (tf - self.tb)
            self.a_blend = a_blend

        def sample(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

            t = np.clip(np.asarray(t), 0, self.tf)
            # Reshape t to column vector for broadcasting with joint arrays
            t_col = t.reshape(-1, 1)

            # Output arrays
            q = np.zeros((t_col.shape[0], self.q0.shape[0]))
            dq = np.zeros_like(q)
            ddq = np.zeros_like(q)

            # --- masks for the three phases ---
            # Phase 1: First parabolic blend (acceleration)
            mask1 = t_col <= self.tb
            # Phase 2: Linear segment (constant velocity)
            mask2 = (t_col > self.tb) & (t_col <= self.tf - self.tb)
            # Phase 3: Second parabolic blend (deceleration)
            mask3 = t_col > self.tf - self.tb

            # --- Calculations for Each Phase ---
            # Phase 1
            q[mask1] = (self.q0 + 0.5 * self.a_blend * t_col ** 2)[mask1]
            dq[mask1] = (self.a_blend * t_col)[mask1]
            ddq[mask1] = np.broadcast_to(self.a_blend, q.shape)[mask1]

            # Phase 2
            q[mask2] = (self.q0 + self.v_const * (t_col - 0.5 * self.tb))[mask2]
            dq[mask2] = np.broadcast_to(self.v_const, q.shape)[mask2]
            # ddq is 0 for this phase

            # Phase 3
            q[mask3] = (self.qf - 0.5 * self.a_blend * (self.tf - t_col) ** 2)[mask3]
            dq[mask3] = (self.a_blend * (self.tf - t_col))[mask3]
            ddq[mask3] = np.broadcast_to(-self.a_blend, q.shape)[mask3]

            # --- static joint conditions ---
            if np.any(self.static_mask):
                q[:, self.static_mask] = self.q0[self.static_mask]
                dq[:, self.static_mask] = 0
                ddq[:, self.static_mask] = 0

            return q, dq, ddq


def plot_trajectory(t, q, dq, ddq, title):
    """
    Helper plotting function
    """
    n_joints = q.shape[1]
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(title, fontsize=16)

    # Plot Position
    axs[0].plot(t, q)
    axs[0].set_ylabel('Position (rad)')
    axs[0].grid(True)
    axs[0].legend([f'Joint {i + 1}' for i in range(n_joints)], loc='upper right')
    axs[0].set_title('Joint Positions')

    # Plot Velocity
    axs[1].plot(t, dq)
    axs[1].set_ylabel('Velocity (rad/s)')
    axs[1].grid(True)
    axs[1].set_title('Joint Velocities')

    # Plot Acceleration
    axs[2].plot(t, ddq)
    axs[2].set_ylabel('Acceleration (rad/s²)')
    axs[2].set_xlabel('Time (s)')
    axs[2].grid(True)
    axs[2].set_title('Joint Accelerations')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == '__main__':
    # Example usage

    q0 = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    qf = np.array([1.0, 0.0, 2.5, 2.0, -1.0, 0.05])
    dq0 = np.zeros(6)
    dqf = np.zeros(6)
    ddq0 = np.zeros(6)
    ddqf = np.zeros(6)
    ddqb = np.ones(6)  # blend acceleration for LSPB
    tf = 5.0

    # time vector for sampling
    t = np.linspace(0, tf, 500)  # 500 points for a smooth plot

    print("Generating Cubic Polynomial Trajectory...")
    cubic_traj = TrajectoryGeneration.cubic_polynomial(q0, qf, dq0, dqf, tf)
    q_cubic, dq_cubic, ddq_cubic = cubic_traj.sample(t)
    plot_trajectory(t, q_cubic, dq_cubic, ddq_cubic, 'Cubic Polynomial Trajectory')

    print("Generating Quintic Polynomial Trajectory...")
    quintic_traj = TrajectoryGeneration.quintic_polynomial(q0, qf, dq0, dqf, ddq0, ddqf, tf)
    q_quintic, dq_quintic, ddq_quintic = quintic_traj.sample(t)
    plot_trajectory(t, q_quintic, dq_quintic, ddq_quintic, 'Quintic Polynomial Trajectory')

    print("Generating LSPB Trajectory...")
    try:
        lspb_traj = TrajectoryGeneration.LSPB(q0, qf, ddqb, tf)
        q_lspb, dq_lspb, ddq_lspb = lspb_traj.sample(t)
        plot_trajectory(t, q_lspb, dq_lspb, ddq_lspb, 'LSPB Trajectory')
    except ValueError as e:
        print(f"Could not generate LSPB trajectory: {e}")

