import numpy as np
import matplotlib.pyplot as plt
from dynamics.lagrangian_method import Lagrangian
from params import DH, m, Ic, Pc

class TimeOptimal:
    def __init__(self, q_start, q_goal, T_min, T_max):
        self.q_start = q_start
        self.q_goal = q_goal
        self.T_min = T_min
        self.T_max = T_max
        self.eps = 1e-6
        self.dynamics = Lagrangian(dh_params=DH, masses=m, inertias=Ic, com_positions=Pc)
        self.num_joints = q_start.shape[0]

    def _path_stuff(self, s):
        """I use a linear path in joint space."""
        qs = (1 - s) * self.q_start + s * self.q_goal
        qs_prime = self.q_goal - self.q_start
        qs_double_prime = np.zeros_like(self.q_start)
        return qs, qs_prime, qs_double_prime

    def path_dynamics(self, s, s_dot):
        """Calculate the dynamics terms in terms of s"""
        qs, qs_prime, _ = self._path_stuff(s)

        M = self.dynamics.calculate_mass_matrix(qs)
        q_dot_for_C = qs_prime * s_dot
        C_vector = self.dynamics.calculate_coriolis_vector(qs, q_dot_for_C)
        G_vector = self.dynamics.calculate_gravity_vector(qs)

        a = M @ qs_prime
        b = C_vector + G_vector  # For a linear path, the M*q'' term is zero

        return a, b

    def get_s_dotdot(self, s, s_dot):
        """Calculates the maximum and minimum possible path acceleration s̈."""
        a, b = self.path_dynamics(s, s_dot)

        with np.errstate(divide='ignore', invalid='ignore'):
            lb = np.where(a > 0, (self.T_min - b) / (a + self.eps), (self.T_max - b) / (a - self.eps))
            ub = np.where(a > 0, (self.T_max - b) / (a + self.eps), (self.T_min - b) / (a - self.eps))

        s_ddot_min = np.max(lb)
        s_ddot_max = np.min(ub)

        return s_ddot_min, s_ddot_max

    def generate_trajectory(self, num_points=100):
        """
        Generates the time-optimal trajectory and calculates the required torques.
        """
        s_path = np.linspace(0, 1, num_points)
        ds = s_path[1] - s_path[0]

        # Forward Pass
        s_dot_fwd = np.zeros(num_points)
        for i in range(num_points - 1):
            s = s_path[i]
            s_dot = s_dot_fwd[i]
            _, s_ddot_max = self.get_s_dotdot(s, s_dot)
            s_dot_sq_next = s_dot ** 2 + 2 * s_ddot_max * ds
            s_dot_fwd[i + 1] = np.sqrt(s_dot_sq_next) if s_dot_sq_next > 0 else 0

        # Backward Pass
        s_dot_bwd = np.zeros(num_points)
        for i in range(num_points - 1, 0, -1):
            s = s_path[i]
            s_dot = s_dot_bwd[i]
            s_ddot_min, _ = self.get_s_dotdot(s, s_dot)
            s_dot_sq_prev = s_dot ** 2 - 2 * s_ddot_min * ds
            s_dot_bwd[i - 1] = np.sqrt(s_dot_sq_prev) if s_dot_sq_prev > 0 else 0

        # Combine to get the optimal velocity profile
        s_dot_optimal = np.minimum(s_dot_fwd, s_dot_bwd)

        # timing and acceleration
        time = np.zeros(num_points)
        s_ddot_optimal = np.zeros(num_points)
        for i in range(num_points - 1):
            s_dot_i = s_dot_optimal[i]
            s_dot_i_plus_1 = s_dot_optimal[i + 1]
            dt = (2 * ds) / (s_dot_i + s_dot_i_plus_1 + self.eps)
            time[i + 1] = time[i] + dt
            s_ddot_optimal[i] = (s_dot_i_plus_1 - s_dot_i) / (dt + self.eps)
        s_ddot_optimal[-1] = s_ddot_optimal[-2]

        # Calculate the optimal torques
        torques_optimal = np.zeros((num_points, self.num_joints))
        for i in range(num_points):
            s = s_path[i]
            s_dot = s_dot_optimal[i]
            s_ddot = s_ddot_optimal[i]

            a, b = self.path_dynamics(s, s_dot)
            torques_optimal[i, :] = a * s_ddot + b

        # Results
        traj = {
            "time": time,
            "s_path": s_path,
            "s_dot_optimal": s_dot_optimal,
            "s_ddot_optimal": s_ddot_optimal,
            "torques_optimal": torques_optimal,
            "s_dot_fwd_limit": s_dot_fwd,
            "s_dot_bwd_limit": s_dot_bwd,
        }
        return traj


def plot_trajectory_results(trajectory, T_min, T_max):
    """
    Helper plotting function
    """
    time = trajectory['time']
    torques = trajectory['torques_optimal']
    num_joints = torques.shape[1]

    print(f"Optimal trajectory time: {time[-1]:.4f} seconds")

    # Phase Plot
    plt.figure(figsize=(10, 6))
    plt.plot(trajectory['s_path'], trajectory['s_dot_fwd_limit'], 'g--', label='Forward Integration Limit')
    plt.plot(trajectory['s_path'], trajectory['s_dot_bwd_limit'], 'r--', label='Backward Integration Limit')
    plt.plot(trajectory['s_path'], trajectory['s_dot_optimal'], 'b-', linewidth=2, label='Optimal Velocity Profile')
    plt.title('Phase Plot (Path Velocity vs. Path Position)')
    plt.xlabel('Path Position $s$')
    plt.ylabel('Path Velocity $\dot{s}$ (rad/s)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Time-based Trajectories (s, s_dot, s_ddot vs. Time)
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Time-Optimal Path Trajectories')

    axs[0].plot(time, trajectory['s_path'], 'k-')
    axs[0].set_ylabel('Position $s(t)$')
    axs[0].grid(True)

    axs[1].plot(time, trajectory['s_dot_optimal'], 'b-')
    axs[1].set_ylabel('Velocity $\dot{s}(t)$')
    axs[1].grid(True)

    axs[2].plot(time, trajectory['s_ddot_optimal'], 'r-')
    axs[2].set_ylabel('Acceleration $\ddot{s}(t)$')
    axs[2].set_xlabel('Time (s)')
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for suptitle
    plt.show()

    # Optimal Torques
    plt.figure(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, num_joints))

    for i in range(num_joints):
        plt.plot(time, torques[:, i], label=f'Joint {i + 1} Torque', color=colors[i])

    # Plot torque limits for reference (here they are the same for all joints)
    plt.plot(time, np.full_like(time, T_max[0]), 'k--', label='Max Torque Limit')
    plt.plot(time, np.full_like(time, T_min[0]), 'k--', label='Min Torque Limit')

    plt.title('Optimal Joint Torques vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    q0 = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.])
    qf = np.zeros(6)

    # Joint torque limits
    T_min = np.full(6, -150.0)
    T_max = np.full(6, 200.0)

    # Instantiate and generate the trajectory
    time_optim = TimeOptimal(q0, qf, T_min, T_max)
    trajectory_data = time_optim.generate_trajectory(num_points=200)
    plot_trajectory_results(trajectory_data, T_min, T_max)
