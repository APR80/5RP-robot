import mujoco as mj
import numpy as np
import matplotlib.pyplot as plt
from renderer import MujocoRenderer
from trajectory_generation.trajectory_generator import TrajectoryGeneration
from dynamics.lagrangian_method import Lagrangian
from dynamics.newton_euler_method import NewtonEuler
from params import DH, m, Ic, Pc

class InverseDynamics:
    def __init__(self, ref_generator, Kp, Kv):
        self.ref_generator = ref_generator
        self.Kp = Kp
        self.Kv = Kv
        # self.dynamics = NewtonEuler(dh_params=DH, masses=m, inertias=Ic, com_positions=Pc)
        # self.dynamics = Lagrangian(dh_params=DH, masses=m, inertias=Ic, com_positions=Pc)

    # mj.set_mjcb_control expects a callable function with arguments model and data.
    def __call__(self, model, data):
        # Sample desired trajectory
        qd, dqd, ddqd = self.ref_generator.sample(data.time)

        # Errors
        e = qd - data.qpos[:model.nu]
        de = dqd - data.qvel[:model.nu]

        #Compute dynamic terms
        #using newton_euler or lagrangian
        # alpha = self.dynamics.calculate_mass_matrix(data.qpos)
        # beta = self.dynamics.calculate_coriolis_vector(data.qpos, data.qvel)+\
        # self.dynamics.calculate_gravity_vector(data.qpos)

        #Using mujoco
        M = np.zeros((model.nu, model.nu))
        mj.mj_fullM(model, M, data.qM)
        qfrc_bias = np.asarray(data.qfrc_bias[:model.nu])[:, None]
        alpha = M
        beta = qfrc_bias

        #Control torque
        tau_prime = np.squeeze(ddqd + self.Kv * de + self.Kp * e)
        tau = alpha @ tau_prime + beta.squeeze()
        data.ctrl = tau



        # Store for logging
        self.current_tau = tau.copy()

def main():
    xml_path = '../../models/direct_drive.xml'
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    q0 = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.])
    qf = np.zeros(6)
    dq0 = np.zeros(6)
    dqf = np.zeros(6)
    tf = 5
    traj = TrajectoryGeneration.cubic_polynomial(q0, qf, dq0, dqf, tf)

    simend = 10
    renderer = MujocoRenderer(model, data)

    Kp = np.array([400, 400, 300, 200, 100, 50])
    Kd = np.array([40, 40, 30, 20, 10, 5])
    controller = InverseDynamics(traj, Kp, Kd)
    mj.set_mjcb_control(controller)

    # Logging arrays
    time_history = []
    q_actual_history = []
    q_desired_history = []
    torque_history = []

    # Set initial joint positions
    data.qpos = q0.copy()

    # Simulation loop
    while not renderer.is_window_closed() and data.time < simend:
        t_prev = data.time
        # Step until next render frame
        while data.time - t_prev < 1.0 / 60.0:
            mj.mj_step(model, data)

            # Log
            if data.time <= tf:
                time_history.append(data.time)
                q_actual_history.append(data.qpos[:model.nu].copy())
                qd, _, _ = traj.sample(data.time)
                q_desired_history.append(qd)
                torque_history.append(controller.current_tau)

        renderer.render()

    renderer.close()

    # Convert logs to arrays
    time_history = np.array(time_history)
    q_actual_history = np.array(q_actual_history)
    q_desired_history = np.array(q_desired_history).squeeze()
    torque_history = np.vstack(torque_history)

    if time_history.size == 0:
        print("No data recorded. Check controller gains or simulation settings.")
        return

    # Plot joint angles
    fig1, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    axs = axs.flatten()
    for i in range(model.nu):
        axs[i].plot(time_history, q_actual_history[:, i], label=f'Actual q{i + 1}')
        axs[i].plot(time_history, q_desired_history[:, i], '--', label=f'Desired q{i + 1}')
        axs[i].set_ylabel('Angle (rad)')
        axs[i].set_title(f'Joint {i + 1}')
        axs[i].legend()
        axs[i].grid(True)
    for ax in axs[-2:]:
        ax.set_xlabel('Time (s)')
    axs[5].set_ylim(-1, 1)
    fig1.suptitle('Joint Tracking: Desired vs Actual')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig('linear_joint_plot.png')

    # Plot torques
    fig2, axs2 = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    axs2 = axs2.flatten()
    for i in range(model.nu):
        axs2[i].plot(time_history, torque_history[:, i], label=f'Torque τ{i + 1}')
        axs2[i].set_ylabel('Torque (Nm)')
        axs2[i].set_title(f'Joint {i + 1} Torque')
        axs2[i].legend()
        axs2[i].grid(True)
    for ax in axs2[-2:]:
        ax.set_xlabel('Time (s)')
    fig2.suptitle('Linear Controller Torques Over Time')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig('linear_torque_plot.png')

    plt.show()


if __name__ == '__main__':
    main()