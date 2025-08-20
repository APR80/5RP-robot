import mujoco as mj
import numpy as np
import matplotlib.pyplot as plt
from renderer import MujocoRenderer
from kinematics.jacobian_explicit import jacobian

def get_reference_trajectory(time):
    """
    Gives the desired end-effector position (xd) and velocity (dxd) for a given time for a
    circular trajectory in the X-Y plane (offset in Y).
    """
    xd = np.array([
        0.1 * np.cos(time),
        -0.7 + 0.1 * np.sin(time),
        0.4
    ])
    dxd = np.array([
        -0.1 * np.sin(time),
        0.1 * np.cos(time),
        0.0
    ])
    return xd, dxd

class MTJ_Controller:
    def __init__(self, Kp, Kv, Kd, emax, demax, nu, history, traj_func=get_reference_trajectory):
        self.Kp    = Kp
        self.Kv    = Kv
        self.Kd    = Kd
        self.emax  = np.array(emax)
        self.demax = np.array(demax)
        # Memory of previous "ideal" task wrench Q̂ = Kv·de + Kp·e
        self.prev_Qhat = np.zeros(3)
        # Storage of nu, for convenience
        self.nu = nu
        # History dict for logging end-effector and torque data
        self.history = history
        # Trajectory function
        self.traj_func = traj_func


    # mj.set_mjcb_control expects a callable function with arguments model and data.
    def __call__(self, model, data):
        t = data.time
        xd, dxd = self.traj_func(t)

        x_act = data.sensor('eepos').data.copy()
        v_act = data.sensor('linvel').data.copy()

        # Task-space errors
        e  = xd  - x_act
        de = dxd - v_act

        # Exponential weight vector k(t)
        # k_i = exp( - (|e_i|/emax_i + |de_i|/demax_i) )

        k_vec = np.exp(- (np.abs(e)/self.emax + np.abs(de)/self.demax))
        K_mod = np.diag(k_vec)   # 3×3

        # ideal PD task-wrench Q̂(t)
        Qhat = self.Kv @ de + self.Kp @ e

        # Memory term: h(t) = K_mod · Q̂(t - Δt)
        h = K_mod @ self.prev_Qhat

        aug_wrench = Qhat + h

        # Map back to joint torques
        J_full = jacobian(data.qpos)       # expected shape (3, nu)
        tau_task  = J_full[:3,:].T @ aug_wrench

        # joint-space damping
        tau_damping = - self.Kd @ data.qvel

        # Gravity compensation / bias forces
        tau_grav = data.qfrc_bias[:self.nu]

        cmd = tau_task + tau_damping + tau_grav
        data.ctrl[:self.nu] = cmd

        # save Q̂ for the next time step
        self.prev_Qhat = Qhat.copy()

        # Logging
        self.history['time'].append(t)
        self.history['ref_x'].append(xd[0]); self.history['act_x'].append(x_act[0])
        self.history['ref_y'].append(xd[1]); self.history['act_y'].append(x_act[1])
        self.history['ref_z'].append(xd[2]); self.history['act_z'].append(x_act[2])
        self.history['torque'].append(cmd.copy())


def main():
    xml_path = '../../models/direct_drive.xml'
    model = mj.MjModel.from_xml_path(xml_path)
    data  = mj.MjData(model)

    q0 = np.array([-1.5708, -1.5708,  1.5708,
                   -1.5708, -1.5708,  0.0])

    history = {
        'time': [],
        'ref_x': [], 'act_x': [],
        'ref_y': [], 'act_y': [],
        'ref_z': [], 'act_z': [],
        'torque': []  # will store nu-vector per timestep
    }

    renderer = MujocoRenderer(model, data)

    # Controller gains
    Kp   = np.diag([100.0, 100.0, 100.0])   # task-space P gains
    Kv   = np.diag([ 20.0,  20.0,  20.0])   # task-space D gains
    Kd   = np.diag([  5.0] * model.nu)      # joint damping
    emax  = [0.2, 0.2, 0.2]                 # max pos error before injection fades
    demax = [0.2, 0.2, 0.2]                 # max vel error

    controller = MTJ_Controller(Kp, Kv, Kd, emax, demax, model.nu, history, traj_func=get_reference_trajectory)
    mj.set_mjcb_control(controller)

    # initial joint positions
    data.qpos[:] = q0.copy()

    # Simulation loop
    simend = 20.0
    while not renderer.is_window_closed() and data.time < simend:
        t_prev = data.time
        while data.time - t_prev < 1.0/60.0:
            mj.mj_step(model, data)
        renderer.render()

    renderer.close()

    t = np.array(history['time'])
    ref_x = np.array(history['ref_x'])
    act_x = np.array(history['act_x'])
    ref_y = np.array(history['ref_y'])
    act_y = np.array(history['act_y'])
    ref_z = np.array(history['ref_z'])
    act_z = np.array(history['act_z'])

    torque_history = np.vstack(history['torque'])  # shape (n_steps, 6)

    # Plot end-effector tracking
    plt.figure(figsize=(8,5))
    plt.plot(t, ref_x, label='ref x')
    plt.plot(t, act_x, '--', label='act x')
    plt.plot(t, ref_y, label='ref y')
    plt.plot(t, act_y, '--', label='act y')
    plt.plot(t, ref_z, label='ref z')
    plt.plot(t, act_z, '--', label='act z')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('Reference vs Actual End-Effector Position (MTJ)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig('MTJ_tracking.png')

    # Plot joint torques
    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    axs = axs.flatten()
    for i in range(model.nu):
        axs[i].plot(t, torque_history[:, i], label=f'Torque τ{i+1}')
        axs[i].set_ylabel('Torque (Nm)')
        axs[i].set_title(f'Joint {i+1} Torque')
        axs[i].legend()
        axs[i].grid(True)

    for ax in axs[-2:]:
        ax.set_xlabel('Time [s]')

    fig.suptitle('MTJ Control Torques Over Time')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig('MTJ_torques.png')
    plt.show()


if __name__ == "__main__":
    main()
