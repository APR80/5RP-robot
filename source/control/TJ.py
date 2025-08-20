"""
This is an implementation of the transpose-Jacobian algorithm from Introduction to Robotics by Craig.
"""

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

class TJ_controller:
    def __init__(self, Kp, Kv, history, traj_func=get_reference_trajectory):
        self.Kp = Kp
        self.Kv = Kv
        self.history = history  # record data
        self.traj_func = traj_func

    def __call__(self, model, data):

        xd, dxd = self.traj_func(data.time)

        x_actual = data.sensor('eepos').data.copy()
        v_actual = data.sensor('linvel').data.copy()

        # Log end-effector pos
        self.history['time'].append(data.time)
        self.history['ref_x'].append(xd[0]); self.history['act_x'].append(x_actual[0])
        self.history['ref_y'].append(xd[1]); self.history['act_y'].append(x_actual[1])
        self.history['ref_z'].append(xd[2]); self.history['act_z'].append(x_actual[2])

        e   = xd - x_actual
        de  = dxd - v_actual
        JT  = jacobian(data.qpos)[:3, :].T

        # tau = J^T * (Kv * de + Kp * e) - qvel + qfrc_bias
        tau = JT @ (self.Kv @ de + self.Kp @ e) - data.qvel + data.qfrc_bias

        # apply control
        data.ctrl = tau

        # for logging
        self.history['torque'].append(tau.copy())

def main():
    xml_path = '../../models/direct_drive.xml'
    model    = mj.MjModel.from_xml_path(xml_path)
    data     = mj.MjData(model)

    q0      = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    simend  = 30.0  # seconds

    # Data history
    history = {
        'time': [],
        'ref_x': [], 'act_x': [],
        'ref_y': [], 'act_y': [],
        'ref_z': [], 'act_z': [],
        'torque': []  # will store 6-d vector per timestep
    }

    # Renderer & controller
    renderer = MujocoRenderer(model, data)
    Kp        = np.diag([150.0]*3)
    Kv = np.diag([40.0] * 3)
    controller = TJ_controller(Kp, Kv, history, traj_func=get_reference_trajectory)
    mj.set_mjcb_control(controller)

    data.qpos = q0.copy()

    # Simulation loop
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

    torque_history = np.vstack(history['torque'])

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
    plt.title('Reference vs Actual End-Effector Position')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig('TJ_tracking.png')

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

    fig.suptitle('TJ Control Torques Over Time')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig('TJ_torques.png')
    plt.show()

if __name__ == '__main__':
    main()
