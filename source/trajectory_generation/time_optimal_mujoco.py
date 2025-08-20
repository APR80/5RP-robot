"""
Here I just simulate the time optimal trajectory using mujoco
"""

import mujoco as mj
import numpy as np
from renderer import MujocoRenderer
from time_optimal_time_scaling import TimeOptimal

def main():

    xml_path = '../../models/highly_geared.xml'
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    simend = 5
    renderer = MujocoRenderer(model, data)

    # Start and Goal
    q0 = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.])
    qf = np.zeros(6)
    T_min = np.ones(6) * -80
    T_max = np.ones(6) * 80

    # Pre-calculate the trajectory(it's the computationally expensive part)
    print("Generating time-optimal trajectory...")
    timeoptim = TimeOptimal(q0, qf, T_min, T_max)
    trajectory = timeoptim.generate_trajectory(num_points=200)
    print("Trajectory generation complete.")

    # Extract the results
    time_points = trajectory["time"]
    s_path = trajectory["s_path"]

    # Set initial position
    data.qpos[:] = q0

    # Simulation
    while not renderer.is_window_closed() and data.time < simend:
        time_prev = data.time

        while data.time - time_prev < 1.0 / 60.0:

            # Interpolate to find the desired path parameter(s) at the current time
            s_current = np.interp(data.time, time_points, s_path)

            # Calculate the desired joint positions 'q' based on 's'
            q_desired = (1 - s_current) * q0 + s_current * qf

            # here we set the robot's position directly
            data.qpos[:] = q_desired

            mj.mj_step(model, data)

        # Render the scene
        renderer.render()

    # Cleanup
    renderer.close()

if __name__ == "__main__":
    main()
