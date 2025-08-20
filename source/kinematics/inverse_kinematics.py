"""
Implementation of inverse kinematics of the 5RP robot.
I used a combination of geometric and algebraic methods in that I calculated q6(the length of the prismatic joint)
using the robot’s geometry, while the remaining variables were solved algebraically.
"""

import numpy as np
from kinematics.forward_kinematics import forward
pi = np.pi

def inverse_kinematics(T06):
    pi = np.pi
    solutions = []

    c234 = -T06[2, 1]
    denominator = 1 - c234**2
    if np.isclose(denominator, 0):
        raise ValueError("Wrist singularity (c234 is +/-1).")

    c2_5 = (T06[2, 0]**2 - T06[2, 2]**2) / denominator
    s2_5 = (2 * T06[2, 0] * T06[2, 2]) / denominator
    angle_2q5 = np.arctan2(s2_5, c2_5)
    q5_options = [angle_2q5 / 2, (angle_2q5 / 2) + pi]

    for q5 in q5_options:
        s5 = np.sin(q5); c5 = np.cos(q5)
        if np.isclose(c5, 0): continue

        s234 = -T06[2, 0] / c5
        if np.isclose(s234, 0): continue

        s1 = -T06[1, 1] / s234
        c1 = -T06[0, 1] / s234
        q1 = np.arctan2(s1, c1)

        T16_13 = -s1 * T06[0, 3] + c1 * T06[1, 3]
        q6 = ((T16_13 - 0.134) / c5) - 0.205

        # Enforce q6 joint limit
        # If q6 is outside the desired range, this solution branch is invalid.
        # We skip it and move to the next possibility.
        if not (0 <= q6 <= 0.075):
            continue

        q234 = np.arctan2(s234, c234)

        A = -0.205*s1*c5 - 0.134*s1 + 0.205*s5*c1*c234 - 0.1*s234*c1 + q6*(s5*c1*c234 - s1*c5)
        B =  0.205*s1*s5*c234 - 0.1*s1*s234 + 0.205*c1*c5 + 0.134*c1 + q6*(s1*s5*c234 + c1*c5)
        C = -q6*s5*s234 - 0.205*s5*s234 - 0.1*c234 + 0.163

        gamma = c1*T06[0, 3] + s1*T06[1, 3] - A*c1 - B*s1
        z = C - T06[2, 3]

        c3_val = (gamma**2 + z**2 - 0.425**2 - 0.392**2) / (2 * 0.425 * 0.392)
        if abs(c3_val) > 1: continue

        c3 = np.clip(c3_val, -1.0, 1.0)
        s3_options = [np.sqrt(1 - c3**2), -np.sqrt(1 - c3**2)]

        for s3 in s3_options:
            q3 = np.arctan2(s3, c3)
            q2 = np.arctan2(z, gamma) - np.arctan2(0.392 * s3, 0.425 + 0.392 * c3)
            q4 = q234 - q2 - q3
            solutions.append(np.array([q1, q2, q3, q4, q5, q6]))

    if not solutions:
        raise ValueError("No solution found that satisfies the q6 joint limit.")

    solutions.sort(key=lambda q: np.linalg.norm(q))
    return solutions

if __name__ == '__main__':
    # Verification

    q_test = np.array([-3.1416, -1.4248,  1.5 ,   -0.5752,  4.2832 , 0.05  ] )
    fk = forward(q_test)
    T06_target = fk.Ts_abs[-1]
    print("--- Target T06 Matrix ---\n", T06_target, "\n")

    sols = inverse_kinematics(T06_target)

    print("--- Solutions Sorted by Least Rotation (with 0 <= q6 <= 0.075) ---")
    for i, q_sol in enumerate(sols):
        norm = np.linalg.norm(q_sol)
        print(f"Solution {i+1}: {np.round(q_sol, 4)} (Norm: {norm:.4f})")

    T06_sol1 = forward(sols[0]).Ts_abs[-1]
    print("\nDifference T06_target - T06_recon:")
    print(T06_target - T06_sol1)

    T06_sol2 = forward(sols[1]).Ts_abs[-1]
    print("\nDifference T06_target - T06_recon:")
    print(T06_target - T06_sol2)



