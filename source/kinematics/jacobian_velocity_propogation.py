"""
In this module I demonstrate how to derive the basic Jacobian via recursive velocity propagation.
The explicit analytical form is much more practical and that’s what I’ll use elsewhere in this project.
Here I’ve worked through the velocity–propagation approach simply to show that it can be done.
"""

import sympy as sym
import numpy as np
from sympy import pi
from sympy.utilities.lambdify import lambdify

q1, q2, q3, q4, q5, q6 = sym.symbols('q1 q2 q3 q4 q5 q6', real=True)
q1dot, q2dot, q3dot, q4dot, q5dot, q6dot = sym.symbols(
    'q1dot q2dot q3dot q4dot q5dot q6dot', real=True)

def T(a, alpha, d, theta):
    c_theta = sym.cos(theta); s_theta = sym.sin(theta)
    c_alpha = sym.cos(alpha); s_alpha = sym.sin(alpha)
    return sym.Matrix([
        [c_theta,      -s_theta,       0,   a],
        [s_theta*c_alpha,  c_theta*c_alpha,   -s_alpha, -s_alpha*d],
        [s_theta*s_alpha,  c_theta*s_alpha,    c_alpha,  c_alpha*d],
        [0,        0,       0,    1]
    ])

DH = [
    (0,      0,      0.163,    q1),
    (0,     -pi/2,   0.138,    q2),
    (0.425,   0,     -0.131,   q3),
    (0.392,   0,      0.127,   q4),
    (0,     -pi/2,   0.100,    q5),
    (0,      pi/2,   0.205+q6, 0  )
]

Ts = [T(*p) for p in DH]
Rs = [T[:3, :3] for T in Ts]
Ps = [T[:3, 3]   for T in Ts]

qd = sym.Matrix([q1dot, q2dot, q3dot, q4dot, q5dot, q6dot])
w = [sym.zeros(3,1)]
v = [sym.zeros(3,1)]
z = sym.Matrix([0, 0, 1])

for i in range(6):
    wi = Rs[i].T * w[i] + qd[i] * z
    extra = qd[i] * z if i == 5 else sym.zeros(3,1)  # prismatic at joint 6
    vi = Rs[i].T * (v[i] + w[i].cross(Ps[i]) + extra)
    w.append(wi)
    v.append(vi)

# End-effector velocities in base frame
v_ee = v[-1]
w_ee = w[-1]
T06 = sym.eye(4)
for T in Ts:
    T06 *= T
R06 = T06[:3, :3]
v_base = R06 * v_ee
w_base = R06 * w_ee

# Lambdify for numeric evaluation
all_syms = (q1, q2, q3, q4, q5, q6, q1dot, q2dot, q3dot, q4dot, q5dot, q6dot)
v_func = lambdify(all_syms, v_base, 'numpy')
w_func = lambdify(all_syms, w_base, 'numpy')

# Numeric inputs
q_vals    = [np.pi/2, 0, 0, 0, 0, 0.05]
qdot_vals = [1,      -1, 0, 0, 0, 0.01]

# Evaluate
v_num = np.array(v_func(*q_vals, *qdot_vals)).flatten()
w_num = np.array(w_func(*q_vals, *qdot_vals)).flatten()

print("Linear velocity :", v_num)
print("Angular velocity :", w_num)
