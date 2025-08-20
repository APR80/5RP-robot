import numpy as np
from dataclasses import dataclass
from typing import List
from params import DH

def T(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [   ct,    -st,     0,      a    ],
        [ st*ca,  ct*ca,  -sa,  -sa*d   ],
        [ st*sa,  ct*sa,   ca,   ca*d   ],
        [    0,      0,     0,      1    ],
    ])

@dataclass
class FKResult:
    Ts_rel: List[np.ndarray]  # [T01, T12, …, T5→6]
    Ts_abs: List[np.ndarray]  # [T0→1, T0→2, …, T0→6]

    @property
    def R_rel(self) -> List[np.ndarray]:
        return [T[:3, :3] for T in self.Ts_rel]

    @property
    def P_rel(self) -> List[np.ndarray]:
        return [T[:3,  3] for T in self.Ts_rel]

    @property
    def R_abs(self) -> List[np.ndarray]:
        return [T[:3, :3] for T in self.Ts_abs]

    @property
    def P_abs(self) -> List[np.ndarray]:
        return [T[:3,  3] for T in self.Ts_abs]


def forward(q) -> FKResult:
    # build relative transforms
    Ts_rel = []
    for i, (a, alpha, d_off, theta_off, joint_type) in enumerate(DH):
        if joint_type == 'R':
            theta = q[i] + theta_off
            d     = d_off
        else:
            theta = theta_off
            d     = d_off + q[i]
        Ts_rel.append(T(a, alpha, d, theta))

    # accumulate into absolute
    Ts_abs = []
    T_curr = np.eye(4)
    for T_rel in Ts_rel:
        T_curr = T_curr @ T_rel
        Ts_abs.append(T_curr)

    return FKResult(Ts_rel=Ts_rel, Ts_abs=Ts_abs)

def subchain(q, i: int, j: int) -> np.ndarray:
    """
    Return the transform from frame i to frame j, given joint vector q.
    """
    fk = forward(q)
    Ts = fk.Ts_abs
    T0_i = np.eye(4) if i == 0 else Ts[i - 1]
    T0_j = np.eye(4) if j == 0 else Ts[j - 1]
    return np.linalg.inv(T0_i) @ T0_j

