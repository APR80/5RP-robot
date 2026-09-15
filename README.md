## Modeling, control, and simulation of a 5RP robot (MuJoCo)
This project implements the **modeling and control of a 5RP robot** and validates using the **MuJoCo** physics engine. It covers key robotics topics.

https://github.com/user-attachments/assets/f9af8e9d-b5ae-4d3b-bcb9-022d286ebe08

> **Note.** This project was completed as the final project for a robotics course at the University of Tehran. The work closely follows *Introduction to Robotics: Mechanics and Control* by John J. Craig (except where noted). The course required a 5RP configuration for that I started from the UR5e XML available in the MuJoCo menagerie, disabled the original wrist 1 joint, and added a linear actuator at the end-effector to get 5RP. The XML did not use the DH frame convention from Craig, so I reassigned frames in the XML and updated link masses and inertia tensors accordingly.
## Algorithms
<details>
  <summary><b>Kinematics</b></summary>

  - Forward Kinematics  
  - Inverse Kinematics  
  - Velocity Propagation  
  - Jacobian Calculation  
    - Explicit method  
    - Using velocity propagation  
</details>

<details>
  <summary><b>Dynamics</b></summary>

  - Newton–Euler method  
  - Lagrangian method  
</details>

<details>
  <summary><b>Trajectory Generation</b></summary>

  - Cubic polynomial  
  - Quintic polynomial  
  - Linear segments with parabolic blend (LSPB)  
  - Time-optimal time scaling  
</details>

<details>
  <summary><b>Control</b></summary>

  - Linear control  
  - Inverse dynamics control  
  - Transpose-Jacobian control  
  - Modified transpose-Jacobian control  
</details>

##  Usage
```bash
git clone https://github.com/APR80/5RP-Robot.git
cd 5RP-Robot
# Tested on Python 3.8; other Python 3.x versions (e.g. 3.9–3.11) will likely work.
py -3.8 -m venv .venv
.\.venv\Scripts\activate.bat
# On macOS / Linux:
# python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```
