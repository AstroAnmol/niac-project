import math
import numpy as np

def rv_to_oe(r, v):
    mu = 398600.4418
    r_mag = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    v_mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    
    h = [
        r[1]*v[2] - r[2]*v[1],
        r[2]*v[0] - r[0]*v[2],
        r[0]*v[1] - r[1]*v[0]
    ]
    h_mag = math.sqrt(h[0]**2 + h[1]**2 + h[2]**2)
    n = [-h[1], h[0], 0]
    n_mag = math.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
    
    if n_mag == 0:
        raan = 0
    else:
        raan = math.acos(max(-1.0, min(1.0, n[0] / n_mag)))
        if n[1] < 0:
            raan = 2*math.pi - raan
            
    return math.degrees(raan)

# Satellite state at Omega=0, i=45, theta=45
r_sat = np.array([5000.0, 5000.0, 5000.0]) # just arbitrary high Z point
v_sat = np.array([-5.0, 5.0, 0.0])

h_vec = np.cross(r_sat, v_sat)
u_radial = r_sat / np.linalg.norm(r_sat)
u_cross = h_vec / np.linalg.norm(h_vec)
u_along = np.cross(u_cross, u_radial)

v_mag = 7.5
for angle_deg in range(0, 360, 45):
    angle = math.radians(angle_deg)
    v_deb = v_mag * math.cos(angle) * u_along + v_mag * math.sin(angle) * u_cross
    raan = rv_to_oe(r_sat, v_deb)
    print(f"Angle {angle_deg}: RAAN = {raan:.2f}")

