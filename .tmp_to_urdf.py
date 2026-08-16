import json, math
import numpy as np

data = json.load(open(r'F:\robot_arm_atlas\calib_out\extrinsics.json', encoding='utf-8'))

def T_to_xyzrpy(T):
    x, y, z = T[0][3], T[1][3], T[2][3]
    R = np.array([[T[0][0], T[0][1], T[0][2]],
                  [T[1][0], T[1][1], T[1][2]],
                  [T[2][0], T[2][1], T[2][2]]])
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        roll = math.atan2(R[2,1], R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw = math.atan2(R[1,0], R[0,0])
    else:
        roll = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw = 0.0
    return (x, y, z, roll, pitch, yaw)

for cam in ['handeye', 'front']:
    T = data[cam]['T']
    x, y, z, r, p, yw = T_to_xyzrpy(T)
    print(f'{cam}: xyz="{x:.6f} {y:.6f} {z:.6f}" rpy="{r:.6f} {p:.6f} {yw:.6f}"')
