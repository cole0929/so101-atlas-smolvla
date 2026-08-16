import json
import math

import numpy as np

data = json.load(open(r'F:\robot_arm_atlas\calib_out\front_extrinsic_geometric.json', encoding='utf-8'))
T = np.array(data['T'])

x, y, z = T[0, 3], T[1, 3], T[2, 3]
R = T[:3, :3]
sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
if sy > 1e-6:
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.atan2(-R[2, 0], sy)
    yaw = math.atan2(R[1, 0], R[0, 0])
else:
    roll = math.atan2(-R[1, 2], R[1, 1])
    pitch = math.atan2(-R[2, 0], sy)
    yaw = 0.0
print('xyz="{:.6f} {:.6f} {:.6f}" rpy="{:.6f} {:.6f} {:.6f}"'.format(x, y, z, roll, pitch, yaw))
