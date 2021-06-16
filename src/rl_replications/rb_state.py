import numpy as np
from scipy.spatial.transform import Rotation
import quaternion

class rb_state:
    def __init__(self,framenumber,nan=False):
        self.framenumber = framenumber
        self.position = [0,0,0]
        self.linear_velocity = [0,0,0]
        self.quaternion = [0,0,0,0]
        self.euler = [0,0,0]
        self.angular_velocity = [0,0,0]
        self.angular_velocity_body = [0,0,0]
        self.estimated_aerial_input = [0,0,0]
        self.rotation_matrix = np.zeros((3,3))
        if nan:
            self.position = [np.nan,np.nan,np.nan]
            self.linear_velocity = [np.nan,np.nan,np.nan]
            self.quaternion = [np.nan,np.nan,np.nan,np.nan]
            self.euler = [np.nan,np.nan,np.nan]
            self.angular_velocity = [np.nan,np.nan,np.nan]
            self.angular_velocity_body = [np.nan,np.nan,np.nan]
            self.estimated_aerial_input = [np.nan,np.nan,np.nan]
            self.rotation_matrix[:,:] = np.nan

    def compute_euler_from_quaternion(self):
        w, x, y, z = self.quaternion[0], self.quaternion[1], self.quaternion[2], self.quaternion[3]
        ysqr = y * y
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = np.arctan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.where(t2>+1.0,+1.0,t2)
        t2 = np.where(t2<-1.0, -1.0, t2)
        Y = np.arcsin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = np.arctan2(t3, t4)
        self.euler = np.array((-1*Y,-1*X,Z))
        for x in range(3):
            self.euler[x] += 2*np.pi
            self.euler[x] %= 2*np.pi

    # def compute_euler_from_quaternion(self):
    #     rot = Rotation.from_quat(self.quaternion)
    #     self.euler = rot.as_euler('yxz', degrees=False)

    def compute_quaternion_from_euler(self):
        roll, pitch, yaw = self.euler[0], self.euler[1], self.euler[2]
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        self.quaternion = [qw, qx, qy, qz]

    # def compute_quaternion_from_euler(self):
    #     # r, p, y = self.euler
    #     # c1 = np.cos(y/2)
    #     # c2 = np.cos(p/2)
    #     # c3 = np.cos(r/2)
    #     # s1 = np.sin(y/2)
    #     # s2 = np.sin(p/2)
    #     # s3 = np.sin(r/2)
    #     # w = c1*c2*c3 - s1*s2*s3
    #     # x = s1*s2*c3 + c1*c2*s3
    #     # y = s1*c2*c3 + c1*s2*s3
    #     # z = c1*s2*c3 - s1*c2*s3
    #     # self.quaternion = (w,x,y,z)
    #     rot = Rotation.from_euler('yxz',e,degrees=False)
    #     self.quaternion = rot.as_quat()

    def compute_rotation_matrix(self):
        w, x, y, z = self.quaternion[0], self.quaternion[1], self.quaternion[2], self.quaternion[3]
        rotation = np.zeros((3,3),dtype=np.float32)
        rotation[0][0] = 1-2*y*y-2*z*z
        rotation[0][1] = 2*x*y - 2*z*w
        rotation[0][2] = 2*x*z + 2*y*w
        rotation[1][0] = 2*x*y + 2*z*w
        rotation[1][1] = 1-2*x*x-2*z*z
        rotation[1][2] = 2*y*z - 2*x*w
        rotation[2][0] = 2*x*z - 2*y*w
        rotation[2][1] = 2*y*z + 2*x*w
        rotation[2][2] = 1-2*x*x-2*y*y
        self.rotation_matrix = np.array([rotation[0],rotation[1],rotation[2]])
        self.rotation_matrix[2] *= -1
        self.rotation_matrix[:,2] *= -1

        # from euler instead:
            # roll, pitch, yaw = self.euler[0], self.euler[1], self.euler[2]
            # cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
            # theta = np.zeros((3,3),dtype=np.float32)
            # theta[0][0] = cp*cy
            # theta[1][0] = -1*cp*sy
            # theta[2][0] = sp
            # theta[0][1] = -1*cy*sp*sr + cr*sy
            # theta[1][1] = sy*sp*sr + cr*cy
            # theta[2][1] = -1*cp*sr
            # theta[0][2] = cr*cy*sp + sr*sy
            # theta[1][2] = -1*cr*sy*sp + sr*cy
            # theta[2][2] = cp*cr
            # self.rotation_matrix = np.transpose(theta)

    def is_max_angular_velocity(self):
        return (54500 < np.sqrt(self.angular_velocity[0]**2+self.angular_velocity[1]**2+self.angular_velocity[2]**2))

    def is_in_aerial_box(self):
        x = (-400000 < self.position[0] < 400000)
        y = (-505000 < self.position[1] < 505000)
        z = (2000 < self.position[2] < 200000)
        return x and y and z

    def is_parallel_to_floor_or_ceiling(self):
        euler = np.array(self.euler)
        for x in range(3):
            if euler[x] > np.pi: euler[x] -= 2*np.pi
        floor = np.isclose(euler[0],0,atol=0.03) and np.isclose(euler[1],0,atol=0.03)
        floor2 = np.isclose(euler[0],np.pi,atol=0.03) and np.isclose(euler[1],np.pi,atol=0.03)
        ceiling = np.isclose(euler[0],0,atol=0.03) and np.isclose(euler[1],np.pi,atol=0.03)
        ceiling2 = np.isclose(euler[1],0,atol=0.03) and np.isclose(euler[0],np.pi,atol=0.03)
        return floor or floor2 or ceiling or ceiling2

    def compute_angular_velocity_body(self):
        ang_vel_0 = np.array(self.angular_velocity)
        self.angular_velocity_body = ang_vel_0.dot(self.rotation_matrix)