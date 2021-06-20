import numpy as np

def byte_to_percent_float(byte,signed=True):
    if signed:
        if ((byte-128)*100/128) == 99: return 1.0
        else: return ((byte-128)/128)
    else:
        return (byte/256)

class bot_input:
    def __init__(self,d):
        self.delta = d
        self.boost_bool = None
        self.jump_bool = None
        # self.double_jump_bool = None
        # self.dodge_bool = None
        # self.dodge_x_float = 0
        # self.dodge_y_float = 0
        # self.flipcar_bool = None
        # self.handbrake_bool = None
        # self.steer_float = None
        self.throttle_float = None
        self.roll_float = 0
        self.pitch_float = 0
        self.yaw_float = 0
    def fix_nans(self):
        if np.isnan(self.roll_float): self.roll_float = 0.0
        if np.isnan(self.pitch_float): self.pitch_float = 0.0
        if np.isnan(self.yaw_float): self.yaw_float = 0.0
    def print_self(self):
        np.set_printoptions(suppress=True)
        print('delta:'+str(self.delta))
        print('boost:'+str(self.boost_bool))
        print('jump:'+str(self.jump_bool or self.double_jump_bool or self.dodge_bool or self.flipcar_bool))
        print('handbrake:'+str(self.handbrake_bool))
        print('steer:'+str(self.steer_byte)+','+str(byte_to_percent_float(self.steer_byte)))
        print('throttle:'+str(byte_to_percent_float(self.throttle_byte)))
        print('roll:'+str(self.roll_float))
        print('pitch:'+str(self.pitch_float))
        print('yaw:'+str(self.yaw_float))
