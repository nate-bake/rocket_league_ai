import numpy as np

class frame():
    def __init__(self,num):
        self.team = "none"
        self.goal_diff = None
        self.framenumber = num
        self.rb_state = None
        self.boost_float = None
        self.boost_bool = None
        self.jump_bool = None
        self.double_jump_bool = None
        self.dodge_timer = None
        self.dodge_bool = None
        self.dodge_x_float = 0
        self.dodge_y_float = 0
        self.flipcar_bool = None
        self.handbrake_bool = None
        self.steer_float = None
        self.throttle_float = None
        self.time_left = None
        self.delta = None
        self.post_dodge_bool = False
    def jump_button_bool(self):
        if self.dodge_bool and (self.dodge_x_float != 0 or self.dodge_y_float != 0): return True
        else: return (self.jump_bool or self.double_jump_bool or self.flipcar_bool)
        # return (self.jump_bool or self.double_jump_bool or self.flipcar_bool)