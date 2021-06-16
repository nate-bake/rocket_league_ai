import numpy as np
from .actor import *
from .frame import *
from scipy.spatial.transform import Rotation
import quaternion

class player(actor):
    def __init__(self,username,rep_id,total_frames):
        super(player,self).__init__(total_frames)
        self.username = username
        self.replication_id = rep_id
        self.actor_id = -1
        self.boost_id = -1
        self.jump_id = -1
        self.dodge_id = -1
        self.flipcar_id = -1
        self.double_jump_id = -1
        self.team = None
        self.goals = 0
        self.recent_goal_diff = 0
        self.recent_rb_state = None
        self.recent_boost = 1/3.0
        self.recent_boost_bool = False
        self.recent_jump_bool = False
        self.recent_double_jump_bool = False
        self.recent_dodge_timer = False
        self.recent_dodge_bool = False
        self.recent_flipcar_bool = False
        self.recent_handbrake_bool = False
        self.recent_steer = 0.0
        self.recent_throttle = 0.0
        self.steer_guess_error = [0,0]
    
    def estimate_aerial_inputs(self,framenumber):
        if self.frames[framenumber].rb_state is None: return [0,0,0]
        if self.frames[framenumber+1].rb_state is None: return [0,0,0]
        if self.frames[framenumber+1].delta == 0: return [0,0,0]
        if self.frames[framenumber].double_jump_bool: return [0,0,0]
        # if not np.isnan(player.frames[framenumber].dodge_x_float):
            # return (0,((player.frames[framenumber].dodge_y_float)),((-1*player.frames[framenumber].dodge_x_float)))
            # return (np.nan,np.nan,np.nan) # meaning that the data here will not be used
        # if player.frames[framenumber].rb_state.position[2] < 2000: return (0,0,0)
        T_r = -36.07956616966136
        T_p = -12.14599781908070
        T_y = 8.91962804287785
        D_r = -4.47166302201591
        D_p = -2.798194258050845
        D_y = -1.886491900437232
        max = False
        if self.frames[framenumber].post_dodge_bool:
            D_p = 0
            T_p = -2.14599781908070
            D_r = -12
        elif self.frames[framenumber].rb_state.is_max_angular_velocity() and self.frames[framenumber+1].rb_state.is_max_angular_velocity():
            # print (framenumber)
            D_p = -4.26
            T_p = -7.05
            D_y = -4.9
            T_y = 6.7
            D_r = -12.4
            T_r = -25
            max = True
        else:
            D_p = -3.6
            T_p = -11.4 
            D_y = -2.9
            T_y = 9.5 
            D_r = -6
            T_r = -66
        loc_0 = np.array(self.frames[framenumber].rb_state.angular_velocity_body)/10000.
        loc_1 = np.array(self.frames[framenumber+1].rb_state.angular_velocity_body)/10000.
        tau = (loc_1-loc_0)/self.frames[framenumber+1].delta
        rhs = np.zeros(3)
        rhs[0] = tau[0]-D_r*loc_0[0]
        rhs[1] = tau[1]-D_p*loc_0[1]
        rhs[2] = tau[2]-D_y*loc_0[2]
        input = np.zeros(3)
        input[0] = rhs[0] / T_r
        input[1] = rhs[1] / (T_p + np.sign(rhs[1]) * loc_0[1] * D_p)
        input[2] = rhs[2] / (T_y - np.sign(rhs[2]) * loc_0[2] * D_y)
        if input[0] != 0.: input[0] *= -1*np.minimum(1.0,1.0/np.abs(input[0]))
        if input[1] != 0.: input[1] *= np.minimum(1.0,1.0/np.abs(input[1]))
        if input[2] != 0.: input[2] *= np.minimum(1.0,1.0/np.abs(input[2]))
        self.frames[framenumber].rb_state.estimated_aerial_input = input
        # if is_parallel_to_surface(player.frames[framenumber].rb_state.euler) and not is_in_aerial_box(player.frames[framenumber].rb_state.position):
        self.estimate_steer(framenumber)

    def estimate_steer(self,framenumber):
        loc_0 = np.array(self.frames[framenumber].rb_state.angular_velocity_body)/10000.
        loc_1 = np.array(self.frames[framenumber+1].rb_state.angular_velocity_body)/10000.

        lin = self.frames[framenumber].rb_state.linear_velocity
        lin = np.sqrt(lin[0]**2+lin[1]**2+lin[2]**2)
        lin /= 230000

        # if lin <= 0.33: lin = 0.5
        # elif lin <= 0.67: lin = 1
        # else: lin = 2

        # lin = 1

        T_y = 22.792792792792792#28.91962804287785
        D_y = -12.792792792792792#-15.886491900437232
        if self.frames[framenumber].handbrake_bool:
            D_y = -0.586491900437232
        #
        # if lin < 0.5:
        #     T_y = 28.4984984984985
        #     D_y = -17.34734734734735
        #     if player.frames[framenumber].handbrake_bool:
        #         D_y = -0.586491900437232

        # T_y = 23.323323323323336
        # D_y = -2.902902902902895 / lin
        # if player.frames[framenumber].handbrake_bool:
        #     D_y = -0.586491900437232 / lin

        tau = (loc_1-loc_0)/self.frames[framenumber+1].delta
        rhs = tau[2]-D_y*loc_0[2]
        user = rhs / (T_y - np.sign(rhs) * loc_0[2] * D_y)

        i = framenumber-1
        while self.frames[i].steer_float is None:
            i -= 1
        recent_steer = self.frames[i].steer_float
        maybe_new_steer = np.clip(user,-1,1)
        # if recent_steer != 0: maybe_new_steer = recent_steer
        if self.frames[framenumber+1].steer_float is None:
            self.frames[framenumber+1].steer_float = maybe_new_steer
        # elif not is_in_aerial_box(player.frames[framenumber+1].rb_state.position):
            # print(np.round(maybe_new_steer,2),np.round(player.frames[framenumber+1].steer_float,2))
            # player.steer_guess_error[0] += np.abs(maybe_new_steer-player.frames[framenumber+1].steer_float)
            # player.steer_guess_error[1] += 1
            # if np.round(player.frames[framenumber+1].steer_float,2) != 12:
                # print (loc_0[2], tau[2], lin)
    
    def print_positions(self):
        for frame in self.frames:
            print([frame.framenumber,frame.rb_state.position])
    
    def print_rotations(self):
        for frame in self.frames:
            if frame.rb_state is not None: print([frame.framenumber,np.round(frame.rb_state.euler,3)])
    
    def print_angular_velocities(self):
        for frame in self.frames:
            print([frame.framenumber,frame.rb_state.angular_velocity,frame.rb_state.angular_velocity_body])
    
    def print_boost(self):
        for frame in self.frames:
            print([frame.framenumber,frame.boost_float])
    
    def print_positions_boost(self):
        for frame in self.frames:
            print([frame.framenumber,frame.rb_state.position,frame.boost_float])
    
    def print_frames(self):
        np.set_printoptions(suppress=True)
        for frame in self.frames:
            if frame.rb_state is not None:
                print('frame:'+str(frame.framenumber)+'  clock:'+str(frame.time_left)+'  goal_diff:'+str(frame.goal_diff)+'  pos:'+str(frame.rb_state.position)+'  boost:'+str(frame.boost_float)+','+str(frame.boost_bool)+'  jump:'+str(frame.jump_bool)+','+str(frame.double_jump_bool)+'  dodge:'+str(frame.dodge_timer)+','+str(frame.dodge_bool)+','+str(frame.dodge_x_float)+','+str(frame.dodge_y_float)+'  drift:'+str(frame.handbrake_bool)+'  steer:'+str(frame.steer_float)+'  accel:'+str(frame.throttle_float)+',  inputs:'+str(np.round(frame.rb_state.estimated_aerial_input,2)))
    
    def debug_aerial(self):
        np.set_printoptions(suppress=True)
        for f, frame in enumerate(self.frames):
            if frame.rb_state is not None and f < 1000:
                print('f:'+str(frame.framenumber)+'  p:'+str(frame.rb_state.position[2])+'  w:'+str(np.round(frame.rb_state.angular_velocity_body,3))+'  e:'+str(np.round(frame.rb_state.euler,3))+'  s:'+str(np.round(frame.steer_float,2))+' j:'+str(int(frame.jump_bool or frame.dodge_bool))+' i:'+str(np.round(frame.rb_state.estimated_aerial_input,2))+' pd:'+str(int(frame.post_dodge_bool)))

    def debug_airroll(self):
        np.set_printoptions(suppress=True)
        for f, frame in enumerate(self.frames):
            if frame.rb_state is not None and f < 300 and (bool(frame.rb_state.estimated_aerial_input[0] == -1.0) != frame.handbrake_bool):
                print ('frame: '+str(f)+', position:'+str(frame.rb_state.position)+',  handbrake:'+str(frame.handbrake_bool)+',  airroll:'+str(frame.rb_state.estimated_aerial_input[0]))

    def print_steers(self):
        for frame in self.frames:
            print(frame.framenumber, frame.steer_float)