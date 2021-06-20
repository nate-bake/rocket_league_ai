from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket

from RLUtilities.GameInfo import GameInfo
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.game_state_util import *
from rlbot.utils.structures.game_data_struct import GameTickPacket
from RLUtilities.LinearAlgebra import *

from util.boost_pad_tracker import BoostPadTracker
from util.sequence import Sequence

from rlbot.utils.game_state_util import *
import keras.models

import time
import numpy as np

import tensorflow as tf
tf.get_logger().setLevel('INFO')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def euler_to_quaternion(roll, pitch, yaw):

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qw, qx, qy, qz]

def quaternion_to_rotation_matrix(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
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
    return np.array([rotation[0],rotation[1],rotation[2]*-1])

def split_activation(l):
    l_s = tf.keras.activations.sigmoid(l[...,0:4])
    l_t = tf.keras.activations.tanh(l[...,4:])
    lnew = tf.concat([l_s, l_t], axis = 1)
    return lnew

model = keras.models.load_model('..\\models\\naynaybotbot_02-04-2021.h5',compile=False,custom_objects={"split_activation": split_activation})

class MyBot(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()
        self.controls = SimpleControllerState()
        self.count = 0
        if team: self.team = 1
        else: self.team = -1
        self.time = time.time()
        self.last_jump = -1
        self.on_ground = True
        self.printed = False

        self.info = GameInfo(index,team)

        self.model = model

        self.rows = np.empty((2000,92))

    def initialize_agent(self):
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())

    def get_output(self, packet: GameTickPacket):# -> SimpleControllerState:
        """
        This function will be called by the framework many times per second. This is where you can
        see the motion of the ball, etc. and return controls to drive your car.
        """

        self.count += 1

        self.info.read_packet(packet)
        ball = self.info.ball
        me = self.info.my_car
        opponent = self.info.opponents[0]

        me_quat = euler_to_quaternion(self.info.my_rotation[0],self.info.my_rotation[1],self.info.my_rotation[2])
        theta = quaternion_to_rotation_matrix(me_quat)
        their_quat = euler_to_quaternion(self.info.their_rotation[0],self.info.their_rotation[1],self.info.their_rotation[2])
        opp_theta = quaternion_to_rotation_matrix(their_quat)

        ball_position = (ball.pos[0],ball.pos[1],ball.pos[2]*-1)
        ball_linear_velocity = (ball.vel[0],ball.vel[1],ball.vel[2]*-1)
        my_position = (me.pos[0],me.pos[1],me.pos[2]*-1)
        my_linear_velocity = (me.vel[0],me.vel[1],me.vel[2]*-1)
        my_angular_velocity = (me.omega[0]*-1,me.omega[1]*-1,me.omega[2])
        their_position = (opponent.pos[0],opponent.pos[1],opponent.pos[2]*-1)
        their_linear_velocity = (opponent.vel[0],opponent.vel[1],opponent.vel[2]*-1)
        their_angular_velocity = (opponent.omega[0]*-1,opponent.omega[1]*-1,opponent.omega[2])

        theta[2] *= -1
        opp_theta[2] *= -1


        if self.team == -1: # flip the field
            ball_position = (ball_position[0]*-1,ball_position[1]*-1,ball_position[2])
            ball_linear_velocity = (ball_linear_velocity[0]*-1,ball_linear_velocity[1]*-1,ball_linear_velocity[2])
            my_position = (my_position[0]*-1,my_position[1]*-1,my_position[2])
            my_linear_velocity = (my_linear_velocity[0]*-1,my_linear_velocity[1]*-1,my_linear_velocity[2])
            temp_rotation = np.transpose(theta)
            theta = np.transpose([temp_rotation[0]*-1,temp_rotation[1]*-1,temp_rotation[2]])
            my_angular_velocity = (my_angular_velocity[0]*-1,my_angular_velocity[1]*-1,my_angular_velocity[2])
            their_position = (their_position[0]*-1,their_position[1]*-1,their_position[2])
            their_linear_velocity = (their_linear_velocity[0]*-1,their_linear_velocity[1]*-1,their_linear_velocity[2])
            temp_rotation = np.transpose(opp_theta)
            opp_theta = np.transpose([temp_rotation[0]*-1,temp_rotation[1]*-1,temp_rotation[2]])
            their_angular_velocity = (their_angular_velocity[0]*-1,their_angular_velocity[1]*-1,their_angular_velocity[2])

        ball_pos_relative_to_me = np.subtract(np.asarray(ball_position),np.asarray(my_position)).dot(theta)
        ball_vel_relative_to_me = np.subtract(np.asarray(ball_linear_velocity),np.asarray(my_linear_velocity)).dot(theta)
        my_vel_relative_to_me = (np.asarray(my_linear_velocity)).dot(theta)
        my_ang_velocity_body = (np.asarray(my_angular_velocity)).dot(theta)
        their_pos_relative_to_me = np.subtract(np.asarray(their_position),np.asarray(my_position)).dot(theta)
        their_vel_relative_to_me = np.subtract(np.asarray(their_linear_velocity),np.asarray(my_linear_velocity)).dot(theta)
        my_goal_relative_to_me = np.subtract(np.asarray((0,1000,17)),np.asarray(my_position)).dot(theta)
        their_goal_relative_to_me = np.subtract(np.asarray((0,-1000,17)),np.asarray(my_position)).dot(theta)

        ball_pos_relative_to_opponent = np.subtract(np.asarray(ball_position),np.asarray(their_position)).dot(opp_theta)
        ball_vel_relative_to_opponent = np.subtract(np.asarray(ball_linear_velocity),np.asarray(their_linear_velocity)).dot(opp_theta)
        their_vel_relative_to_opponent = (np.asarray(their_linear_velocity)).dot(opp_theta)
        their_ang_velocity_body = (np.asarray(their_angular_velocity)).dot(opp_theta)
        my_pos_relative_to_opponent = np.subtract(np.asarray(my_position),np.asarray(their_position)).dot(opp_theta)
        my_vel_relative_to_opponent = np.subtract(np.asarray(my_linear_velocity),np.asarray(their_linear_velocity)).dot(opp_theta)
        my_goal_relative_to_opponent = np.subtract(np.asarray((0,1000,17)),np.asarray(their_position)).dot(opp_theta)
        their_goal_relative_to_opponent = np.subtract(np.asarray((0,-1000,17)),np.asarray(their_position)).dot(opp_theta)

        # time_left = np.max(300-self.info.time,0)
        time_left = self.info.remaining

        if self.last_jump >= 0:
            dodge_timer = (time.time()-self.last_jump)/1.25 # used to be 1.25 - delta t
            if dodge_timer >= 1: dodge_timer = -1
        elif self.last_jump == -4: # infinite flip
            dodge_timer = 1
        elif self.last_jump == -2:
            dodge_timer = 0
        else:
            dodge_timer = -1

        team = self.team

        # row = [time_left/300,0,\
        row = [ball_position[0]/4500.,ball_position[1]/6000.,ball_position[2]/2125.,\
        ball_linear_velocity[0]/5000.,ball_linear_velocity[1]/5000.,ball_linear_velocity[2]/5000.,\
        # ball.omega[0],ball.omega[1],ball.omega[2],\
        \
        \
        my_position[0]/4500.,my_position[1]/6000.,my_position[2]/2125.,\
        my_linear_velocity[0]/2300.,my_linear_velocity[1]/2300.,my_linear_velocity[2]/2300.,\
        \
        my_vel_relative_to_me[0]/2300.,my_vel_relative_to_me[1]/2300.,my_vel_relative_to_me[2]/2300.,\
        ball_pos_relative_to_me[0]/10000.,ball_pos_relative_to_me[1]/10000.,ball_pos_relative_to_me[2]/10000.,\
        ball_vel_relative_to_me[0]/7300.,ball_vel_relative_to_me[1]/7300.,ball_vel_relative_to_me[2]/7300.,\
        their_pos_relative_to_me[0]/10000.,their_pos_relative_to_me[1]/10000.,their_pos_relative_to_me[2]/10000.,\
        their_vel_relative_to_me[0]/4600.,their_vel_relative_to_me[1]/4600.,their_vel_relative_to_me[2]/4600.,\
        my_goal_relative_to_me[0]/10000.,my_goal_relative_to_me[1]/10000.,my_goal_relative_to_me[2]/10000.,\
        their_goal_relative_to_me[0]/10000.,their_goal_relative_to_me[1]/10000.,their_goal_relative_to_me[2]/10000.,\
        \
        theta[0][0],theta[0][1],theta[0][2],theta[1][0],theta[1][1],theta[1][2],theta[2][0],theta[2][1],theta[2][2],\
        \
        my_angular_velocity[0]/5.5,my_angular_velocity[1]/5.5,my_angular_velocity[2]/5.5,\
        my_ang_velocity_body[0]/5.5,my_ang_velocity_body[1]/5.5,my_ang_velocity_body[2]/5.5,\
        me.boost/100,dodge_timer,\
        \
        their_position[0]/4500.,their_position[1]/6000.,their_position[2]/2125.,\
        their_linear_velocity[0]/2300.,their_linear_velocity[1]/2300.,their_linear_velocity[2]/2300.,\
        \
        their_vel_relative_to_opponent[0]/2300.,their_vel_relative_to_opponent[1]/2300.,their_vel_relative_to_opponent[2]/2300.,\
        ball_pos_relative_to_opponent[0]/10000.,ball_pos_relative_to_opponent[1]/10000.,ball_pos_relative_to_opponent[2]/10000.,\
        ball_vel_relative_to_opponent[0]/7300,ball_vel_relative_to_opponent[1]/7300,ball_vel_relative_to_opponent[2]/7300,\
        my_pos_relative_to_opponent[0]/10000.,my_pos_relative_to_opponent[1]/10000.,my_pos_relative_to_opponent[2]/10000.,\
        my_vel_relative_to_opponent[0]/4600.,my_vel_relative_to_opponent[1]/4600.,my_vel_relative_to_opponent[2]/4600.,\
        my_goal_relative_to_opponent[0]/10000.,my_goal_relative_to_opponent[1]/10000.,my_goal_relative_to_opponent[2]/10000.,\
        their_goal_relative_to_opponent[0]/10000.,their_goal_relative_to_opponent[1]/10000.,their_goal_relative_to_opponent[2]/10000.,\
        \
        opp_theta[0][0],opp_theta[0][1],opp_theta[0][2],opp_theta[1][0],opp_theta[1][1],opp_theta[1][2],opp_theta[2][0],opp_theta[2][1],opp_theta[2][2],\
        \
        their_angular_velocity[0]/5.5,their_angular_velocity[1]/5.5,their_angular_velocity[2]/5.5,\
        their_ang_velocity_body[0]/5.5,their_ang_velocity_body[1]/5.5,their_ang_velocity_body[2]/5.5,\
        ]

        row = np.asarray([row])
        newX = np.round(row,3)
        yhat = np.asarray(self.model.predict(newX)[0])


        controls = SimpleControllerState()
        controls.boost = (bool(np.round(yhat[0])) and me.boost > 0)
        controls.jump = (bool(np.round(yhat[1])) and (me.on_ground or dodge_timer > 0))
        controls.handbrake = (bool(np.round(yhat[2])) and me.on_ground)
        # dodge = (bool(np.round(yhat[3])) and dodge_timer > 0)

        # if self.count % 10 != 0:
        #     controls.jump = self.controls.jump
            # dodge = False

        if self.count % 2 == 0 and self.count < 2000:
            print(f'boost: {yhat[0]:.2f}, jump: {yhat[1]:.2f}, handbrake: {yhat[2]:.2f}, throttle: {yhat[3]:.2f}, roll: {yhat[4]:.2f}, pitch: {yhat[5]:.2f}, yaw: {yhat[6]:.2f}')
        elif self.count % 2 == 0 and self.count in range(2000,4000): 
            print(np.round(theta,3))
            print()
        elif self.count % 2 == 0 and self.count in range(4000,6000):
            print (f'last_jump: {self.last_jump}, dodge_timer: {dodge_timer}')
            if controls.jump and self.last_jump == -3: print(f'pitch: {controls.pitch}, yaw: {controls.yaw}') # trying to debug dodges

        # -1 corresponds to no recent jump
        # -2 corresponds to currently holding jump button
        # positive value is time when button released [used for dodge_timer]
        # -3 corresponds to currently using second jump [or dodge]
        # -4 corresponds to potential flip reset
        if controls.jump and self.last_jump == -1: self.last_jump = -2
        elif (not controls.jump) and self.last_jump == -2: self.last_jump = time.time()
        elif controls.jump and self.last_jump >= 0: self.last_jump = -3
        elif not controls.jump and self.last_jump == -3: self.last_jump = -1
        elif dodge_timer == 0 and self.last_jump >= 0: self.last_jump = -1

        controls.throttle = yhat[3]
        controls.pitch = yhat[5]*-1
        controls.steer = controls.yaw = yhat[6]
        controls.roll = yhat[4]

        if me.on_ground and not self.on_ground: # updates whether car is on the ground or not
            self.on_ground = True
            controls.jump = False
            self.last_jump = -1
        # elif not me.on_ground:
        #     if self.on_ground and self.last_jump == -1: self.last_jump = -4 # infinite flip?
        #     self.on_ground = False

        # if self.controls.jump and self.on_ground: # bot was trying to jump and it wasn't happening, let go.
        #     controls.jump = False
        #     self.last_jump = -1

        # print('\n'+str(time.time()-self.time)+'\n')

        # if 50 <= self.count <= 500 and self.team == -1:
        #     self.rows[self.count-50] = newX
        # if self.count == 500 and self.team == -1:
        #     np.savetxt('rows.csv',self.rows,delimiter=',')
        #     print('ROWS SAVED')

        self.time = time.time()
        self.controls = controls

        return controls
