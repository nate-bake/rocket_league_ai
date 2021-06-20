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

import time
import struct
import numpy as np

from script_input import *

def read_binary_file():
    input_list = []
    with open('script.bin',mode='rb') as file:
        while True:
            try:
                delta = struct.unpack('f',file.read(4))[0]
                i = bot_input(delta)
                i.boost_bool, i.jump_bool, i.handbrake_bool = struct.unpack('???',file.read(3))
                i.throttle_float, i.roll_float, i.pitch_float, i.yaw_float = struct.unpack('ffff',file.read(16))
                i.fix_nans()
                input_list.append(i)
            except:
                return input_list

def build_states(bot):
    controls_list = []
    for i in bot.input_list:
        controls = SimpleControllerState()
        controls.steer = i.yaw_float#*-1 #i.steer_float
        controls.throttle = i.throttle_float
        controls.pitch = i.pitch_float*-1
        controls.yaw = i.yaw_float#*-1
        controls.roll = i.roll_float
        controls.jump = i.jump_bool# or i.double_jump_bool or i.dodge_bool or i.flipcar_bool
        controls.boost = i.boost_bool
        controls.handbrake = i.handbrake_bool
        # if i.dodge_bool:
        #     controls.pitch = i.dodge_x_float*-1
        #     controls.yaw = i.dodge_y_float
        controls_list.append(controls)
    return controls_list

class MyBot(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()
        self.input_list = read_binary_file()
        self.controls_list = build_states(self)
        self.input_index = 0
        self.time = time.time()+6.39
        self.controls = SimpleControllerState()
        self.count = 0
        if team: self.team = -1
        else: self.team = 1
        self.last_jump = -1

        self.info = GameInfo(index,team)

    def initialize_agent(self):
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())

    def get_output(self, packet: GameTickPacket):# -> SimpleControllerState:
        """
        This function will be called by the framework many times per second. This is where you can
        see the motion of the ball, etc. and return controls to drive your car.
        """

        if time.time() >= self.time:
        #     self.count += 1
        # if time.time() >= self.time(i.delta*0):
        # if self.count % 42 != 0 and time.time() >= self.time:
            self.time = time.time()
            time.sleep(.0002)
            self.last_control = self.controls_list[self.input_index]
            self.input_index += 1
            return self.last_control

        else: return SimpleControllerState()
