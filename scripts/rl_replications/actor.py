import numpy as np
from .frame import *

class actor():
    def __init__(self,total_frames):
        self.actor_id = -1
        self.frames = [frame(x) for x in range(total_frames)]
    def print_frames(self):
        for frame in self.frames:
            if frame.rb_state is not None:
                print([frame.framenumber,frame.rb_state.position])
            else: print([frame.framenumber,None])