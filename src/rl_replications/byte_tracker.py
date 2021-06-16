import numpy as np

class byte_tracker:
    def __init__(self,x,y,id,frame):
        self.x = x
        self.y = y
        self.id = id
        self.framenumber = frame
        self.byte = -1
        self.type = ''