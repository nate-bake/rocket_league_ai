from .actor import *
from .byte_tracker import *
from .frame import *
from .player import *
from .rb_state import *

def find_rb_state(updates):
    for update in updates:
        try:
            rb_state = update['value']['rigid_body_state']
            return rb_state
        except:
            continue
    return None

def get_rb_state(framenumber, replication):
    try:
        rbs = find_rb_state(replication['value']['updated'])
        output = rb_state(framenumber)
        output.position = [rbs['location']['x'],rbs['location']['y'],rbs['location']['z']]
        output.linear_velocity = [rbs['linear_velocity']['x'],rbs['linear_velocity']['y'],rbs['linear_velocity']['z']]
        output.quaternion = [rbs['rotation']['quaternion']['w'],rbs['rotation']['quaternion']['x'],rbs['rotation']['quaternion']['y'],rbs['rotation']['quaternion']['z']]
        output.angular_velocity = [-1*rbs['angular_velocity']['x'],-1*rbs['angular_velocity']['y'],rbs['angular_velocity']['z']]
        output.compute_euler_from_quaternion()
        output.compute_rotation_matrix()
        output.compute_angular_velocity_body()
        return output
    except:
        return None

def get_handbrake(replication):
    try:
        for update in replication['value']['updated']:
            if update['name'] == "TAGame.Vehicle_TA:bReplicatedHandbrake":
                return update['value']['boolean']
    except:
        return None

def get_steer(replication):
    try:
        for update in replication['value']['updated']:
            if update['name'] == "TAGame.Vehicle_TA:ReplicatedSteer":
                return update['value']['byte']
    except:
        return None

def get_throttle(replication):
    try:
        for update in replication['value']['updated']:
            if update['name'] == "TAGame.Vehicle_TA:ReplicatedThrottle":
                return update['value']['byte']
    except:
        return None

def get_boost_amount(replication):
    try:
        for update in replication['value']['updated']:
            if update['name'] == "TAGame.CarComponent_Boost_TA:ReplicatedBoostAmount":
                return update['value']['byte']
    except:
        return None

def get_active(replication):
    try:
        for update in replication['value']['updated']:
            if update['name'] == "TAGame.CarComponent_TA:ReplicatedActive":
                return update['value']['byte']
    except:
        return None

def get_torque(replication):
    try:
        for update in replication['value']['updated']:
            if update['name'] == "TAGame.CarComponent_Dodge_TA:DodgeTorque":
                return update['value']['location']['x'], update['value']['location']['y']
        return None
    except:
        return None

def get_ball_spawn_id(replication):
    try:
        if replication['value']['spawned']['class_name'] == 'TAGame.Ball_TA':
                properties = replication['value']['spawned']
                rb_state = properties['initialization']['location']
                if rb_state['x'] == rb_state['y'] == 0: return replication['actor_id']['value']
    except:
        return None

def get_tracker(replication,framenumber):
    try:
        tracker_type = None
        if replication['value']['spawned']['class_name'] == 'TAGame.CarComponent_Boost_TA':
            tracker_type = 'boost'
        elif replication['value']['spawned']['class_name'] == 'TAGame.CarComponent_Jump_TA':
            tracker_type = 'jump'
        elif replication['value']['spawned']['class_name'] == 'TAGame.CarComponent_DoubleJump_TA':
            tracker_type = 'double_jump'
        elif replication['value']['spawned']['class_name'] == 'TAGame.CarComponent_Dodge_TA':
            tracker_type = 'dodge'
        elif replication['value']['spawned']['class_name'] == 'TAGame.CarComponent_FlipCar_TA':
            tracker_type = 'flipcar'
        if tracker_type is not None:
            loc = replication['value']['spawned']['initialization']['location']
            tracker = byte_tracker(loc['x'],loc['y'],replication['actor_id']['value'],framenumber)
            tracker.type = tracker_type
            return tracker
    except:
        return None

def get_new_player(replication,total_frames):
    try:
        for update in replication['value']['updated']:
            if update['name'] == "Engine.PlayerReplicationInfo:PlayerName":
                output = player(update['value']['string'],replication['actor_id']['value'],total_frames)
                return output
    except:
        return None

def get_game_clock(replication):
    try:
        for update in replication['value']['updated']:
            if update['name'] == "TAGame.GameEvent_Soccar_TA:SecondsRemaining":
                return update['value']['int']
    except:
        return None

def get_team_color(replication):
    try:
        for update in replication['value']['updated']:
            if update['name'] == "TAGame.Car_TA:TeamPaint":
                color = update['value']['team_paint']['team']
                if color == 0: return "blue"
                elif color == 1: return "orange"
    except:
        return None

def get_player_goals(replication):
    try:
        for update in replication['value']['updated']:
            if update['name'] == "TAGame.PRI_TA:MatchGoals": return update['value']['int']
    except:
        return None

def get_car_id(replication):
    try:
        for update in replication['value']['updated']:
            if update['name'] == "Engine.Pawn:PlayerReplicationInfo":
                return update['value']['flagged_int']['int']
    except:
        return None
