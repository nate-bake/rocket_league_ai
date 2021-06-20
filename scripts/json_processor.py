import json
import numpy as np
import time
import copy
from rl_replications.get_replications import *
from rl_replications.actor import *
from rl_replications.byte_tracker import *
from rl_replications.frame import *
from rl_replications.player import *
from rl_replications.rb_state import *
import sys
import quaternion

def bool_to_float(bool):
    if np.isnan(bool): return np.nan
    elif bool: return 1.0
    else: return 0.0

def byte_to_bool(byte):
    if byte < 0: return False
    return byte % 2 != 0

def player_exists(new_player):
    global players
    for player in players:
        if player.username == new_player.username:
            return True
    return False

def initialize_players(frame,total_frames,making_script):
    global players
    global me
    for replication in frame['replications']:
        new_player = get_new_player(replication,total_frames)
        if new_player is not None and not player_exists(new_player):
            players.append(new_player)
    if len(players) != 2 and not making_script: # only solo duel
        sys.exit(1)

def choose_me_color(color):
    global players
    for player in players:
        if player.team == color: me = player
        else: opponent = player
    return (me, opponent)

def choose_me_username(username):
    global players
    me = None
    for player in players:
        if player.username == username: me = player
    return me

def define_teams(frame):
    for replication in frame['replications']:
        team_color = get_team_color(replication)
        if team_color is not None:
            for player in players:
                if replication['actor_id']['value'] == player.actor_id and player.team is None:
                    player.team = team_color
                    break

def update_game_score():
    blue = orange = 0
    for player in players:
        if player.team == "blue": blue += player.goals
        else: orange += player.goals
    orange_diff = orange - blue
    for player in players:
        if player.team == "orange": player.recent_goal_diff = orange_diff
        else: player.recent_goal_diff = -1*orange_diff

def read_frame_spawns(frame,framenumber):
    for replication in frame['replications']:

        ball_spawn_id = get_ball_spawn_id(replication)
        if ball_spawn_id is not None:
            ball.actor_id = ball_spawn_id
            continue

        car_id = get_car_id(replication)
        if car_id is not None:
            for player in players:
                if car_id == player.replication_id and player.actor_id != replication['actor_id']['value']:
                    player.actor_id = replication['actor_id']['value']
                    player.recent_boost = player.frames[framenumber].boost_float = 1/3.0
                    player.recent_boost_bool = player.frames[framenumber].boost_bool = False
                    player.recent_jump_bool = player.frames[framenumber].jump_bool = False
                    player.recent_double_jump_bool = player.frames[framenumber].double_jump_bool = False
                    player.recent_dodge_bool = player.frames[framenumber].dodge_bool = False
                    player.recent_flipcar_bool = player.frames[framenumber].flipcar_bool = False
                    player.recent_handbrake_bool = player.frames[framenumber].handbrake_bool = False
                    player.recent_steer = player.frames[framenumber].steer_float = 0
                    player.recent_throttle = player.frames[framenumber].throttle_float = 0
                    break

def read_rb_state(framenumber,replication):
    rb_state = get_rb_state(framenumber,replication)
    if rb_state is not None:
        if replication['actor_id']['value'] == ball.actor_id:
            ball.frames[framenumber].rb_state = rb_state
        else:
            for player in players:
                if replication['actor_id']['value'] == player.actor_id:
                    player.recent_rb_state = player.frames[framenumber].rb_state = rb_state
                    break

def read_handbrake_state(framenumber,replication):
    handbrake = get_handbrake(replication)
    if handbrake is not None:
        for player in players:
            if replication['actor_id']['value'] == player.actor_id:
                player.recent_handbrake_bool = player.frames[framenumber].handbrake_bool = byte_to_bool(handbrake)
                break

def read_steer_state(framenumber,replication):
    steer = get_steer(replication)
    if steer is not None:
        for player in players:
            if replication['actor_id']['value'] == player.actor_id:
                player.recent_steer = player.frames[framenumber].steer_float = (steer-128)/127
                break

def read_throttle_state(framenumber, replication):
    throttle = get_throttle(replication)
    if throttle is not None:
        for player in players:
            if replication['actor_id']['value'] == player.actor_id:
                player.recent_throttle = player.frames[framenumber].throttle_float = (throttle-128)/127
                break

def read_game_clock(replication):
    game_clock = get_game_clock(replication)
    global in_regulation
    global time_left
    if game_clock is not None:
        if in_regulation:
            time_left = game_clock
            if game_clock == 0: in_regulation = False
        else:
            time_left = 0

def read_game_score(replication):
    player_goals = get_player_goals(replication)
    if player_goals is not None:
        for player in players:
            if replication['actor_id']['value'] == player.replication_id:
                player.goals = player_goals
                update_game_score()


def read_frame_states(frame, framenumber):
    for replication in frame['replications']:
    
        read_rb_state(framenumber,replication)
        read_handbrake_state(framenumber,replication)
        read_steer_state(framenumber,replication)
        read_throttle_state(framenumber, replication)

        read_game_clock(replication)
        read_game_score(replication)

def get_frame_delta(frame,framenumber):
    for player in players:
        player.frames[framenumber].delta = frame['delta']
    return frame['delta']

def belongs_to(player,tracker):
    x = np.isclose(tracker.x/10, player.recent_rb_state.position[0]/1000,atol=500)
    y = np.isclose(tracker.y/10, player.recent_rb_state.position[1]/1000,atol=500)
    return x and y

def assign_frame_trackers(frame, framenumber):
    global unclaimed_trackers
    for replication in frame['replications']:
        tracker = get_tracker(replication,framenumber)
        if tracker is not None:
            found_owner = False
            for player in players:
                if belongs_to(player,tracker):
                    if tracker.type == 'boost': player.boost_id = tracker.id
                    elif tracker.type == 'jump': player.jump_id = tracker.id
                    elif tracker.type == 'double_jump': player.double_jump_id = tracker.id
                    elif tracker.type == 'dodge': player.dodge_id = tracker.id
                    elif tracker.type == 'flipcar': player.flipcar_id = tracker.id
                    found_owner = True
                    break
            if not found_owner:
                unclaimed_trackers.append(tracker)

def adopt_unclaimed_trackers(framenumber):
    global unclaimed_trackers
    for tracker in unclaimed_trackers:
        if (framenumber - tracker.framenumber) > 10:
            unclaimed_trackers.remove(tracker)
            continue
        for player in players:
            if belongs_to(player,tracker):
                if tracker.type == 'boost':
                    player.boost_id = tracker.id
                    if tracker.byte != -1: player.frames[framenumber].boost_float = (tracker.byte)/255.0
                elif tracker.type == 'jump':
                    player.jump_id = tracker.id
                elif tracker.type == 'double_jump':
                    player.double_jump_id = tracker.id
                elif tracker.type == 'dodge':
                    player.dodge_id = tracker.id
                elif tracker.type == 'flipcar':
                    player.flipcar_id = tracker.id
                unclaimed_trackers.remove(tracker)
                break

def read_boost_amount_update(framenumber, replication):
    boost_amount = get_boost_amount(replication)
    if boost_amount is not None:
        for player in players:
            if replication['actor_id']['value'] == player.boost_id:
                player.recent_boost = player.frames[framenumber].boost_float = (boost_amount)/255.0
                return
        for tracker in unclaimed_trackers:
            if tracker.type == 'boost' and tracker.id == replication['actor_id']['value']:
                tracker.byte = boost_amount
                return

def read_tracker_active_update(framenumber, replication):
    active = get_active(replication)
    if active is not None:
        for player in players:
            if replication['actor_id']['value'] == player.boost_id:
                player.recent_boost_bool = player.frames[framenumber].boost_bool = byte_to_bool(active)
                break
            elif replication['actor_id']['value'] == player.jump_id:
                if byte_to_bool(active): player.recent_dodge_timer = 0 # prompts update_dodge_timers() to wait for release
                player.recent_jump_bool = player.frames[framenumber].jump_bool = byte_to_bool(active)
                break
            elif replication['actor_id']['value'] == player.double_jump_id:
                if byte_to_bool(active): player.recent_dodge_timer = -1 # tells update_dodge_timers() to reset
                player.recent_double_jump_bool = player.frames[framenumber].double_jump_bool = byte_to_bool(active)
                break
            elif replication['actor_id']['value'] == player.dodge_id:
                if byte_to_bool(active):
                    player.recent_dodge_timer = -1
                    torque = get_torque(replication)
                    if torque is not None:
                        player.frames[framenumber].dodge_x_float, player.frames[framenumber].dodge_y_float = torque
                        max_direction = np.max((np.abs(player.frames[framenumber].dodge_x_float), np.abs(player.frames[framenumber].dodge_y_float)))
                        if max_direction != 0:
                            player.frames[framenumber].dodge_x_float /= max_direction
                            player.frames[framenumber].dodge_y_float /= max_direction
                player.recent_dodge_bool = player.frames[framenumber].post_dodge_bool = player.frames[framenumber].dodge_bool = byte_to_bool(active)
                break
            elif replication['actor_id']['value'] == player.flipcar_id:
                player.recent_flipcar_bool = player.frames[framenumber].flipcar_bool = byte_to_bool(active)
                break

def read_frame_updates(frame,framenumber):
    for replication in frame['replications']:
        read_boost_amount_update(framenumber, replication)
        read_tracker_active_update(framenumber, replication)

def update_boost_amounts(framenumber, delta):
    consumption_rate = 1/3.0 # 85 per second [full boost = 255] 
    for player in players:
        if player.frames[framenumber].boost_float == None: # otherwise we already know how much boost they have this frame
            if player.recent_boost_bool:
                new_amount = (np.maximum((player.recent_boost - consumption_rate*delta),0))
                player.recent_boost = player.frames[framenumber].boost_float = new_amount
            # else boost will be copied over in fill_none_with_recent()

def update_dodge_timers(framenumber, delta):
    # timer is actually counting up. decimal [0,1] describing % elapsed
    # -1 means timer not initiated
    # 0 means player is still jumping
    for player in players:
        if player.recent_dodge_timer >= 0 and not player.frames[framenumber].jump_bool:
            new_timer = (player.recent_dodge_timer*1.25 + delta)/1.25
            if new_timer > 1: # dodge timer expired, reset
                player.recent_dodge_timer = -1
            else:
                player.recent_dodge_timer = player.frames[framenumber].dodge_timer = new_timer
                # this is the only time the frame's dodge timer values are set.
                # wherever None, the dodge_timer will be -1 in the dataset

def fill_none_with_recent(framenumber):
    for player in players:
        if player.frames[framenumber].boost_float is None:
            player.frames[framenumber].boost_float = player.recent_boost
        if player.frames[framenumber].boost_bool is None:
            if player.recent_boost == 0: player.recent_boost_bool = False
            player.frames[framenumber].boost_bool = player.recent_boost_bool
        if player.frames[framenumber].jump_bool is None:
            player.frames[framenumber].jump_bool = player.recent_jump_bool
        if player.frames[framenumber].double_jump_bool is None:
            player.frames[framenumber].double_jump_bool = player.recent_double_jump_bool
        if player.frames[framenumber].dodge_bool is None:
            player.frames[framenumber].dodge_bool = False # b/c dodges are instant
            if player.recent_dodge_bool: player.frames[framenumber].post_dodge_bool = True # b/c we still need to know when dodge ends for estimate_aerial_inputs()
        if player.frames[framenumber].flipcar_bool is None:
            player.frames[framenumber].flipcar_bool = player.recent_flipcar_bool
        if player.frames[framenumber].handbrake_bool is None:
            player.frames[framenumber].handbrake_bool = player.recent_handbrake_bool
        # if player.frames[framenumber].steer_float is None:
        #     player.frames[framenumber].steer_float = player.recent_steer
        # ^ disabled as we try out estimate_steer()
        if player.frames[framenumber].throttle_float is None:
            player.frames[framenumber].throttle_float = player.recent_throttle
        if player.frames[framenumber].time_left is None:
            player.frames[framenumber].time_left = time_left
        player.frames[framenumber].team = player.team
        player.frames[framenumber].goal_diff = player.recent_goal_diff

def interpolate_euler(open,close,multiplier,divisor):
    o = np.array(open)
    c = np.array(close)
    result = np.zeros(3)
    for x in range(3):
        diff = (c[x]-o[x]+np.pi) % (2*np.pi) - np.pi
        result[x] = o[x] + diff * multiplier/divisor
        if result[x] < 0.0: result[x] += (2*np.pi)
    return result

def interpolate_quat(open,close,multiplier,divisor):
    p0 = np.quaternion(open[0],open[1],open[2],open[3])
    p1 = np.quaternion(close[0],close[1],close[2],close[3])
    t = multiplier/divisor
    slerped = (np.slerp_vectorized(p0, p1, t))
    return (slerped.w,slerped.x,slerped.y,slerped.z)

def linear_fill_rb_states(player,firstnone,how_many):
    open = player.frames[firstnone-1].rb_state
    close = player.frames[firstnone+how_many].rb_state
    if close.position[0] % 100 == 0 and close.position[1] % 100 == 0 and close.linear_velocity[0:2] == (0,0): return
    divisor = how_many+1
    for i in range(1,how_many+1):
        new_rb_state = rb_state(firstnone+i)
        new_rb_state.position = (open.position+i*np.subtract(close.position,open.position)/divisor).astype(np.int)
        new_rb_state.euler = interpolate_euler(open.euler,close.euler,i,divisor)
        new_rb_state.linear_velocity = (open.linear_velocity+i*np.subtract(close.linear_velocity,open.linear_velocity)/divisor).astype(np.int)
        new_rb_state.quaternion = interpolate_quat(open.quaternion,close.quaternion,i,divisor)
        new_rb_state.angular_velocity = tuple((open.angular_velocity+i*np.subtract(close.angular_velocity,open.angular_velocity)/divisor).astype(np.int))
        new_rb_state.compute_rotation_matrix()
        new_rb_state.compute_angular_velocity_body()        
        player.frames[firstnone+i-1].rb_state = new_rb_state

def fill_copy_rb_states(player,firstnone,previous_state):
    while firstnone < len(player.frames) and player.frames[firstnone].rb_state is None:
        player.frames[firstnone].rb_state = copy.copy(previous_state)
        firstnone += 1

def linear_fill_steers(player,firstnone,how_many):
    open = player.frames[firstnone-1].steer_float
    close = player.frames[firstnone+how_many].steer_float
    divisor = how_many+1
    for i in range(1,how_many+1):
        player.frames[firstnone+i-1].steer_float = (int)(open+i*(close-open)/divisor)

def fill_copy_steers(player,firstnone,previous_steer):
    while firstnone < len(player.frames) and player.frames[firstnone].steer_float is None:
        player.frames[firstnone].steer_float = previous_steer
        firstnone += 1

def interpolate_rb_state_nones(max_gap=3,max_copy=80): # smoothes for gaps <= 3, fills with previous for larger gaps less than 80
    for player in players+[ball]:
        last_existed = True
        previous_rb_state = None
        wait_until_frame = -1
        for frame in player.frames:
            if frame.framenumber < wait_until_frame: continue
            if (len(player.frames)-frame.framenumber) > max_gap and frame.rb_state is None:
                nones = 1
                while frame.framenumber+nones < len(player.frames):
                    if player.frames[frame.framenumber+nones].rb_state is None: nones += 1
                    else: break
                if nones <= max_gap and last_existed: linear_fill_rb_states(player,frame.framenumber,nones)
                else:
                    last_existed = False
                    if nones <= max_copy: fill_copy_rb_states(player,frame.framenumber,previous_rb_state)
                    else: wait_until_frame = frame.framenumber + nones
            elif frame.rb_state is not None: # to keep previous from becoming None
                previous_rb_state = frame.rb_state
                last_existed = True
        for x in range(1,max_gap+1): # previous loop misses last few frames due to gap
            if player.frames[-1*x].rb_state is None: player.frames[-1*x].rb_state = previous_rb_state

def interpolate_steer_nones(max_gap=2): # smoothes for gaps <= 2, fills with previous for larger gaps
    for player in players:
        last_existed = False
        previous_steer = 0
        for frame in player.frames:
            if (len(player.frames)-frame.framenumber) > max_gap and frame.steer_float is None:
                nones = 1
                while frame.framenumber+nones < len(player.frames):
                    if player.frames[frame.framenumber+nones].steer_float is None: nones += 1
                    else: break
                if nones <= max_gap and last_existed: linear_fill_steers(player,frame.framenumber,nones)
                else:
                    last_existed = False
                    fill_copy_steers(player,frame.framenumber,previous_steer) # if gap too big, just fill with previous
            elif frame.steer_float is not None:  # to keep previous from becoming None
                previous_steer = frame.steer_float
                last_existed = True
        for x in range(1,max_gap+1): # previous loop misses last few frames due to gap
            if player.frames[-1*x].steer_float is None: player.frames[-1*x].steer_float = previous_steer

def resolve_kickoff_states(total_frames):
    # removes all but one kickoff state to avoid redundant rows during countdown
    for player in players:
        for frame in player.frames:
            if frame.rb_state is not None and frame.rb_state.position[0] % 100 == 0 and frame.rb_state.position[1] % 100 == 0:
                same_spot_count = 1
                spot = [frame.rb_state.position[0],frame.rb_state.position[1]]
                i = frame.framenumber+1
                while i < total_frames-1 and (player.frames[i].rb_state is None or np.array_equal(player.frames[i].rb_state.position[0:2], spot)):
                    same_spot_count += 1
                    i += 1
                if same_spot_count >= 3:
                    if player.frames[i-1].rb_state is None:
                        j=2
                        while True:
                            if player.frames[i-j].rb_state is not None: break
                            else: j += 1
                        player.frames[i-1].rb_state = player.frames[i-j].rb_state
                    player.frames[i-1].rb_state.position = [spot[0],spot[1],1701]
                    for f in range(frame.framenumber,i-1):
                        player.frames[f].rb_state = None
    for frame in ball.frames:
        if frame.rb_state is not None and frame.rb_state.position[0] == 0 and frame.rb_state.position[1] == 0 and frame.rb_state.position[2] > 9274:
            frame.rb_state = None
    last_was_centered = True
    for frame in ball.frames:
        if frame.rb_state is None and last_was_centered: frame.rb_state = ball.frames[frame.framenumber-1].rb_state
        elif frame.rb_state is not None and np.array_equal(frame.rb_state.position, [0,0,9274]): last_was_centered = True
        elif frame.rb_state is not None: last_was_centered = False

def estimate_all_inputs():
    for player in players:
        for i in range(len(player.frames)-1):
            player.estimate_aerial_inputs(i)

def write_frame(framenumber,output,count):

    frame = me.frames[framenumber]
    next_frame = me.frames[framenumber+1]
    ball_state = ball.frames[framenumber].rb_state
    my_state = me.frames[framenumber].rb_state
    their_state = opponent.frames[framenumber].rb_state

    if ball_state is None or their_state is None or \
    my_state is None or next_frame.rb_state is None: return 0

    if not my_state.is_in_aerial_box():
        next_frame.rb_state.estimated_aerial_input[2] = next_frame.steer_float
        next_frame.rb_state.estimated_aerial_input[0:2] = [0,0]
    else:
        next_frame.handbrake_bool = False
    if next_frame.dodge_bool:
        next_frame.rb_state.estimated_aerial_input[1] = next_frame.dodge_y_float
        next_frame.rb_state.estimated_aerial_input[2] = next_frame.dodge_x_float*-1
    if next_frame.double_jump_bool:
        next_frame.rb_state.estimated_aerial_input[1:3] = [0,0]

    if frame.boost_float == 0: next_frame.boost_bool = False

    # if next_frame.dodge_bool: next_frame.rb_state.estimated_aerial_input = (np.nan,np.nan,np.nan)
    # else: next_frame.dodge_x_float = next_frame.dodge_y_float = np.nan

    # if is_in_aerial_box(my_state.position):
        # next_frame.handbrake_bool = np.nan
        # next_frame.steer_float = np.nan
        # if frame.dodge_timer is None: next_frame.jump_bool = next_frame.dodge_bool = np.nan
    # else:
        # next_frame.rb_state.estimated_aerial_input = (np.nan,np.nan,np.nan)
        # if frame.dodge_timer is None:
            # next_frame.dodge_bool = np.nan
            # next_frame.dodge_x_float = next_frame.dodge_y_float = np.nan

    # if frame.boost_float == 0: next_frame.boost_bool = np.nan

    if ball_state.position[2] > 0: ball_state.position[2] *= -1
    my_state.position[2] *= -1
    their_state.position[2] *= -1
    ball_state.linear_velocity[2] *= -1
    my_state.linear_velocity[2] *= -1
    their_state.linear_velocity[2] *= -1

    if me.team == "blue": # flip the field

        ball_state.position = [ball_state.position[0]*-1,ball_state.position[1]*-1,ball_state.position[2]]
        ball_state.linear_velocity = [ball_state.linear_velocity[0]*-1,ball_state.linear_velocity[1]*-1,ball_state.linear_velocity[2]]
        my_state.position = [my_state.position[0]*-1,my_state.position[1]*-1,my_state.position[2]]
        my_state.linear_velocity = [my_state.linear_velocity[0]*-1,my_state.linear_velocity[1]*-1,my_state.linear_velocity[2]]
        temp_rotation = np.transpose(my_state.rotation_matrix)
        my_state.rotation_matrix = np.transpose([temp_rotation[0]*-1,temp_rotation[1]*-1,temp_rotation[2]])
        my_state.angular_velocity = [my_state.angular_velocity[0]*-1,my_state.angular_velocity[1]*-1,my_state.angular_velocity[2]]
        their_state.position = [their_state.position[0]*-1,their_state.position[1]*-1,their_state.position[2]]
        their_state.linear_velocity = [their_state.linear_velocity[0]*-1,their_state.linear_velocity[1]*-1,their_state.linear_velocity[2]]
        temp_rotation = np.transpose(their_state.rotation_matrix)
        their_state.rotation_matrix = np.transpose([temp_rotation[0]*-1,temp_rotation[1]*-1,temp_rotation[2]])
        their_state.angular_velocity = [their_state.angular_velocity[0]*-1,their_state.angular_velocity[1]*-1,their_state.angular_velocity[2]]

        my_state.angular_velocity_body = (np.asarray(my_state.angular_velocity)/10000.0).dot(my_state.rotation_matrix)
        their_state.angular_velocity_body = (np.asarray(their_state.angular_velocity)/10000.0).dot(their_state.rotation_matrix)


    ball_pos_relative_to_me = np.subtract(np.asarray(ball_state.position),np.asarray(my_state.position)).dot(my_state.rotation_matrix)
    ball_vel_relative_to_me = np.subtract(np.asarray(ball_state.linear_velocity),np.asarray(my_state.linear_velocity)).dot(my_state.rotation_matrix)
    my_vel_relative_to_me = np.asarray(my_state.linear_velocity).dot(my_state.rotation_matrix)
    their_pos_relative_to_me = np.subtract(np.asarray(their_state.position),np.asarray(my_state.position)).dot(my_state.rotation_matrix)
    their_vel_relative_to_me = np.subtract(np.asarray(their_state.linear_velocity),np.asarray(my_state.linear_velocity)).dot(my_state.rotation_matrix)
    my_goal_relative_to_me = np.subtract(np.asarray((0,100000,1701)),np.asarray(my_state.position)).dot(my_state.rotation_matrix)
    their_goal_relative_to_me = np.subtract(np.asarray((0,-100000,1701)),np.asarray(my_state.position)).dot(my_state.rotation_matrix)

    ball_pos_relative_to_opponent = np.subtract(np.asarray(ball_state.position),np.asarray(their_state.position)).dot(their_state.rotation_matrix)
    ball_vel_relative_to_opponent = np.subtract(np.asarray(ball_state.linear_velocity),np.asarray(their_state.linear_velocity)).dot(their_state.rotation_matrix)
    their_vel_relative_to_opponent = np.asarray(their_state.linear_velocity).dot(their_state.rotation_matrix)
    my_pos_relative_to_opponent = np.subtract(np.asarray(my_state.position),np.asarray(their_state.position)).dot(their_state.rotation_matrix)
    my_vel_relative_to_opponent = np.subtract(np.asarray(my_state.linear_velocity),np.asarray(their_state.linear_velocity)).dot(their_state.rotation_matrix)
    my_goal_relative_to_opponent = np.subtract(np.asarray((0,100000,1701)),np.asarray(their_state.position)).dot(their_state.rotation_matrix)
    their_goal_relative_to_opponent = np.subtract(np.asarray((0,-100000,1701)),np.asarray(their_state.position)).dot(their_state.rotation_matrix)

    if frame.dodge_timer is None: frame.dodge_timer = -1

    # output[count] = [frame.time_left/300.,me.frames[framenumber].goal_diff/10.,\
    output[count] = [ball_state.position[0]/450000.,ball_state.position[1]/600000.,ball_state.position[2]/212500.,\
    ball_state.linear_velocity[0]/500000,ball_state.linear_velocity[1]/500000,ball_state.linear_velocity[2]/500000,\
    \
    \
    my_state.position[0]/450000.,my_state.position[1]/600000.,my_state.position[2]/212500.,\
    my_state.linear_velocity[0]/230000.,my_state.linear_velocity[1]/230000.,my_state.linear_velocity[2]/230000.,\
    \
    my_vel_relative_to_me[0]/230000.,my_vel_relative_to_me[1]/230000.,my_vel_relative_to_me[2]/230000.,\
    ball_pos_relative_to_me[0]/1000000.,ball_pos_relative_to_me[1]/1000000.,ball_pos_relative_to_me[2]/1000000.,\
    ball_vel_relative_to_me[0]/730000,ball_vel_relative_to_me[1]/730000,ball_vel_relative_to_me[2]/730000,\
    their_pos_relative_to_me[0]/1000000.,their_pos_relative_to_me[1]/1000000.,their_pos_relative_to_me[2]/1000000.,\
    their_vel_relative_to_me[0]/460000,their_vel_relative_to_me[1]/460000,their_vel_relative_to_me[2]/460000,\
    my_goal_relative_to_me[0]/1000000.,my_goal_relative_to_me[1]/1000000.,my_goal_relative_to_me[2]/1000000.,\
    their_goal_relative_to_me[0]/1000000.,their_goal_relative_to_me[1]/1000000.,their_goal_relative_to_me[2]/1000000.,\
    \
    my_state.rotation_matrix[0][0],my_state.rotation_matrix[0][1],my_state.rotation_matrix[0][2],\
    my_state.rotation_matrix[1][0],my_state.rotation_matrix[1][1],my_state.rotation_matrix[1][2],\
    my_state.rotation_matrix[2][0],my_state.rotation_matrix[2][1],my_state.rotation_matrix[2][2],\
    \
    my_state.angular_velocity[0]/55000.,my_state.angular_velocity[1]/55000.,my_state.angular_velocity[2]/55000.,\
    my_state.angular_velocity_body[0]/5.5,my_state.angular_velocity_body[1]/5.5,my_state.angular_velocity_body[2]/5.5,\
    frame.boost_float,frame.dodge_timer,\
    \
    \
    their_state.position[0]/450000.,their_state.position[1]/600000.,their_state.position[2]/212500.,\
    their_state.linear_velocity[0]/230000.,their_state.linear_velocity[1]/230000.,their_state.linear_velocity[2]/230000.,\
    \
    their_vel_relative_to_opponent[0]/230000.,their_vel_relative_to_opponent[1]/230000.,their_vel_relative_to_opponent[2]/230000.,\
    ball_pos_relative_to_opponent[0]/1000000.,ball_pos_relative_to_opponent[1]/1000000.,ball_pos_relative_to_opponent[2]/1000000.,\
    ball_vel_relative_to_opponent[0]/730000,ball_vel_relative_to_opponent[1]/730000,ball_vel_relative_to_opponent[2]/730000,\
    my_pos_relative_to_opponent[0]/1000000.,my_pos_relative_to_opponent[1]/1000000.,my_pos_relative_to_opponent[2]/1000000.,\
    my_vel_relative_to_opponent[0]/460000,my_vel_relative_to_opponent[1]/460000,my_vel_relative_to_opponent[2]/460000,\
    my_goal_relative_to_opponent[0]/1000000.,my_goal_relative_to_opponent[1]/1000000.,my_goal_relative_to_opponent[2]/1000000.,\
    their_goal_relative_to_opponent[0]/1000000.,their_goal_relative_to_opponent[1]/1000000.,their_goal_relative_to_opponent[2]/1000000.,\
    \
    their_state.rotation_matrix[0][0],their_state.rotation_matrix[0][1],their_state.rotation_matrix[0][2],\
    their_state.rotation_matrix[1][0],their_state.rotation_matrix[1][1],their_state.rotation_matrix[1][2],\
    their_state.rotation_matrix[2][0],their_state.rotation_matrix[2][1],their_state.rotation_matrix[2][2],\
    \
    their_state.angular_velocity[0]/55000.,their_state.angular_velocity[1]/55000.,their_state.angular_velocity[2]/55000.,\
    their_state.angular_velocity_body[0]/5.5,their_state.angular_velocity_body[1]/5.5,their_state.angular_velocity_body[2]/5.5,\
    \
    \
    bool_to_float(next_frame.boost_bool),bool_to_float(next_frame.jump_button_bool()),bool_to_float(next_frame.handbrake_bool),\
    # bool_to_float(next_frame.dodge_bool),\
    next_frame.throttle_float,\
    # next_frame.steer_float,\
    next_frame.rb_state.estimated_aerial_input[0],next_frame.rb_state.estimated_aerial_input[1],next_frame.rb_state.estimated_aerial_input[2],\
    # next_frame.dodge_y_float, next_frame.dodge_x_float*-1\
    ]

    return 1

def write_inputs(framenumber,output,count):
    frame = me.frames[framenumber]
    next_frame = me.frames[framenumber+1]
    my_state = me.frames[framenumber].rb_state

    if my_state is None or next_frame.rb_state is None:
        output[count] = [next_frame.delta,0,0,0,0,0,0,0]
        return 1

    if not my_state.is_in_aerial_box():
        next_frame.rb_state.estimated_aerial_input[2] = next_frame.steer_float
        next_frame.rb_state.estimated_aerial_input[0:2] = [0,0]
    else:
        next_frame.handbrake_bool = False
    if next_frame.dodge_bool:
        next_frame.rb_state.estimated_aerial_input[1] = next_frame.dodge_y_float
        next_frame.rb_state.estimated_aerial_input[2] = next_frame.dodge_x_float*-1
    if next_frame.double_jump_bool:
        next_frame.rb_state.estimated_aerial_input[1:3] = [0,0]

    if frame.boost_float == 0: next_frame.boost_bool = False

    output[count] = [next_frame.delta,\
    bool_to_float(next_frame.boost_bool),bool_to_float(next_frame.jump_button_bool()),bool_to_float(next_frame.handbrake_bool),\
    # bool_to_float(next_frame.dodge_bool),\
    next_frame.throttle_float,\
    # next_frame.steer_float,\
    next_frame.rb_state.estimated_aerial_input[0],next_frame.rb_state.estimated_aerial_input[1],next_frame.rb_state.estimated_aerial_input[2],\
    # next_frame.dodge_y_float, next_frame.dodge_x_float*-1\
    ]

    return 1


def process_replay(filename, script_player=None):

    global players
    global unclaimed_trackers
    players = []
    unclaimed_trackers = []

    global time_left
    global in_regulation
    time_left = 300
    in_regulation = True

    global me
    global opponent
    me = None
    opponent = None

    j = json.load(open(filename))
    all_frames = j['content']['body']['frames']

    global total_frames
    total_frames = len(all_frames)
    
    global ball
    ball = actor(total_frames)
    
    initialize_players(all_frames[0],total_frames,script_player)
    
    for framenumber, frame in enumerate(all_frames):
    
        read_frame_spawns(frame, framenumber)
        read_frame_states(frame, framenumber)

        delta = get_frame_delta(frame, framenumber)

        adopt_unclaimed_trackers(framenumber)
        assign_frame_trackers(frame,framenumber)

        update_boost_amounts(framenumber,delta)
        update_dodge_timers(framenumber,delta)
        
        read_frame_updates(frame,framenumber)

        fill_none_with_recent(framenumber)

        if framenumber == 0:
            define_teams(frame)
            ball.frames[0].rb_state = rb_state(0)
            ball.frames[0].rb_state.position = [0,0,9274]

    interpolate_rb_state_nones()
    estimate_all_inputs()
    interpolate_steer_nones()

    resolve_kickoff_states(total_frames)

    if script_player is not None:
        me = choose_me_username(script_player)
        if me is None: sys.exit(f'No player found with username {script_player}')
        count = 0
        output = np.zeros((50000,8),dtype=np.float16)
        for framenumber, frame in enumerate(all_frames):
            if framenumber < total_frames-1: count += write_inputs(framenumber,output,count)
        for i, row in enumerate(output):
            if np.array_equal(row,np.zeros(8)):
                output = output[0:i]
                return output
                
    else: # get both blue and orange perspectives
        me, opponent = choose_me_color('blue') # will swap automatically for second output

    count = 0
    output1 = np.zeros((50000,99),dtype=np.float16)
    for framenumber, frame in enumerate(all_frames):
        if framenumber < total_frames-1: count += write_frame(framenumber,output1,count)
    for i, row in enumerate(output1):
        if np.array_equal(row,np.zeros(99)):
            output1 = output1[0:i]
            break

    temp = me
    me = opponent
    opponent = temp

    count = 0
    output2 = np.zeros((50000,99),dtype=np.float16)
    for framenumber, frame in enumerate(all_frames):
        if framenumber < total_frames-1: count += write_frame(framenumber,output2,count)
    for i, row in enumerate(output2):
        if np.array_equal(row,np.zeros(99)):
            output2 = output2[0:i]
            break

    if output1.shape[1] == 0 or output2.shape[1] == 0: print('WARNING. NO DATA IN OUTPUT ARRAY. MAYBE YOU CHANGED THE OUTPUT SHAPE AND CUTOFF IS STUCK AT ZERO.')
    return np.concatenate((output1,output2),axis=0)
