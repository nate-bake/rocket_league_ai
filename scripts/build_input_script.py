import os, sys
import json_processor
import struct

json_file = sys.argv[1] 
path = json_file
if not os.path.isfile(path):
    sys.exit(f'THE FILE \'{json_file}\' COULD NOT BE FOUND')

username = sys.argv[2] # player to get inputs from

output = json_processor.process_replay(json_file, script_player=username)
out = open("../bot/script.bin","wb")

for frame in output:
    out.write(struct.pack('f',frame[0]))
    out.write(struct.pack('?',frame[1]))
    out.write(struct.pack('?',frame[2]))
    out.write(struct.pack('?',frame[3]))
    out.write(struct.pack('f',frame[4]))
    out.write(struct.pack('f',frame[5]))
    out.write(struct.pack('f',frame[6]))
    out.write(struct.pack('f',frame[7]))

out.close()