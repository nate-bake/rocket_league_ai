import requests
import numpy as np
import time
import os, sys

token = 'vYg4neVZT4ppbCELiU6kZKYu9EEdFMciRG0tUl5g'

FOLDER = sys.argv[1]
path = f'/mnt/d/rl_ai/data/{FOLDER}/'
if not os.path.isdir(path):
    sys.exit(f'THE FOLDER \'{FOLDER}\' COULD NOT BE FOUND')

ids = np.loadtxt(f'/mnt/d/rl_ai/data/{FOLDER}/ids.csv', dtype='str')
print(len(ids))

for id in ids[:200]:
    r = requests.get('https://ballchasing.com/api/replays/' +
                     id+'/file', headers={'Authorization': token})

    with open(f'/mnt/d/rl_ai/data/{FOLDER}/new_replays/'+id+'.replay', 'wb') as f:
        f.write(r.content)

    print(id)
    size = os.stat(f'/mnt/d/rl_ai/data/{FOLDER}/new_replays/'+id+'.replay').st_size
    if size < 409600:
        os.remove(f'/mnt/d/rl_ai/data/{FOLDER}/new_replays/'+id+'.replay')
        print('less than 400kb. discarding...')
    if size < 1024:
        print('less than 1kb. request limit has likely been reached.')
        break
    ids = ids[1:]
    time.sleep(1)
    
print(len(ids))
np.savetxt(f'/mnt/d/rl_ai/data/{FOLDER}/ids.csv', ids, fmt="%s")
