import requests
import numpy as np
import time
import os

token = 'vYg4neVZT4ppbCELiU6kZKYu9EEdFMciRG0tUl5g'

ids = np.loadtxt(f'../data/ids.csv', dtype='str')
print(len(ids))

for id in ids[:200]:
    r = requests.get('https://ballchasing.com/api/replays/' +
                     id+'/file', headers={'Authorization': token})

    with open('../data/new_replays/'+id+'.replay', 'wb') as f:
        f.write(r.content)

    print(id)
    size = os.stat('../data/new_replays/'+id+'.replay').st_size
    if size < 409600:
        os.remove('../data/new_replays/'+id+'.replay')
        print('less than 400kb. discarding...')
    if size < 1024:
        print('less than 1kb. request limit has likely been reached.')
        break
    ids = ids[1:]
    time.sleep(1)
    
np.savetxt('../data/ids.csv', ids, fmt="%s")
print(f'saved \'../data/ids.csv\' with {len(ids)} remaining ids')