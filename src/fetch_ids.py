import requests
import json
import numpy as np
import os, sys

FOLDER = sys.argv[1]
path = f'/mnt/d/rl_ai/data/{FOLDER}/'
if not os.path.isdir(path):
    sys.exit(f'THE FOLDER \'{FOLDER}\' COULD NOT BE FOUND')

url = 'https://ballchasing.com/api/replays/ae92d578-6faa-4f7d-82ae-511fcd0fbadc/file'
token = 'vYg4neVZT4ppbCELiU6kZKYu9EEdFMciRG0tUl5g'

ids = []

last_date = '2021-01-02T15:00:05Z'

for i in range(5):
    r = requests.get('https://ballchasing.com/api/replays', headers={'Authorization': token}, params={
                     'playlist': 'ranked-duels', 'min-rank': f'{FOLDER}-1', 'max-rank': f'{FOLDER}-3', 'replay-date-before': last_date, 'sort-by': 'replay-date', 'count': '200'})

    j = json.loads(r.content.decode('utf-8'))
    # if i > 48:
    for item in j['list']:
        ids.append(item['id'])
    print(len(j['list']))
    last_date = (j['list'][len(j['list'])-1]['date'])
    print(last_date)
    # if last_date.startswith('2020-08-04'): break

print(len(ids))
for entry in os.scandir(f'/mnt/d/rl_ai/data/{FOLDER}/parsed_replays'):
    id = entry.path.split('/')[7].split('.')[0]
    print(id)
    if id in ids:
        ids.remove(id)
for entry in os.scandir(f'/mnt/d/rl_ai/data/{FOLDER}/new_replays'):
    id = entry.path.split('/')[7].split('.')[0]
    print(id)
    if id in ids:
        ids.remove(id)
print(len(ids))
np_ids = np.array(ids)
np.savetxt(f'/mnt/d/rl_ai/data/{FOLDER}/ids.csv', ids, fmt="%s")
