import requests
import json
import numpy as np
import os

url = 'https://ballchasing.com/api/replays/ae92d578-6faa-4f7d-82ae-511fcd0fbadc/file'
token = 'vYg4neVZT4ppbCELiU6kZKYu9EEdFMciRG0tUl5g'

last_date = '2021-01-02T15:00:05Z'

ids = []

for i in range(5): # 200 replays per iteration
    r = requests.get('https://ballchasing.com/api/replays', headers={'Authorization': token}, params={
                     'playlist': 'ranked-duels', 'min-rank': 'silver-1', 'max-rank': 'gold-3', 'replay-date-before': last_date, 'sort-by': 'replay-date', 'count': '200'})

    j = json.loads(r.content.decode('utf-8'))
    for item in j['list']:
        ids.append(item['id'])
    print(len(j['list']))
    last_date = (j['list'][len(j['list'])-1]['date'])
    print(last_date)

print(len(ids))
for entry in os.scandir('../data/parsed_replays'):
    id = entry.path.split('parsed_replays/')[1].split('.')[0]
    if id in ids:
        print(f'already have {id}')
        ids.remove(id)
for entry in os.scandir('../data/new_replays'):
    id = entry.path.split('new_replays/')[1].split('.')[0]
    if id in ids:
        print(f'already have {id}')
        ids.remove(id)
np_ids = np.array(ids)
np.savetxt('../data/ids.csv', ids, fmt="%s")
print(f'saved \'../data/ids.csv\' with length {len(ids)}')
