for f in /mnt/d/rl_ai/data/silver/new_replays/*.replay
do
 rattletrap -c -i "$f" -o "${f/.replay/.json}"
done
mv /mnt/d/rl_ai/data/silver/new_replays/*.json /mnt/d/rl_ai/data/silver/next_json/
mv /mnt/d/rl_ai/data/silver/new_replays/*.replay /mnt/d/rl_ai/data/silver/parsed_replays/
