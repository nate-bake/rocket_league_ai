for f in ../data/new_replays/*.replay
do
 rattletrap -c -i "$f" -o "${f/.replay/.json}"
done
mv ../data/new_replays/*.json ../data/new_json/
mv ../data/new_replays/*.replay ../data/parsed_replays/
