# rocket_league_ai
My attempt at creating a Rocket League bot using a neural network trained exclusively on human replays

I understand that this has been attempted numerous times before and that the unique challenges of using human replay data are difficult to overcome, but I still wanted to try it and this is what I have. If you want to learn more about my process or see the performance of one of my models, visit [natebake.dev/code/rl-ai].

**File Structure**
- scripts :: contains runnable code for collecting/processing replays as well as a jupyter notebook for training.
- data :: used to organize .replay files, .json files [parsed replays], and .npy files that comprise the dataset.
- models :: for storing tensorflow models and checkpoints, which can be loaded by RLBot.
- bot :: the source code for my RLBot, which can predict outputs using a model or follow a script of controller inputs.

**Instructions**
Here is a brief explanation of how to run the scripts and what they do.
>`python3 fetch_ids.py` creates a csv file containing a bunch of names of replays from the ballchasing.com API.
>`python3 fetch_replays.py` gets the replays listed in ids.csv and places them in data/new_replays. there is a limit on how many can be fetched in a certain time period.
>`./parse_new_replays.sh` uses rattletrap to create .json files for each replay and puts them in data/new_json. replays are moved to data/parsed_replays.
>`python3 process_new_json.py` reads json files in parallel and puts a numpy file in data/npy which contains a row of game-state, outputs for each frame.
>`train_model.ipynb` is my jupyter notebook where i have tried training a bunch of models with various hyperparameters.
