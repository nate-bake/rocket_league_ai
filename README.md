# rocket_league_ai
My attempt at creating a Rocket League bot using a neural network trained exclusively on human replays

I understand that this has been attempted numerous times before and that the unique challenges of using human replay data are difficult to overcome, but I still wanted to try it and this is what I have. If you want to learn more about my process or see the performance of one of my models, visit <a href="https://natebake.dev/code/rl-ai" target="_blank">natebake.dev/code/rl-ai</a>.
<br><br>
## File Structure
| folder  | description                                                                                             |
| :------ | :------------------------------------------------------------------------------------------------------ |
| scripts | contains runnable code for collecting/processing replays as well as a jupyter notebook for training.    |
| data    | used to organize .replay files, .json files [parsed replays], and .npy files that comprise the dataset. |
| models  | for storing tensorflow models and checkpoints, which can be loaded by RLBot.                            |
| bot     | source code for my RLBot, which can follow a script of controller inputs or predict them using a model. |
<br>

## Dependencies
You will need Python 3 installed with NumPy, Tensorflow, and maybe some others. I don't remember.
This proejct also requires <a target="_blank" href="https://github.com/tfausak/rattletrap">rattletrap</a> for parsing and <a target="_blank" href="https://rlbot.org">RLBot</a> for integration with Rocket League.
<br><br>
## Instructions
Here is a brief explanation of how to run the scripts and what they do.
```sh
cd scripts/

# create a csv file containing a bunch of names of replays from the ballchasing.com API.
python3 fetch_ids.py

# get the replays in ids.csv and put them in data/new_replays. there's a limit on fetches in a certain time period.
python3 fetch_replays.py

# use rattletrap to create .json files in data/new_json for replay and move those to data/parsed_replays.
./parse_new_replays.sh

# read json files in parallel and make a numpy file in data/npy containing [game-state, outputs] rows for each frame.
python3 process_new_json.py

jupyter notebook
# train_model.ipynb is the notebook where i've tried training a bunch of models with various hyperparameters.
```
<br>

## Notes
I used these scripts to collect and process about 15,000 replays of gold/silver 1v1 matches from <a target="_blank" href="https://ballchasing.com">ballchasing.com</a>, which yielded a dataset of about 250 million rows [50 GB].
<br><br>
I included a few replays in the data folders just so that GitHub would not delete the folders, but if you try to train on just these your model will immediately overfit.
<br><br>
I also added one of my old .h5 files which you can test in game if you create your own <a target="_blank" href="https://rlbot.org">RLBot</a>.
<br><br>
## Resources
- <a target="_blank" href="https://samuelpmish.github.io/notes/RocketLeague/">Sam Mish's Notes and Code</a> are extremely useful for estimating aerial inputs from replay files, and I can't take credit for any of it.
- The <a target="_blank" href="https://discord.com/invite/xuWjbw7A?utm_source=Discord%20Widget&utm_medium=Connect">RLBot Discord</a> is also a terrific resource if you have any interest in making some sort of Rocket League bot. Lots of incredibly knowledgable programmers and friendly folks!
<br><br>
