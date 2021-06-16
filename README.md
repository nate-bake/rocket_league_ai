# rocket_league_ai
my attempt at creating a rocket league bot using only ML from human replays


many brilliant members of the rl_bot community [https://discord.gg/uFW4sDVSs3] have all but concluded that training a model to play rocket league based solely on replays of human gameplay is not really feasible [due to many factors including human inconsistency and the lack of detail and accuracy in the game's existing replay system.]
i was not aware of these conclusions when i embarked on this project, but i still saw some results that i felt were encouraging so i decided to share my work here.

i ended up collecting about 15,000 1v1 replays in the silver and gold rank categories, and processed them from the perspective of both players. this yielded a dataset of around 50 gigabytes with 250 million rows, upon which to train. obviously i have not uploaded all the data here, only about 10 replays to test with.

this repository contains the full scope of my endeavor. from automating replay collection from ballchasing.com, to parsing these replays and compiling a dataset, to training a model on the data via tensorflow, all the way to connecting the model to the game using rl_bot.
