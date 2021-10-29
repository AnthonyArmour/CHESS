# Asynchronous Actor Critic Guided Monte Carlo Tree Search Self Play Chess Algorithm. Distributed On a 3 Piece Amazon Elastic Compute Cloud Framework.

## Project Information
    This repository includes  APIs existing in 2 easily distributable folders. The MidServer folder, to be deployed on a light weight AWS server, has an app for storing states, actions, rewards, and weights. It includes memory replay training, and sends requests to compute servers for game information. The ComputeServer folder, to be deployed on 2 separate compute optimized AWS servers (5 cpu, 32 GB), includes an app strictly for running games using and Asynchronous Actor Critic guided Monte Carlo Tree Search algorithm on 3 separate parallel threads. A total of six series of games on six different cpu threads are available between the two compute optimized servers. These games return byte encoded data back to the MidServer for training. The Local folder includes a main script for instructing the MidServer.

## MidServer
    Hosted on an 8 GB 4 cpu Amazon Elastic Compute Cloud server. Responsible for making requests to compute servers, keeping model data, and training model weights. Includes a flask API which sends byte encoded model weights to compute servers for game play. The compute servers return byte encoded arrays of data including the states, actions, rewards, wins, and losses from the played games. After ample data is collected, the MidServer proceeds to train weights of identical dimensions on a random sample of states, actions, and rewards from its memory bank. The trained weights are saved and ready to be loaded into the compute server models for the next round of game playing. All the dependecies are included in the packages_setup file, aside from the python3 installation.

## Compute Servers
    Each hosted on a 32 GB 5 cpu AWS EC2 server. Responsible for running the policy and value network guided monte carlo tree search self play chess algorithm. Includes API which recieves model weights from request and uploads them before executing the instructed number of games. The x number of games are run across 3 different threads and the state, action, reward information is joined and returned to the MidServer. Between the two compute optimized servers, a total of six threads worth of work is returned to the MidServer. All the dependecies are included in the packages_setup file, aside from the python3 installation.


## Architecture

    ----------------------------------                ----------------------------------
    |                                |                |                                |
    |         Compute Server         |                |         Compute Server         |
    |                                |                |                                |
    |    Runs Games on 3 threads     |                |    Runs Games on 3 threads     |
    |                                |                |                                |
    |                                |                |                                |
    ----------------------------------                ----------------------------------
                    |                                                   |
                    |                                                   |
                    |                                                   |
                    |                                                   |
                    |___________________________________________________|
                                            |
                                            |
                                            |
                                            |
                            -----------------------------------
                            |                                 |
                            |            MidServer            |
                            |                                 |
                            |   Parallel requests to server.  |
                            |      Trains on information.     |
                            |        Stores information.      |
                            |                                 |
                            -----------------------------------
                                            |
                                            |
                                            |
                                            |
                                            |
                            ------------------------------------
                            |                                  |
                            |           Local Mac              |
                            |                                  |
                            |    Requests training session     |
                            |          from MidServer.         |
                            |    Recieves optimized weights    |
                            |           upon request.          |
                            |                                  |
                            ------------------------------------