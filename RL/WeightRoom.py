# from VersionControl.RoboRL6 import TrainRoboRL
# from RoboRL7 import TrainRoboRL

# Model = TrainRoboRL(episodes=500, version="V2", loadfile=False)
# Model.RunTrainingSession()


# if __name__ == "__main__":
#     env_name = 'Pong-v0'
#     agent = A3CAgent(env_name)
#     #agent.run() # use as A2C
#     agent.train(n_threads=5) # use as A3C
#     #agent.test('Pong-v0_A3C_2.5e-05_Actor.h5', 'Pong-v0_A3C_2.5e-05_Critic.h5')

from RoboA2C_Mcts import A3CAgent
Model = A3CAgent(episodes=1000, version="MCTS_Mem_Replay", loadfile=False)

Model.TrainingSession()

# Model = A3CAgent(episodes=300, version="V3", loadfile=True)
# Model.TrainingSession(test=True, oponent_benchmark="Op_Benchmark_1")
