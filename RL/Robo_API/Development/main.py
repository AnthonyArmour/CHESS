from Learning_Zone import LearningZone


LZ = LearningZone()
# LZ.save_weights()
# LZ.Actor.save_weights("../Weights/Coach")
wins, losses = LZ.Collect_Games("18.217.42.216", "5000", episodes=1, mcts_iterations=10, load=0)