from VersionControl.RoboRL6 import TrainRoboRL

Model = TrainRoboRL(episodes=5000, version="V2", loadfile=False)
Model.RunTrainingSession()
