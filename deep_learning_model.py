from header_import import *

class DeepQLearning(object):
    def __init__ (self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = 0.001
        self.model = self.create_model()

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=MEMORY_REPLAY)

        self.tensorboard = model_train(log_dir="logs/{}-{}".format("Q_Learning", int(time.time())))
        self.target_update_counter = 0.001
