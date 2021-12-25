from header_import import *


class model_train(TensorBoard):
    def __init__(self, **kwargs):
        self.step = 1

    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DeepQLearning(object):
    def __init__ (self, observation_space, action_space):
        
        self.memory_delay = 50000
        self.batch = 32
        self.discount_factor = 0.99
        self.target_update = 5

        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = 0.001
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen = self.memory_delay)
        self.tensorboard = model_train(log_dir="logs/{}-{}".format("Q_Learning", int(time.time())))
        self.target_update_counter = 0.001


    def create_model(self):
        self.model = Sequential()
        self.model.add(Dense(64,input_shape=self.observation_space))
        self.model.add(Activation("relu"))
        self.model.add(Dense(64))
        self.model.add(Activation("relu"))
        self.model.add(Dense(self.action_space, activation="relu"))
        self.model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])


    def update_replay_memory (self, transition):
        self.replay_memory.append(transition)


    def get_q_value(self, state):
        state = np.array((state).reshape(-1, *state.shape))
        return self.model.predict(state)[0]


    def train(self, terminnal_state):
        
        X = []
        Y = []

        if len(self.replay_memory) < (self.memory_delay/5):
            return

        batch = random.sample(self.replay_memory, self.batch)
        current_states = np.array([transition[0] for transition in batch]) 
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in batch])
        future_qs_list = self.target_model.predict(new_current_states)

        for index, (state, action, reward, next_state, done) in enumerate(batch):
            if not done:
                q_value = reward + self.discount_factor *  np.max(future_qs_list[index])
            else:
                q_value = reward
        
            current_qs = current_qs_list[index]
            current_qs[action] = q_value

            X.append(state)
            Y.append(current_qs)
        
        self.model.fit(np.array(X), np.array(Y), batch_size=self.batch, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminnal_state else None)

        if terminnal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.target_update:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


    def save_model(self, max_reward, min_reward, average_reward):
        self.model.save(f'models/{"Q_Learning"}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')



