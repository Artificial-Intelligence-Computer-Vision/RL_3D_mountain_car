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


    def create_model(self):

        self.model = Sequential()
        self.model.add(Dense(64,input_shape=self.observation_space))
        self.model.add(Activation("relu"))
        self.model.add(Dense(64))
        self.model.add(Activation("relu"))
        self.model.add(Dense(self.action_space, activation="relu"))
        self.model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])


    def train(self, terminnal_state, step):
        if len(self.replay_memory) < (MEMORY_REPLAY/5):
            return

        batch = random.sample(self.replay_memory, BATCH)
        current_states = np.array([transition[0] for transition in batch]) 
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in batch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        Y = []
        
        for index, (current_state, action, reward, new_current_state, done) in enumerate(batch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount_factor * max_future_q
            else:
                new_q = reward
        
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            Y.append(current_qs)
        
        self.model.fit(np.array(X), np.array(Y), batch_size=BATCH, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminnal_state else None)

        if terminnal_state:
            self.target_update_counter += 1

        if self.target_update_counter > TARGET_UPDATE:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


    def save_model(self, max_reward, min_reward, average_reward):
        self.model.save(f'models/{"Q_Learning"}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')



