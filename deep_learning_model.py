from header_import import *


class DeepQLearning(MountainCar3D, TensorBoard):
    def __init__ (self, observation_space=(4,), action_space=5):
        super().__init__()

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
        self.tensorboard = TensorBoard(log_dir="logs/{}-{}".format("Q_Learning", int(time.time())))
        self.target_update_counter = 0.001


    def create_model(self):

        model = Sequential()
        model.add(Dense(64,input_shape=self.observation_space))
        model.add(Activation("relu"))
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dense(self.action_space, activation="relu"))
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])

        return model


    def update_replay_memory (self, transition):
        self.replay_memory.append(transition)


    def get_q_values(self, state):
        state = np.array((state).reshape(-1, *state.shape))
        return self.model.predict(state)[0]


    def train(self, reached_goal):
        
        X = []
        Y = []

        if len(self.replay_memory) < (self.memory_delay/5):
            return

        batch = random.sample(self.replay_memory, self.batch)
        current_states = np.array([transition[0] for transition in batch]) 
        current_state_value = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in batch])
        target_state_value = self.target_model.predict(new_current_states)

        for index, (state, action, reward, next_state, done) in enumerate(batch):
            if not done:
                state_value = reward + self.discount_factor *  np.max(target_state_value[index])
            else:
                state_value = reward
        
            current_q_value = current_state_value[index]
            current_q_value[action] = state_value

            X.append(state)
            Y.append(current_q_value)
        
        self.model.fit(np.array(X), np.array(Y), batch_size=self.batch, verbose=0, shuffle=False, callbacks=[self.tensorboard] if reached_goal else None)

        if reached_goal:
            self.target_update_counter += 1

        if self.target_update_counter > self.target_update:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


    def save_model(self):
        self.model.save("models/" +"Deep_q_learning"+"_model.h5")
