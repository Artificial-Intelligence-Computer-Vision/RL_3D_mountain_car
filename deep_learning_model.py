from header_import import *


class DeepQLearning(MountainCar3D, TensorBoard):
    def __init__ (self, observation_space=(4,), action_space=5, dense_size = 24, batch_size=32, previous_model_path = "none", algorithm_name = "Deep_q_learning_experience_replay"):
        super().__init__()

        self.memory_delay = 50000
        self.batch = batch_size
        self.dense_size = dense_size
        self.algorithm_name = algorithm_name
        self.gamma = 0.95
        self.target_update = 5
        self.previous_model_path = previous_model_path
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = 0.001
        self.epochs = [1, 5, 15, 50, 100, 200]
        self.model_path = "models/" + self.algorithm_name + "_model.h5"
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999), metrics=["accuracy"]
        
        if previous_model_path == "none":
            self.model = self.create_model()
        else:
            self.model.load(self.previous_model_path)

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen = self.memory_delay)
        self.callback_1 = TensorBoard(log_dir="logs/{}-{}".format(self.algorithm_name, int(time.time())))
        self.callback_2 = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path, save_weights_only=True, verbose=1)
        self.callback_3 = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor= 0.5, min_lr=0.00001)
        self.target_update_counter = 0.001


    def create_model(self):

        model = Sequential()
        model.add(Dense(self.dense_size,input_shape=self.observation_space, activation="relu"))
        model.add(Dense(self.dense_size, activation="relu"))
        model.add(Dense(self.action_space, activation="relu"))
        model.compile(loss="mse", optimizer=self.optimizer)

        return model


    def update_replay_memory (self, transition):
        self.replay_memory.append(transition)


    def get_q_values(self, state):
        state = np.array((state).reshape(-1, *state.shape))
        return self.model.predict(state)[0]


    def train(self, reached_goal):
        
        X = []
        Y = []

        if len(self.replay_memory) > (self.batch):
            return

        batch = random.sample(self.replay_memory, self.batch)
        current_states = np.array([transition[0] for transition in batch]) 
        current_state_value = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in batch])
        target_state_value = self.target_model.predict(new_current_states)

        for index, (state, action, reward, next_state, done) in enumerate(batch):
            if not done:
                state_value = reward + self.gamma *  np.max(target_state_value[index])
            else:
                state_value = reward
        
            current_q_value = current_state_value[index]
            current_q_value[action] = state_value

            X.append(state)
            Y.append(current_q_value)
        
        self.model.fit(np.array(X), np.array(Y), 
            batch_size=self.batch, 
            verbose=0, 
            epochs=self.epochs[1], 
            shuffle=False, 
            callbacks=[self.callback_1, self.callback_2, self.callback_3] if reached_goal else None)

        if reached_goal:
            self.target_update_counter += 1

        if self.target_update_counter > self.target_update:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


    def save_model(self):
        self.model.save(self.model_path)
