from header_import import *


class deep_q_learning_algorithm(DeepQLearning):
    def __init__(self, episode, noise=0.0, reward_noise=0.0, random_start=False, algorithm_name = "Deep_q_learning_experience_replay"):
        super().__init__(algorithm_name = algorithm_name)

        self.episode = episode
        self.epsilon = 1
        self.delay_epsilon = 0.995
        self.min_epsilon = 0.001
        self.low_state_bound = np.array([-1.2, -0.07])
        self.high_state_bound = np.array([0.6, 0.07])
        self.normalize = np.subtract(self.high_state_bound, self.low_state_bound)
        self.state_size = (4,)
        self.action_size = 5
        self.state_space = 400
        self.position_range = np.linspace(self.low_state_bound[0], self.high_state_bound[0], self.state_space)
        self.velocity_range = np.linspace(self.low_state_bound[1], self.high_state_bound[1], int(self.state_space/5))
        self.episode_rewards = []
        self.step_per_episode = []


    def policy(self, state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.get_q_values(state))
        else:
            return np.random.randint(0, self.action_size)


    def epsilon_reduction(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.delay_epsilon


    def deep_q_learning(self):
    
        for episode in tqdm(range(1, self.episode+1), desc="episode"):
            step = 0
            state = self.reset()
            reached_goal = False
            episode_reward = 0

            while not reached_goal:
                action = self.policy(state)
                action, reward, next_state, reached_goal = self.step(action)
                episode_reward += reward
                self.update_replay_memory((state, action, reward, next_state, reached_goal))
                self.train(reached_goal)
                state = next_state
                step += 1

            self.epsilon_reduction()
            self.step_per_episode.append(step)
            self.episode_rewards.append(episode_reward)

        self.save_model()
        return self.step_per_episode, self.episode_rewards



    def double_deep_q_learning(self):

        for episode in tqdm(range(1, self.episode+1), desc="episode"):
            step = 0
            state = self.reset()
            reached_goal = False
            episode_reward = 0

            while not reached_goal:
                action = self.policy(state)
                action, reward, next_state, reached_goal = self.step(action)
                episode_reward += reward
                self.update_replay_memory((state, action, reward, next_state, reached_goal))
                self.train(reached_goal)
                state = next_state
                step += 1
            
            self.epsilon_reduction()
            self.step_per_episode.append(step)
            self.episode_rewards.append(episode_reward)

        self.save_model()
        return self.step_per_episode, self.episode_rewards


    def dual_deep_q_learning(self):

        for episode in tqdm(range(1, self.episode+1), desc="episode"):
            step = 0
            state = self.reset()
            reached_goal = False
            episode_reward = 0

            while not reached_goal:
                action = self.policy(state)
                action, reward, next_state, reached_goal = self.step(action)
                episode_reward += reward
                self.update_replay_memory((state, action, reward, next_state, reached_goal))
                self.train(reached_goal)
                state = next_state
                step += 1

            self.epsilon_reduction()
            self.step_per_episode.append(step)
            self.episode_rewards.append(episode_reward)

        self.save_model()
        return self.step_per_episode, self.episode_rewards


        

