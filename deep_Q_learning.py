from header_import import *


class deep_q_learning_algorithm(DeepQLearning):
    def __init__(self, episode, epsilon, noise = 0.0, reward_noise = 0.0, random_start = False):
        super().__init__()

        self.episode = episode
        self.epsilon = epsilon
        self.delay_epsilon = 0.99975
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


    def deep_q_learning(self):
    
        for episode in tqdm(range(1, self.episode+1), mininterval=100, unit="episode", desc="episode"):
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

            self.step_per_episode.append(step)
            self.episode_rewards.append(episode_reward)
