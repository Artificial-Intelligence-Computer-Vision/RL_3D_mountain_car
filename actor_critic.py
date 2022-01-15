from header_import import *


class actor_critic(MountainCar3D, plot_graphs):
    def __init__(self, gamma = 0.5, lambda_theta = 0.0005, lambda_weight = 0.0005, alpha_theta = 2**-9, alpha_weight = 2**-6, episode = 10000, algorithm_name="actor_critic"):
        super().__init__()

        self.path = "graphs_charts/"
        self.enviroment_path = self.path + "enviroment_details/"
        self.model_detail_path = self.path + "model_details/"
        self.algorithm_name = algorithm_name
        self.gamma = gamma
        self.lambda_theta = lambda_theta
        self.lambda_weight = lambda_weight
        self.alpha_theta = alpha_theta
        self.alpha_weight = alpha_weight

        self.epsilon = 1
        self.delay_epsilon = 0.999
        self.min_epsilon = 0.001
        self.episode = episode
        self.action_size = 5
        self.weight = np.zeros(self.action_size)
        self.theta = np.zeros(self.action_size)
        self.step_per_episode = []
        self.total_rewards = []
        self.Z_weight = np.zeros(self.weight.size)
        self.Z_theta = np.zeros(self.theta.size)
   

    def action_softmax(self, x_feature_vector):
        return np.exp(x_feature_vector - np.max(x_feature_vector)) / (np.exp(x_feature_vector - np.max(x_feature_vector))).sum(axis=0)
    
    def epsilon_reduction(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.delay_epsilon

    def policy(self, state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.action_softmax(state))
        else:
            return np.random.randint(0, self.action_size)


    def actor_critic_with_eligibility_traces(self):

        for episode in tqdm(range(self.episode), desc = "Episode"):
            step = 0
            reward_count = 0
            state = self.reset()
            reached_goal = False
            I = 1
            state = np.append(0, state)

            while not reached_goal:
                action = self.policy(state)
                action, reward, next_state, reached_goal = self.step(action)
                reward_count += reward
                next_state = np.append(0, next_state)
                
                if reached_goal:
                    delta = reward - np.dot(state, self.weight)
                    break
                else:
                    delta = reward + self.gamma *(np.dot(next_state, self.weight)) - (np.dot(state, self.weight))
                
                self.Z_weight = self.gamma * self.lambda_weight * self.Z_weight + delta * (np.dot(state, self.weight))
                self.Z_theta = self.gamma * self.lambda_theta * self.Z_theta + I * delta * self.action_softmax(state)
                self.weight += self.alpha_weight * delta * self.Z_weight
                self.theta += self.alpha_theta * delta * self.Z_theta
                I = self.gamma * I
                state = next_state
                step +=1

            self.epsilon_reduction()
            self.step_per_episode.append(step)
            self.total_rewards.append(reward_count)

        self.plot_episode_time_step(self.total_rewards, type_graph="cumulative_reward")
        self.plot_episode_time_step(self.step_per_episode, type_graph = "step_number")
