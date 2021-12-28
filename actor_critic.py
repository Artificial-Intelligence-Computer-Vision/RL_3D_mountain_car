from header_import import *


class actor_critic(MountainCar3D):
    def __init__(self):
        super().__init__()
        self.number_of_action = 5
        self.weight = np.zeros(self.number_of_action)
        self.theta = np.zeros(self.number_of_action)
        self.step_per_episode = []
        self.total_rewards = []
        self.Z_weight = np.zeros(self.weight.size)
        self.Z_theta = np.zeros(self.theta.size)
    
    def action_softmax(self, x_feature_vector):
        return np.exp(x_feature_vector - np.max(x_feature_vector)) / (np.exp(x_feature_vector - np.max(x_feature_vector))).sum(axis=0)

    def policy(self, vector):
        action_vector = np.zeros(vector.size)
        action = random.random()
        
        for i in range(vector.size):
            action_vector = np.sum(vector[:i+1])
            if action >= vector[i]:
                return i
            else:
                return np.random.randint(0, 5)
        
    
    def actor_critic_with_eligibility_traces(self, gamma = 1, lambda_theta = 0.0005, lambda_weight = 0.0005, alpha_theta = 2**-9, alpha_weight = 2**-6, number_of_episodes = 10000):

        for episode in tqdm(range(number_of_episodes), desc = "Episode"):
            step = 0
            reward_count = 0
            state = self.reset()
            reached_goal = False
            I = 1
            state = np.append(0, state)
            while not reached_goal:
                action_probability = self.action_softmax(state)
                action = self.policy(action_probability)
                action, reward, next_state, reached_goal = self.step(action)
                reward_count += reward
                next_state = np.append(0, next_state)
                
                if reached_goal:
                    delta = reward - np.dot(state, self.weight)
                else:
                    delta = reward + gamma *(np.dot(next_state, self.weight)) - (np.dot(state, self.weight))
                
                self.Z_weight = gamma * lambda_weight * self.Z_weight + delta * (np.dot(state, self.weight))
                self.Z_theta = gamma * lambda_theta * self.Z_theta + I * delta * self.action_softmax(state)
                self.weight += alpha_weight * delta * self.Z_weight
                self.theta += alpha_theta * delta * self.Z_theta
                I = gamma * I
                state = next_state
                step +=1
            self.step_per_episode.append(step)
            self.total_rewards.append(reward_count)
    
        return self.weight, self.step_per_episode, self.total_rewards
