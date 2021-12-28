from header_import import *


class actor_critic(MountainCar3D):
    def __init__(self):
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
        
    
    def actor_critic_with_eligibility_traces(self, gamma = 0.1, lambda_theta = 0.0005, lambda_weight = 0.0005, alpha_theta = 2**-9, alpha_weight = 2**-6, number_of_episodes = 100):

        for episode in tqdm(range(number_of_episodes), desc = "Episode"):
            step = 1
            reward_count = 0
            reached_goal = False
            state = -0.5
            velocity = 0.01
            I = 1
            
            while not reached_goal:
                old_state = np.append(1, state)
                action_probability = self.action_softmax(old_state)
                action = self.policy(action_probability)
                next_state, velocity, reward, reached_goal =  self.step(action)
                new_state = np.append(1, next_state)
                reward_count += reward
                
                if reached_goal:
                    delta = reward - np.dot(old_state, self.weight)
                else:
                    delta = reward + gamma *(np.dot(new_state, self.weight)) - (np.dot(old_state, self.weight))
                
                self.Z_weight = gamma * lambda_weight * self.Z_weight + delta * (np.dot(old_state, self.weight))
                self.Z_theta = gamma * lambda_theta * self.Z_theta + I * delta * self.action_softmax(old_state)
                self.weight += alpha_weight * delta * self.Z_weight
                self.theta += alpha_theta * delta * self.Z_theta

                I = gamma * I
                state = next_state
                step +=1

            self.step_per_episode.append(step)
            self.total_rewards.append(reward_count)
    
        return self.weight, self.step_per_episode, self.total_rewards
