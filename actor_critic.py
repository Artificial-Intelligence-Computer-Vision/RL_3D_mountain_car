from header_import import *


class actor_critic(object):
    def __init__(self):
        
        mountaincar3d = MountainCar3D()
        self.episode_rewards = []
        self.step_per_episode = []
        self.total_rewards = []
        self.record_last_state = []


    def actor_critic_with_eligibility_traces(gamma = 0.1, lambda_theta = 0.0005, lambda_weight = 0.0005, alpha_theta = 2**-9, alpha_weight = 2**-6, number_of_episodes = 25):

        weight = np.zeros(5)
        theta = np.zeros(5)

        for episode in tqdm(range(0, number_of_episodes), unit="episode"):
            step = 1
            reward_count = 0
            reached_goal = False
            current_state = mountaincar3d.enviroment_start()
            Z_weight = np.zeros(weight.size)
            Z_theta = np.zeros(theta.size)
            I = 1

            while not reached_goal:
                old_state = np.append(1,current_state)
                action_probability = action_softmax(old_state)
                action = choose_action(action_probability)
                action, reward, next_state, reached_goal = mountaincar3d.enviroment_step(action)
                new_state = np.append(1, next_state)

                if episode == number_of_episodes-1:
                    record_last_state.append(current_state)

                reward_count -= reward
                if reached_goal:
                    print("reached_goal")
                    delta = reward - np.dot(old_state, weight)
                else:
                    delta = reward + gamma *(np.dot(new_state, weight)) - (np.dot(old_state, weight))

                Z_weight = gamma * lambda_weight * Z_weight + delta * (np.dot(old_state, weight))
                Z_theta = gamma * lambda_theta * Z_theta + I * delta * action_softmax(old_state)
                weight += alpha_weight * delta * Z_weight
                theta += alpha_theta * delta * Z_theta

                I = gamma * I
                current_state = next_state
                step +=1

            episode_rewards.append(reward)
            step_per_episode.append(step)
            total_rewards.append(reward_count)
    
        return theta, weight, episode_rewards, step_per_episode, total_rewards, record_last_state




    def action_softmax(x_feature_vector):
        return np.exp(x_feature_vector - np.max(x_feature_vector)) / (np.exp(x_feature_vector - np.max(x_feature_vector))).sum(axis=0)


    def choose_action(vector):
        action_vector = np.zeros(vector.size)
        action = random.random()
        for i in range(vector.size):
            action_vector = np.sum(vector[:i+1])
            if action >= vector[i]:
                return i
            else:
                return np.random.randint(0, 5)




