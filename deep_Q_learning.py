from header_import import *


class deep_q_learning(object):
    def __init__(self):
        self.episode = 400
        self.epsilon = 1
        self.delay_epsilon = 0.99975
        self.min_epsilon = 0.001

        self.low_state_bound = np.array([-1.2, -0.07])
        self.high_state_bound = np.array([0.6, 0.07])

        self.normalize = np.subtract(high_state_bound, low_state_bound)
        self.state_dim = mountaincar3d.state_shape
        self.state_dim_range = mountaincar3d.state_shape_range
        self.state_size = (4,)
        self.action_size = 5

        self.state_space = 400
        self.position_range = np.linspace(self.low_state_bound[0], self.high_state_bound[0], self.state_space)
        self.velocity_range = np.linspace(self.low_state_bound[1], self.high_state_bound[1], int(self.state_space/5))
        self.episode_rewards = []
        self.step_per_episode = []


    def choose_action(current_state, action_size):
        if np.random.random() > self.epsilon:
            return np.argmax(deep_q_learning.get_qs(current_state))
        else:
            return np.random.randint(0, action_size)


def deep_q_learning(self):

    for episode in tqdm(range(1, self.episode+1), unit="episode"):
        deep_q_learning.tensorboard.step = episode
        step = 1
        current_state = mountaincar3d.enviroment_start()
        reached_goal = False
        episode_reward = 0

        while not reached_goal and step <= 500:
            action = choose_action(current_state, action_size)
            action, reward, next_state, reached_goal = mountaincar3d.enviroment_step(action)
            episode_reward += reward

            deep_q_learning.update_replay_memory((current_state, action, reward, next_state, reached_goal))
            deep_q_learning.train(reached_goal, step)

            current_state = next_state
            step += 1

        self.step_per_episode.append(step)
        self.episode_rewards.append(episode_reward)
