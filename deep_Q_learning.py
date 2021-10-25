from header_import import *


class deep_q_learning(object):
    def __init__(self):
        EPISODE = 400
        EPSILON = 1
        EPSILON_DECAY = 0.99975
        MIN_EPSILON = 0.001

        low_state_bound = np.array([-1.2, -0.07])
        high_state_bound = np.array([0.6, 0.07])

        normalize = np.subtract(high_state_bound, low_state_bound)
        state_dim = mountaincar3d.state_shape
        state_dim_range = mountaincar3d.state_shape_range
        state_size = (4,)
        action_size = 5

        state_space = 400
        position_range = np.linspace(low_state_bound[0], high_state_bound[0], state_space)
        velocity_range = np.linspace(low_state_bound[1], high_state_bound[1], int(state_space/5))
        episode_rewards = []
        step_per_episode = []



    def choose_action(current_state, action_size):
        if np.random.random() > EPSILON:
            return np.argmax(deep_q_learning.get_qs(current_state))
        else:
            return np.random.randint(0, action_size)



def deep_q_learning(self):

    for episode in tqdm(range(1, EPISODE+1), unit="episode"):
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

        step_per_episode.append(step)
        episode_rewards.append(episode_reward)
