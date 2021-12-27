from header_import import *


class MountainCar3D(object):
    def __init__(self, noise = 0.0, reward_noise = 0.0, random_start = False):
        self.noise = noise
        self.reward_noise = reward_noise
        self.random_start = random_start
        self.state = np.zeros((2,2))
        self.state_range = np.array([[[-1.2, 0.6], [-0.07, 0.07]] for i in range(3-1)])
        self.goal_position = 0.5
        self.acc = 0.005
        self.gravity = -0.0025
        self.hillFreq = 3.0
        self.delta_time = 1.0


    def reset(self):
        if self.random_start:
            self.state = np.random.random_sample(self.state.shape)
            self.state *= (self.state_range[:,:,1] - self.state_range[:,:,0])
            self.state += self.state_range[:,:,0]
        else:
            self.state = np.zeros(self.state.shape)
            self.state[:,0] = -0.5

        return np.array(self.state.flatten().tolist())


    def reached_goal(self):
        return (self.state[:,0] >= self.goal_position).all()


    def step(self, action):
        reached_goal = False
        reward = -1.0
        self.taken_action(action)

        if self.reached_goal():
            reward = 0
            reached_goal = True

        if self.reward_noise > 0:
            reward += np.random.normal(scale=self.reward_noise)
        next_state = np.asarray(self.state.flatten().tolist())

        return action, reward, next_state, reached_goal


    def taken_action(self, intAction):
        intAction = intAction - 1
        direction = np.zeros((self.state.shape[0],))
        if intAction >= 0:
            direction[int(int(intAction)/2)] = ((intAction % 2) - 0.5)*2.0
        if self.noise > 0:
            direction += self.acc * np.random.normal(scale=self.noise, size=direction.shape)

        self.state[:,1] += self.acc*(direction) + self.gravity*np.cos(self.hillFreq*self.state[:,0])
        self.state[:,1] = self.state[:,1].clip(min=self.state_range[:,1,0], max=self.state_range[:,1,1])
        self.state[:,0] += self.delta_time * self.state[:,1]
        self.state[:,0] = self.state[:,0].clip(min=self.state_range[:,0,0], max=self.state_range[:,0,1])



