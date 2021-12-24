from header_import import *

class MountainCar3D(object):
    def __init__(self, **kwargs):
        dimension = int(max(2, kwargs.setdefault('dimension', 3)))
        self.noise = kwargs.setdefault('noise', 0.0)
        self.reward_noise = kwargs.setdefault('reward_noise', 0.0)
        self.random_start = kwargs.setdefault('random_start', False)
        self.state = np.zeros((dimension-1,2))
        self.state_range = np.array([[[-1.2, 0.6], [-0.07, 0.07]] for i in range(dimension-1)])
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


    def makeTaskSpec(self):
        ts = TaskSpec(discount_factor=1.0, reward_range=(-1.0, 0.0))
        ts.setDiscountFactor(1.0)
        ts.addDiscreteAction((0, self.state.shape[0]*2))
        flattened_ranges = self.state_range.reshape((np.prod(self.state_range.shape[:2]), self.state_range.shape[2]))
        for minValue, maxValue in flattened_ranges:
            ts.addContinuousObservation((minValue, maxValue))
        ts.setEpisodic()
        return ts.toTaskSpec()


    def step(self, thisAction):
        episodeOver = False
        theReward = -1.0
        self.taken_action(thisAction)

        if self.reached_goal():
            theReward = 0
            episodeOver = True

        if self.reward_noise > 0:
            theReward += np.random.normal(scale=self.reward_noise)
        observe_state = self.state.flatten().tolist()

        return thisAction, theReward, observe_state, episodeOver


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



