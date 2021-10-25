from header_import import *



class MountainCar3D(object):
    def __init__(self, **kwargs):
        dimension = int(max(2, kwargs.setdefault('dimension', 3)))
        self.noise = kwargs.setdefault('noise', 0.0)
        self.reward_noise = kwargs.setdefault('reward_noise', 0.0)
        self.random_start = kwargs.setdefault('random_start', False)
        self.state = numpy.zeros((dimension-1,2))
        self.state_range = numpy.array([[[-1.2, 0.6], [-0.07, 0.07]] for i in range(dimension-1)])
        self.goal_position = 0.5
        self.acc = 0.005
        self.gravity = -0.0025
        self.hillFreq = 3.0
        self.delta_time = 1.0


