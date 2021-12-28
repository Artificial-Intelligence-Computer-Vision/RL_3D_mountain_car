from header_import import *


class plot_graphs(object):
    def __init__(self):
        self.path = "graphs_charts/"


    def plot_episode_time_step(self, data, algorithm, type_graph = "reward"):

        fig = plt.figure()
        axis = fig.add_subplot(111)
        color_graph = "blue"

        if type_graph == "cumulative_reward":
            axis.plot(data, color=color_graph)
            axis.set_title("Reward vs Time Step")
            axis.set_xlabel("Time Steps")
            axis.set_ylabel("Reward per Step")
        elif type_graph == "step_number":
            axis.plot(data, color=color_graph)
            axis.set_title("Number of steps per episode vs. number of episodes")
            axis.set_xlabel("Number of Steps")
            axis.set_ylabel("Episodes")
        plt.savefig((str(self.path) + algorithm + "_" + type_graph + ".png"), dpi =500)

