from header_import import *

if __name__ == "__main__":
    
    # for now
    deep_q_learning_obj = deep_q_learning_algorithm()
    deep_q_learning_obj.deep_q_learning(episode = 400, epsilon = 0.1)
