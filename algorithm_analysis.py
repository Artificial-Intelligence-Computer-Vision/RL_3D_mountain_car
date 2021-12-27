from header_import import *

if __name__ == "__main__":
    
    deep_q_learning_obj = deep_q_learning_algorithm(episode = 1000, epsilon = 0.1)
    deep_q_learning_obj.deep_q_learning()
