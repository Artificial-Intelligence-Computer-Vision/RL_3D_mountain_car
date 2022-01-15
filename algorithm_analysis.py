from header_import import *

if __name__ == "__main__":
    
    algorithm  = (sys.argv[1])
    
    if algorithm == "deep_q_learning_continues_and_experience_replay":
        deep_q_learning_obj = deep_q_learning_algorithm(episode = 10, algorithm_name="deep_q_learning", transfer_learning="true")
        deep_q_learning_obj.deep_q_learning()

    elif algorithm == "actor_critic":
        actor_critic_obj = actor_critic(episode = 1500, algorithm_name="actor_critic")
        actor_critic_obj.actor_critic_with_eligibility_traces()

    elif algorithm == "double_deep_q_learning_continues_and_experience_replay":
        deep_q_learning_obj = deep_q_learning_algorithm(episode = 3, algorithm_name="double_deep_q_learning", transfer_learning="true")
        deep_q_learning_obj.double_deep_q_learning()
   
    elif algorithm == "dueling_deep_q_learning_continues_and_experience_replay":
        deep_q_learning_obj = deep_q_learning_algorithm(episode = 10, algorithm_name="dueling_deep_q_learning", transfer_learning="false")
        deep_q_learning_obj.dueling_deep_q_learning()
