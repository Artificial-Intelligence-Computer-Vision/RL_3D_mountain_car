from header_import import *

if __name__ == "__main__":
    
    algorithm  = (sys.argv[1])
    plot = plot_graphs()  
    
    if algorithm == "deep_q_learning_continues_and_experience_replay":
        deep_q_learning_obj = deep_q_learning_algorithm(episode = 10, algorithm_name="deep_q_learning", transfer_learning="true")
        deep_q_learning_obj.deep_q_learning()

    elif algorithm == "actor_critic":
        actor_critic_obj = actor_critic()
        weights, step_per_episode, episode_rewards = actor_critic_obj.actor_critic_with_eligibility_traces()
        plot.plot_episode_time_step(episode_rewards, algorithm="actor_critic" ,type_graph = "cumulative_reward")
        plot.plot_episode_time_step(step_per_episode, algorithm="actor_critic", type_graph = "step_number")

    elif algorithm == "double_deep_q_learning_continues_and_experience_replay":
        deep_q_learning_obj = deep_q_learning_algorithm(episode = 3, algorithm_name="double_deep_q_learning", transfer_learning="true")
        deep_q_learning_obj.double_deep_q_learning()
   
    elif algorithm == "dueling_deep_q_learning_continues_and_experience_replay":
        deep_q_learning_obj = deep_q_learning_algorithm(episode = 10, algorithm_name="dueling_deep_q_learning", transfer_learning="false")
        deep_q_learning_obj.dueling_deep_q_learning()
