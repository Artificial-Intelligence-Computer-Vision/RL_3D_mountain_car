from header_import import *

if __name__ == "__main__":
    
    algorithm  = (sys.argv[1])
    plot = plot_graphs()  
    
    if algorithm == "deep_q_learning":
        deep_q_learning_obj = deep_q_learning_algorithm(episode = 10, epsilon = 0.1)
        step_per_episode, episode_reward = deep_q_learning_obj.deep_q_learning()
        plot.plot_episode_time_step(episode_reward, algorithm="deep_q_learning" ,type_graph = "cumulative_reward")
        plot.plot_episode_time_step(step_per_episode, algorithm="deep_q_learning", type_graph = "step_number")
    elif algorithm == "actor_critic":
        actor_critic_obj = actor_critic()
        step_per_episode, episode_rewards = actor_critic_obj.actor_critic_with_eligibility_traces()
        plot.plot_episode_time_step(episode_rewards, algorithm="actor_critic" ,type_graph = "cumulative_reward")
        plot.plot_episode_time_step(step_per_episode, algorithm="actor_critic", type_graph = "step_number")

    
