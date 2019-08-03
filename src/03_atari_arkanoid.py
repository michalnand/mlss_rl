import libs.libs_env.env_atari_arkanoid

import libs.libs_agent.agent
import libs.libs_agent.agent_dqn

env     = libs.libs_env.env_atari_arkanoid.EnvAtariArkanoid(24)

#random dummy agent
#agent   = libs.libs_agent.agent.Agent(env)


#Deep Q-Network agent

gamma               = 0.99
replay_buffer_size  = 8196
epsilon_training    = 1.0
epsilon_testing     = 0.1
epsilon_decay       = 0.9999
network_config_file_name = "networks/arkanoid/network_config.json"

agent   = libs.libs_agent.agent_dqn.DQNAgent(env, network_config_file_name, gamma, replay_buffer_size, epsilon_training, epsilon_testing, epsilon_decay)

training_games_count = 200
while env.get_games_count() < training_games_count:
    agent.main()
    if env.get_iterations()%1024 == 0:
        env.render()
        print("training done = ", env.get_games_count()*100.0/training_games_count , "%, ", env.get_games_count(), " score = ", env.get_score())

agent.run_best_enable()

while True:
    agent.main()
    env.render()
