import libs.libs_env.env_cliff_gui

import libs.libs_agent.agent
import libs.libs_agent.agent_table

env     = libs.libs_env.env_cliff_gui.EnvCliffGui()

#random dummy agent
#agent   = libs.libs_agent.agent.Agent(env)

#Q table agent, using Q-learning
#agent   = libs.libs_agent.agent_table.QLearningAgent(env)

#Q table agent, using SARSA
agent   = libs.libs_agent.agent_table.SarsaAgent(env)


training_iterations = 100000

for iteration in range(0, training_iterations):
    agent.main()
    if iteration%2048 == 0:
        env.render()


while True:
    agent.main()
    env.render()
