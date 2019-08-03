"""
 Q-learning and SARSA agents coded from scratch
 table is used to store Q values
 state vector is converted to table index into table as idx = argmax(state)

 parameters
 gamma - RL discount factor
 alpha - learning rate
 epsilon_training - probability of choosing random action during training
 epsilon_testing  - probability of choosing random action during testing
"""

import libs.libs_agent.agent as libs_agent
import numpy

#basic Q learning reinforcement learning algorithm
class QLearningAgent(libs_agent.Agent):


    """!@brief initialise agent
        @param env - environment instance where agent exists
    """
    def __init__(self, env):

        #init parent class
        libs_agent.Agent.__init__(self, env)

        #init Q learning algorithm parameters
        self.gamma      = 0.9
        self.alpha      = 0.2

        #init probabilities of choosing random action
        #different for training and testing
        self.epsilon_training   = 0.1
        self.epsilon_testing    = 0.01

        #init state
        self.state      = 0
        self.state_prev = self.state;

        #init action ID
        self.action      = 0;
        self.action_prev = 0;

        #get state size, and actions count
        self.states_count  = self.env.get_size()
        self.actions_count = self.env.get_actions_count()


        #init Q table, using number of states and actions
        self.q_table = numpy.zeros((self.states_count, self.actions_count))

    """!@brief learning method
        call this in loop, as many iterations as you need

        - this method looks at the observation (env.get_observation())
        - select action, using q_table
        - execute action env.do_action(action)
        - obtain reward env.get_reward()
        - learn from exepriences - fill q_table
    """
    def main(self):

        #choose epsilon - depends on training or testing mode
        if self.is_run_best_enabled():
            epsilon = self.epsilon_testing
        else:
            epsilon = self.epsilon_training

        #QLearning needs to remember current state + action and previous state + action
        self.state_prev = self.state
        self.state      = self.env.get_observation().argmax()

        self.action_prev    = self.action
        #select action is done by probality selection using epsilon
        self.action         = self.select_action(self.q_table[self.state], epsilon)

        #obtain reward from environment
        reward = self.env.get_reward()

        #process Q learning
        q_tmp = self.q_table[self.state].max()

        d = reward + self.gamma*q_tmp - self.q_table[self.state_prev][self.action_prev]

        self.q_table[self.state_prev][self.action_prev]+= self.alpha*d

        #execute action
        self.env.do_action(self.action)

    #print Q table values
    def print_table(self):
        print(self.q_table)



#basic SARSA reinforcement learning algorithm
class SarsaAgent(libs_agent.Agent):

    """!@brief initialise agent
        @param env - environment instance where agent exists
    """
    def __init__(self, env):

        #init parent class
        libs_agent.Agent.__init__(self, env)

        #init sarsa algorithm parameters
        self.gamma      = 0.9
        self.alpha      = 0.1

        #init probabilities of choosing random action
        #different for training and testing
        self.epsilon_training   = 0.2
        self.epsilon_testing    = 0.01

        #init state
        self.state      = 0
        self.state_prev = self.state;

        #init action ID
        self.action      = 0;
        self.action_prev = 0;

        #get state size, and actions count
        self.states_count  = self.env.get_size()
        self.actions_count = self.env.get_actions_count()


        #init Q table, using number of states and actions
        self.q_table = numpy.zeros((self.states_count, self.actions_count))


    """!@brief learning method
        call this in loop, as many iterations as you need

        - this method looks at the observation (env.get_observation())
        - select action, using q_table
        - execute action env.do_action(action)
        - obtain reward env.get_reward()
        - learn from exepriences - fill q_table
    """
    def main(self):

        #choose epsilon - depends on training or testing mode
        if self.is_run_best_enabled():
            epsilon = self.epsilon_testing
        else:
            epsilon = self.epsilon_training

        #SARSA needs to remember current state + action and previous state + action
        self.state_prev = self.state
        self.state      = self.env.get_observation().argmax()

        self.action_prev    = self.action
        #select action is done by probality selection using epsilon
        self.action         = self.select_action(self.q_table[self.state], epsilon)

        #obtain reward from environment
        reward = self.env.get_reward()

        #process SARSA learning
        d = reward + self.gamma*self.q_table[self.state][self.action] - self.q_table[self.state_prev][self.action_prev]

        self.q_table[self.state_prev][self.action_prev]+= self.alpha*d

        #execute action
        self.env.do_action(self.action)

    #print Q table values
    def print_table(self):
        print(self.q_table)
