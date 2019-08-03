import random
import numpy

"""!@brief Basic reinfrocement learning Agent interface
    all other agents should inherit from this class

    @param name The name of the user.
"""

class Agent():
    """!@brief initialise agent
        @param env - environment instance where agent exists
    """
    def __init__(self, env):
        self.env = env
        self.run_best_disable()

    """!@brief main agent method - learning method
        call this in loop, as many iterations as you need

        - this method looks at the observation (env.get_observation())
        - select action
        - execute action env.do_action(action)
        - obtain reward env.get_reward()
        - learn from exepriences

        overload this method with your own - this basic agent
        isn't learning, just selecting random actions
    """
    def main(self):
        action = random.randint(0, self.env.get_actions_count()-1)
        self.env.do_action(action)


    """!@brief set run_best_enabled = True
        usually call after training, before testing
    """
    def run_best_enable(self):
        self.run_best_enabled = True

    """!@brief set run_best_enabled = False
        usually call before training
    """
    def run_best_disable(self):
        self.run_best_enabled = False

    """!@brief return if run_best_enabled
        this can agent use to choose policy - if running in training mode or testing mode
        in training mode is better to choose not only the best actions
        in testing mode (agent trained and deployment) is better to choose the best actions
    """
    def is_run_best_enabled(self):
        return self.run_best_enabled


    """!@brief select action using q_values as probabilities and epsilon as parameter

        @param q_values - list or vector of q_values for each action in current state
        @param epsilon - probability of choosing non best action, value in range <0, 1>
    """
    def select_action(self, q_values, epsilon = 0.1):
        action = self.__argmax(q_values)

        r = numpy.random.uniform(0.0, 1.0)
        if r <= epsilon:
            action = random.randint(0, self.env.get_actions_count()-1)

        return action

    """!@brief private method, return idx where is the highest value of given vector (or list)
        @param v - vector, list or numpy array
    """
    def __argmax(self, v):
        result = 0
        for i in range(0, len(v)):
            if v[i] > v[result]:
                result = i

        return result
