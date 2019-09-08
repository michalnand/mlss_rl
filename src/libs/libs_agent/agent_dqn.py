import numpy
import random
import math
import libs.libs_agent.agent as libs_agent

from libs.libs_rysy_python.rysy import *



#deep Q network agent
class DQNAgent(libs_agent.Agent):
    def __init__(self, env):
        #init parent class
        libs_agent.Agent.__init__(self, env)

        self.gamma              = 0.99      #close to one, for long term goals
        self.replay_buffer_size = 8192      #buffer size to break states correlations
        self.epsilon_start      = 1.0       #starting epsilon value
        self.epsilon_end        = 0.1       #final epsilon value
        self.epsilon_decay      = 0.9999


        self.model = self.dqn_model_create()
        #self.model = self.dqn_rnn_model_create()

        self.model._print()

        #empty replay buffer
        self.replay_buffer = []

    def dqn_model_create(self):
        #create CNN network, 3 convolutional layers, 2 full connected layers

        #input with the some shape as state
        #depth is 4 stacked RGB frames
        state_shape   = Shape(self.env.get_width(), self.env.get_height(), self.env.get_depth()*self.env.get_time())

        #outputs count == actions_count
        output_shape  = Shape(1, 1, self.env.get_actions_count())

        learning_rate = 0.001

        model = CNN(state_shape, output_shape, learning_rate)

        model.add_layer("convolution", Shape(3, 3, 16))
        model.add_layer("elu")
        model.add_layer("max pooling", Shape(2, 2))

        model.add_layer("convolution", Shape(3, 3, 16))
        model.add_layer("elu")
        model.add_layer("max pooling", Shape(2, 2))

        model.add_layer("convolution", Shape(3, 3, 32))
        model.add_layer("elu")
        model.add_layer("max pooling", Shape(2, 2))

        model.add_layer("flatten")

        model.add_layer("fc", Shape(64))
        model.add_layer("elu")

        model.add_layer("output")

        return model

    def dqn_rnn_model_create(self):
        #create CNN + GRU network, 3 convolutional layers, one GRU recurrent layer and one full connected layer

        #input with the some shape as state
        #depth is 3 (RGB frame), time = 4 past frames
        state_shape   = Shape(self.env.get_width(), self.env.get_height(), self.env.get_depth(), self.env.get_time())

        #outputs count == actions_count
        output_shape  = Shape(1, 1, self.env.get_actions_count())

        learning_rate = 0.001

        model = RNN(state_shape, output_shape, learning_rate)

        model.add_layer("convolution", Shape(3, 3, 16))
        model.add_layer("elu")
        model.add_layer("max pooling", Shape(2, 2))

        model.add_layer("convolution", Shape(3, 3, 16))
        model.add_layer("elu")
        model.add_layer("max pooling", Shape(2, 2))

        model.add_layer("convolution", Shape(3, 3, 32))
        model.add_layer("elu")
        model.add_layer("max pooling", Shape(2, 2))

        model.add_layer("flatten")

        model.add_layer("gru", Shape(64))

        model.add_layer("output")

        model._print()

        return model

    def main(self):

        #choose correct epsilon - check if testing or training mode
        if self.is_run_best_enabled():
            epsilon = self.epsilon_end
        else:
            epsilon = self.epsilon_start
            if self.epsilon_start > self.epsilon_end:
                self.epsilon_start*= self.epsilon_decay

        state           = self.env.get_observation()
        state_vector    = VectorFloat(state)    #convert to C++ vector
        q_values        = VectorFloat(self.env.get_actions_count())

        #obtain Q-values from state
        self.model.forward(q_values, state_vector)

        #select action using q_values from NN and epsilon
        self.action = self.select_action(q_values, epsilon)

        #execute action
        self.env.do_action(self.action)

        #obtain reward
        self.reward = self.env.get_reward()


        #add to experience replay buffer
        #- state, q_values, reward, terminal state flag
        if len(self.replay_buffer) < self.replay_buffer_size:
            buffer_item  = {
                "state"        : state_vector,
                "q_values"     : q_values,
                "action"       : self.action,
                "reward"       : self.reward,
                "terminal"     : self.env.is_done()
            }
            self.replay_buffer.append(buffer_item)
        else:
            #compute buffer Q values, using Q learning
            for n in reversed(range(self.replay_buffer_size-1)):

                #choose zero gamme if current state is terminal
                if self.replay_buffer[n]["terminal"] == True:
                    gamma = 0.0
                else:
                    gamma = self.gamma

                action_id = self.replay_buffer[n]["action"]


                #Q-learning : Q(s[n], a[n]) = R[n] + gamma*max(Q(s[n+1]))
                q_next = max(self.replay_buffer[n+1]["q_values"])
                self.replay_buffer[n]["q_values"][action_id] = self.replay_buffer[n]["reward"] + gamma*max(self.replay_buffer[n+1]["q_values"])

                #clamp Q values into range <-10, 10> to prevent divergence
                for action in range(self.env.get_actions_count()):
                    self.replay_buffer[n]["q_values"][action] = self.__clamp(self.replay_buffer[n]["q_values"][action], -10.0, 10.0)


            '''
            common supervised training
                we have in/out pairs :
                    input         = self.replay_buffer[n]["state"]
                    target output = self.replay_buffer[n]["q_values"]
            '''

            self.model.set_training_mode()

            for i in range(self.replay_buffer_size):

                #choose random item, to break correlations
                idx = random.randint(0, self.replay_buffer_size - 1)

                state = self.replay_buffer[idx]["state"]
                target_q_values = self.replay_buffer[idx]["q_values"]

                #fit network
                self.model.train(target_q_values, state)

            self.model.unset_training_mode()

            #clear buffer
            self.replay_buffer = []


    def __clamp(self, value, min, max):
        if value < min:
            value = min

        if value > max:
            value = max

        return value
