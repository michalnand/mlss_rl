import numpy

class Env:
    def __init__(self):
        self.observation    = numpy.zeros(0)
        self.actions_count  = 0

        self.width  = 0
        self.height = 0
        self.depth  = 0
        self.time   = 1

        self.player_on_move = 0

        self.reset_score()
        self.set_no_terminal_state()

    def observation_init(self):
        self.observation    = numpy.zeros(self.get_size())

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_depth(self):
        return self.depth

    def get_time(self):
        return self.time

    def get_size(self):
        return self.width*self.height*self.depth

    def get_observation(self):
        return self.observation

    def get_score(self):
        return self.score

    def reset_score(self):
        self.reward = 0.0
        self.score  = 0.0
        self.move   = 0

    def next_move(self):
        self.move+= 1
        self.score+= self.get_reward()

    def get_move(self):
        return self.move

    def get_normalised_score(self):
        return self.get_score()/(self.get_move() + 0.00000000000001)

    def get_reward(self):
        return self.reward

    def get_player_on_move(self):
        return self.player_on_move

    def get_actions_count(self):
        return self.actions_count

    def do_action(self, action):
        pass

    def is_done(self):
        return self.terminal_state

    def set_terminal_state(self):
        self.terminal_state = True

    def set_no_terminal_state(self):
        self.terminal_state = False


    def _print(self):
        print("dummy environment")

    def print_info(self):
        print("env info")
        print("state shape   = ", self.get_depth(), self.get_height(), self.get_width())
        print("actions count = ", self.get_actions_count())
        #print("state = ", self.get_observation())
        print()

    def print_state(self):
        idx = 0
        for z in range(0, self.depth):
            for y in range(0, self.height):
                for x in range(0, self.width):
                    print(self.observation[idx],end=" ")
                    idx+= 1
                print()
            print()
        print()

    def render(self):
        pass
