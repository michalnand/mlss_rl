import libs.libs_env.env_atari_interface as env_atari_interface
import libs.libs_gl_gui.gl_gui as libs_gl_gui


import numpy
import time
import random

class EnvAtariPong(env_atari_interface.EnvAtariInterface):

    def __init__(self, size = 32):
        env_atari_interface.EnvAtariInterface.__init__(self, size)

        #3 actions for movements
        self.actions_count  = 3

        #player paddle size
        self.player_size = 5

        self.game_height = self.height
        self.game_width  = self.width

        self.player_a_setpoint = 0
        self.player_b_setpoint = 0

        self.area_mantinel_range = 4

        #init game
        self.__respawn()
        self.window_name = "PONG"



    def __respawn(self):
        self.clear_game_screen()

        self.player_a = self.game_height//2
        self.player_b = self.game_height//2

        #ball to center + some noise
        self.ball_x  = (int)(self.width/2  + random.randint(-1, 1))
        self.ball_y  = (int)(self.height/2 + random.randint(-1, 1))

        #random ball move
        if random.randint(0, 1) == 0:
            self.ball_dx = 1
        else:
            self.ball_dx = -1

        if random.randint(0, 1) == 0:
            self.ball_dy = 1
        else:
            self.ball_dy = -1


    def _print(self):
        #print("move=", self.get_move(), "  score=", self.get_score(), "  normalised score=", self.get_normalised_score())
        print("done game ", self.game_idx, " score ", self.get_score())

    def do_action(self, action):


        if random.random() < 0.25:
            if random.random() < 0.5:
                self.player_b+= 1
            else:
                self.player_b-= 1
        else:
            if self.ball_y > self.player_b:
                self.player_b+= 1

            if self.ball_y < self.player_b:
                self.player_b-= 1


        self.player_b = self.__saturate(self.player_b, 2, self.height-4)


        '''
        if random.random() < 0.1:
            if random.random() < 0.5:
                self.player_a+= 1
            else:
                self.player_a-= 1
        else:
            if self.ball_y > self.player_a:
                self.player_a+= 1

            if self.ball_y < self.player_a:
                self.player_a-= 1
        '''

        if action == 0:
            self.player_a+= 1
        elif action == 1:
            self.player_a-= 1

        self.player_a = self.__saturate(self.player_a, 2, self.height-4)

        result = self.__process_ball()

        self.reward = 0.0
        self.set_no_terminal_state()

        if result == "player_b_miss":
            self.reward = 10.0
            self.__respawn()
            self.player_a_setpoint+= 1

        if result == "player_a_miss":
            self.reward = -10.0
            self.__respawn()
            self.player_b_setpoint+= 1

        if self.player_a_setpoint +  self.player_b_setpoint >= 21:
            self.player_a_setpoint = 0
            self.player_b_setpoint = 0

            self.__respawn()

            self.set_terminal_state()
            self.next_game()

        self.clear_game_screen()

        for y in range(0, self.game_height):
            for x in range(0, self.game_width):
                color = self.item_to_color(13)
                self.set_game_screen_element(x, y, color)

        for x in range(0, self.game_width):
            color = self.item_to_color(10)
            self.set_game_screen_element(x, self.area_mantinel_range, color)
            self.set_game_screen_element(x, self.game_height-1-self.area_mantinel_range, color)

        self.__add_player(self.player_a, self.player_b)
        self.__add_ball(self.ball_x, self.ball_y)

        self.update_state()
        self.next_move()

    def __add_player(self, player_a, player_b):

        for i in range(0, self.player_size):

            ofs = i - self.player_size//2

            color = self.item_to_color(10)
            self.set_game_screen_element(0, player_a + ofs, color)

            color = self.item_to_color(10)
            self.set_game_screen_element(self.game_width-1, player_b + ofs, color)


    def __add_ball(self, x, y):

        color = self.item_to_color(10)
        self.set_game_screen_element(x, y, color)


    def __process_ball(self):

        result = "none"

        if self.ball_y <= self.area_mantinel_range:
            self.ball_dy = 1

        if self.ball_y >= self.game_height-1-self.area_mantinel_range:
            self.ball_dy = -1

        if self.ball_x <= 1:
            player_min = int(self.player_a - self.player_size/2)
            player_max = int(self.player_a + self.player_size/2)
            player_center = int((player_max + player_min)/2)

            if self.ball_y == player_max:
                self.ball_dx = 1
                self.ball_dy*= -1
                result = "player_a_hit"
            elif self.ball_y == player_min:
                self.ball_dx = 1
                self.ball_dy*= -1
                result = "player_a_hit"
            elif self.ball_y == player_center:
                self.ball_dx = 2
                result = "player_a_hit"
            elif self.ball_y <= player_max and self.ball_y >= player_min:
                self.ball_dx = 1
                result = "player_a_hit"
            else:
                result = "player_a_miss"

        if self.ball_x >= self.game_width-2:
            player_min = int(self.player_b - self.player_size/2)
            player_max = int(self.player_b + self.player_size/2)
            player_center = int((player_max + player_min)/2)

            if self.ball_y == player_max:
                self.ball_dx = -1
                self.ball_dy*= -1
                result = "player_b_hit"
            elif self.ball_y == player_min:
                self.ball_dx = -1
                self.ball_dy*= -1
                result = "player_b_hit"
            elif self.ball_y == player_center:
                self.ball_dx = -2
                result = "player_b_hit"
            elif self.ball_y <= player_max and self.ball_y >= player_min:
                self.ball_dx = -1
                result = "player_b_hit"
            else:
                result = "player_b_miss"

        self.ball_x+= self.ball_dx
        self.ball_y+= self.ball_dy
        self.ball_x = self.__saturate(self.ball_x, 0, self.game_width-1)
        self.ball_y = self.__saturate(self.ball_y, 0, self.game_height-1)


        return result


    def __saturate(self, value, min, max):
        if value > max:
            value = max

        if value < min:
            value = min

        return value
