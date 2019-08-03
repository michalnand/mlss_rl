import libs.libs_env.env_atari_interface as env_atari_interface
import libs.libs_gl_gui.gl_gui as libs_gl_gui


import numpy
import time
import random


class EnvAtariArkanoid(env_atari_interface.EnvAtariInterface):

    def __init__(self, size = 32):
        env_atari_interface.EnvAtariInterface.__init__(self, size)

        #3 actions for movements
        self.actions_count  = 3

        #player paddle size
        self.player_size = 3

        self.game_height = (int)(self.height/1.2)
        self.game_width  = (int)(0.9*self.width)

        if self.game_height%2 == 0:
            self.game_height+= 1

        if self.game_width%2 == 0:
            self.game_width+= 1

        self.moves_to_win   = 0
        self.moves_old      = 0

        self.miss_tmp = 0
        self.miss     = 100.0

        #init game
        self.reset()

        self.window_name = "ARKANOID"



    def reset(self):
        self.clear_game_screen()
        self.board    = numpy.zeros((self.game_height, self.game_width))

        for x in range(0, self.game_width):
            self.board[3][x] = 1
            self.board[4][x] = 2
            self.board[5][x] = 3
            self.board[6][x] = 4
            self.board[7][x] = 5
            self.board[8][x] = 6

        for y in range(0, self.game_height):
            self.board[y][0] = 7
            self.board[y][self.game_width-1] = 7

        for x in range(0, self.game_width):
            self.board[0][x] = 7

        self.initial_items_count = self.__count_remaining()

        self.__respawn()


    def _print(self):
        #print("move=", self.get_move(), "  score=", self.get_score(), "  normalised score=", self.get_normalised_score())
        print("done game ", self.game_idx, " moves ", self.moves_to_win, " score ", self.get_score())

    def do_action(self, action):

        '''
        if self.ball_x > self.player_position:
            action = 0
        elif self.ball_x < self.player_position:
            action = 1
        else:
            action = 2
        '''


        if action == 0:
            self.player_position+= 1
        elif action == 1:
            self.player_position-= 1
        else:
            self.player_position+= 0


        self.player_position = self.__saturate(self.player_position, 2, self.game_width - 3)

        ball_result = self.__process_ball()


        self.clear_game_screen()

        offset_x = self.width//2 - self.game_width//2
        offset_y = self.height//2 - self.game_height//2

        for y in range(0, self.game_height):
            for x in range(0, self.game_width):
                item_idx = self.board[y][x]
                color = self.item_to_color(item_idx)
                self.set_game_screen_element(x + offset_x, y + offset_y, color)

        color = self.item_to_color(10)
        self.set_game_screen_element(self.ball_x + offset_x, self.ball_y + offset_y, color)

        color = self.item_to_color(11)
        x = self.player_position + 0 + offset_x
        y = self.game_height-1 + offset_y
        self.set_game_screen_element(x - 1, y, color)
        self.set_game_screen_element(x + 0, y, color)
        self.set_game_screen_element(x + 1, y, color)



        self.reward = 0.0
        self.set_no_terminal_state()

        if ball_result == "brick":
            self.reward = 1.0

        if ball_result == "miss":
            self.__respawn()
            self.reward = -4.0
            self.set_terminal_state()
            self.miss_tmp+= 1


        if self.get_move() - self.moves_old > 2*self.width*self.height:
            self.moves_old = self.get_move()
            self.reset()
            self.reward = -4.0
            self.set_terminal_state()

        if (self.__count_remaining()*100.0/self.initial_items_count < 5.0):
            self.reset()
            self.reward = 2.0
            self.set_terminal_state()

            self.moves_to_win = self.get_move() - self.moves_old
            self.moves_old = self.get_move()
            self.miss = 0.9*self.miss + 0.1*self.miss_tmp
            self.miss_tmp = 0
            self.next_game()


        self.update_state()
        self.next_move()

    def get_miss(self):
        return self.miss



    def __respawn(self):
        #init player position
        self.player_position = self.game_width//2

        #init ball position
        self.ball_x = int(self.game_width/2) + random.randint(-2, 2)
        self.ball_y = self.game_height - 2


        if random.randint(0, 1) == 0:
            self.ball_dx = 1
        else:
            self.ball_dx = -1

        self.ball_dy = -1


    def __process_ball(self):

        result = "none"

        if self.board[self.ball_y][self.ball_x] != 0 and self.board[self.ball_y][self.ball_x] < 7:
            x1 = 2*(self.ball_x//2) + 0
            x2 = 2*(self.ball_x//2) + 1

            x1 = self.__saturate(x1, 0, self.game_width-1)
            x2 = self.__saturate(x2, 0, self.game_width-1)

            if self.board[self.ball_y][x1] != 0 and self.board[self.ball_y][x1] < 7:
                self.board[self.ball_y][x1] = 0

            if self.board[self.ball_y][x2] != 0 and self.board[self.ball_y][x2] < 7:
                self.board[self.ball_y][x2] = 0


            self.ball_dy*= -1
            result = "brick"

        if self.ball_x <= 1:
            self.ball_dx = 1
            self.ball_y+= random.randint(-1, 1)


        if self.ball_x >= self.game_width-2:
            self.ball_dx = -1
            self.ball_y+= random.randint(-1, 1)

        if self.ball_y <= 1:
            self.ball_dy = 1



        if self.ball_y >= self.game_height-2:
            if self.player_position == self.ball_x or self.player_position - 1 == self.ball_x or self.player_position + 1 == self.ball_x:
                self.ball_dy = -1
                if self.ball_dx < 2:
                    self.ball_dx*= 2
                result = "hit"
            elif self.player_position-2 == self.ball_x or self.player_position+2 == self.ball_x:
                self.ball_dy = -1
                self.ball_dx*= -1
                result = "hit"
            else:
                result = "miss"

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

    def __count_remaining(self):
        count = 0
        for y in range(0, self.game_height):
            for x in range(0, self.game_width):
                if self.board[y][x] > 0.0 and self.board[y][x] < 7:
                    count+= 1

        return count
