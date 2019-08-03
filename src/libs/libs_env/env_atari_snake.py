import libs.libs_env.env_atari_interface as env_atari_interface
import libs.libs_gl_gui.gl_gui as libs_gl_gui


import numpy
import time
import random


class EnvAtariSnake(env_atari_interface.EnvAtariInterface):

    def __init__(self, size = 32):
        env_atari_interface.EnvAtariInterface.__init__(self, size)

        #4 actions for movements + 1 for pass
        self.actions_count  = 5

        self.game_height = self.height
        self.game_width  = self.width

        self.food_size = 3

        #init game
        self.reset()

        self.window_name = "SNAKE"


    def do_action(self, action_id):

        if action_id == 0:
            self.dx = 1
            self.dy = 0
        elif action_id == 1:
            self.dx = -1
            self.dy = 0
        elif action_id == 2:
            self.dx = 0
            self.dy = 1
        elif action_id == 3:
            self.dx = 0
            self.dy = -1
        else:
            self.dx = self.dx
            self.dy = self.dy


        colission = self.__snake_move()


        for y in range(0, self.game_height):
            for x in range(0, self.game_width):
                item_idx = self.board[y][x]
                color = self.item_to_color(item_idx)
                self.set_game_screen_element(x, y, color)

        for i in range(0, len(self.snake)):
            x = self.snake[i][0]
            y = self.snake[i][1]
            color = self.item_to_color(11)
            self.set_game_screen_element(x, y, color)

        x = self.food_x
        y = self.food_y
        color = self.item_to_color(1)
        for _y in range(-self.food_size//2, self.food_size//2):
            for _x in range(-self.food_size//2, self.food_size//2):
                self.set_game_screen_element(x + _x, y + _y, color)

        self.set_no_terminal_state()
        self.reward = 0.0

        if colission == -1:
            self.reward = -1.0

        if colission == -2:
            self.__respawn()
            self.reward = -1.0

        if self.__food_colision():
            self.__new_food()

            self.reward = 10.0
            self.points+= 1

            if self.points > 10:
                self.points = 0
                self.__respawn()
                self.next_game()
                self.set_terminal_state()

        if (self.get_iterations()+1)%10000 == 0:
            self.set_terminal_state()
            self.reset()
            self.next_game()


        self.update_state()
        self.next_move()

    def reset(self):
        self.points = 0
        self.__respawn()


    def __respawn(self):

        self.clear_game_screen()
        self.board    = numpy.zeros((self.game_height, self.game_width))

        wall_color = 6
        for y in range(0, self.game_width):
            self.board[y][0] = wall_color
            self.board[y][self.game_width-1] = wall_color

        for x in range(0, self.game_height):
            self.board[0][x] = wall_color
            self.board[self.game_height-1][x] = wall_color

        point = [self.game_width//2, self.game_height//2]

        self.snake = []

        for i in range(0, 16):
            self.snake.append(point.copy())
            point[0]+= 1

        self.dx = 1
        self.dy = 0

        self.__new_food()
        self.set_terminal_state()

    def __snake_move(self):

        for i in range(0, len(self.snake)-1):
            idx = (len(self.snake)-1) - i
            self.snake[idx] = self.snake[idx-1].copy()

        self.snake[0][0]+= self.dx
        self.snake[0][1]+= self.dy

        result = 0

        #itself colission
        for i in range(1, len(self.snake)):
            if self.snake[0] == self.snake[i]:
                result = -1

        #wall colission
        if self.snake[0][0] <= 0:
            self.snake[0][0] = 0
            result = -2

        if self.snake[0][0] >= self.game_width-1:
            self.snake[0][0] = self.game_width-1
            result = -2

        if self.snake[0][1] <= 0:
            self.snake[0][1] = 0
            result = -2

        if self.snake[0][1] >= self.game_height-1:
            self.snake[0][1] = self.game_height-1
            result = -2

        return result


    def __new_food(self):

        min = 1 + self.food_size
        max = self.game_width - 2 - self.food_size

        self.food_x = random.randint(min, max)
        self.food_y = random.randint(min, max)

        while (self.snake[0][0] == self.food_x) and (self.snake[0][1] == self.food_y):
            self.food_x = random.randint(min, max)
            self.food_y = random.randint(min, max)

    def __food_colision(self):
        x = self.snake[0][0]
        y = self.snake[0][1]

        target_x = self.food_x
        target_y = self.food_y
        for _y in range(-self.food_size//2, self.food_size//2):
            for _x in range(-self.food_size//2, self.food_size//2):
                if x == target_x+_x and y == target_y+_y:
                    return True

        return False
