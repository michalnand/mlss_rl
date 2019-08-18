import libs.libs_env.env as libs_env
import libs.libs_gl_gui.gl_gui as libs_gl_gui

import numpy
import time
import random
import sys

numpy.set_printoptions(threshold=sys.maxsize)

class EnvAtariInterface(libs_env.Env):

    def __init__(self, size = 32):

        #init parent class -> environment interface
        libs_env.Env.__init__(self)

        self.time_steps     = 4
        self.colors_count   = 3

        #dimensions
        self.width  = size
        self.height = size
        self.depth  = self.colors_count
        self.time   = self.time_steps

        #init state, as 1D vector (tensor with size depth*height*width)
        self.observation_init()
        self.actions_count  = 2

        self.game_screen    = numpy.zeros((self.colors_count, self.height, self.width))

        size = self.colors_count*self.height*self.width

        self.state_vec_frame_0 = numpy.zeros(size)
        self.state_vec_frame_1 = numpy.zeros(size)
        self.state_vec_frame_2 = numpy.zeros(size)
        self.state_vec_frame_3 = numpy.zeros(size)

        self.iterations     = 0
        self.game_idx       = 0


        self.size_ratio = self.width/self.height
        self.gui = libs_gl_gui.GLVisualisation()
        self.window_name = "game interface"


        self.update_state()



    def set_game_screen_element(self, x, y, z, value):
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if z < 0:
            z = 0

        if x >= self.width:
            x = self.width-1
        if y >= self.height:
            y = self.height-1
        if z >=self.colors_count:
            z =self.colors_count-1

        self.game_screen[z][y][x] = value


    def set_game_screen_element(self, x, y, color):
        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x >= self.width:
            x = self.width-1
        if y >= self.height:
            y = self.height-1

        self.game_screen[0][y][x] = color[0]
        self.game_screen[1][y][x] = color[1]
        self.game_screen[2][y][x] = color[2]

    def clear_game_screen(self):
        self.game_screen    = numpy.zeros((self.colors_count, self.height, self.width))

    def update_state(self):
        self.state_vec_frame_3 = self.state_vec_frame_2.copy()
        self.state_vec_frame_2 = self.state_vec_frame_1.copy()
        self.state_vec_frame_1 = self.state_vec_frame_0.copy()
        self.state_vec_frame_0 = self.game_screen.reshape(self.colors_count*self.height*self.width)

        self.observation = numpy.concatenate((self.state_vec_frame_0, self.state_vec_frame_1, self.state_vec_frame_2, self.state_vec_frame_3))


        self.iterations+= 1


    def render(self, dt = 0.02):

        self.width
        self.gui.init(self.window_name, 768, 768)
        self.gui.start()


        self.gui.push()
        self.gui.translate(0.0, 0.0, 0.45)


        self.gui.push()
        self.gui.set_color(0.1, 0.1, 0.1)
        self.gui.translate(0.0, 0.0, 0.0)
        self.gui.paint_square(4.0)
        self.gui.pop()

        for y in range(0, self.height):
            for x in range(0, self.width):

                    if (self.colors_count == 3):
                        r = self.game_screen[0][y][x]
                        g = self.game_screen[1][y][x]
                        b = self.game_screen[2][y][x]
                    else:
                        r = self.game_screen[0][y][x]
                        g = r
                        b = r


                    if r > 0.0 or g > 0.0 or b > 0.0:
                        element_size = 2.0/self.width

                        self.gui.push()

                        self.gui.set_color(r, g, b)
                        self.gui.translate(self.__x_to_gui_x(x), self.__y_to_gui_y(y), 0.0)
                        self.gui.paint_square(element_size)

                        self.gui.pop()


        self.gui.set_color(1.0, 1.0, 1.0)
        count = "SCORE = " + str(round(self.get_score(), 3))
        self.gui._print(-0.9, 0.95, 0.1, count)

        self.gui.set_color(1.0, 1.0, 1.0)
        game = "GAME = " + str(round(self.get_games_count(), 3))
        self.gui._print(-0.9, 0.89, 0.1, game)


        self.gui.pop()

        self.gui.finish()
        time.sleep(dt)

    def get_games_count(self):
        return self.game_idx

    def next_game(self):
        self.game_idx+= 1

    def get_iterations(self):
        return self.iterations

    def __x_to_gui_x(self, x):
        return self.size_ratio*(x*1.0/self.width - 0.5)*2.0

    def __y_to_gui_y(self, y):
        return -(y*1.0/self.height - 0.5)*2.0

    def item_to_color(self, item_idx):
        result = [0.0, 0.0, 0.0]

        if item_idx == 1:
            result = [1.0, 0.0, 0.0]
        elif item_idx == 2:
            result = [1.0, 0.5, 0.0]
        elif item_idx == 3:
            result = [1.0, 0.75, 0.0]
        elif item_idx == 4:
            result = [1.0, 1.0, 0.0]
        elif item_idx == 5:
            result = [0.0, 1.0, 0.0]
        elif item_idx == 6:
            result = [0.0, 0.0, 1.0]
        elif item_idx == 7:
            result = [0.0, 1.0, 1.0]
        elif item_idx == 10:
            result = [1.0, 1.0, 1.0]
        elif item_idx == 11:
            result = [1.0, 1.0, 1.0]
        elif item_idx == 12:
            result = [0.8, 0.52, 0.247]
        elif item_idx == 13:
            result = [0.545, 0.27, 0.074]
        elif item_idx == 14:
            result = [0.0, 0.0, 0.7]
        elif item_idx == 16:
            result = [0.0, 0.7, 0.0]
        elif item_idx == 17:
            result = [0.7, 0.7, 0.7]
        return result

    def reset(self):
        pass
