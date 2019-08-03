import libs.libs_env.env_cliff as libs_env_cliff
import libs.libs_gl_gui.gl_gui as libs_gl_gui
import numpy
import time

class EnvCliffGui(libs_env_cliff.EnvCliff):

    def __init__(self):

        #init parent class -> EnvCliff
        libs_env_cliff.EnvCliff.__init__(self)

        self.gui = libs_gl_gui.GLVisualisation()


    def render(self):

        self.gui.start()

        for y in range(0, self.height):
            for x in range(0, self.width):

                self.gui.push()

                dim_max = self.width
                if self.height > dim_max:
                    dim_max = self.height

                size = 1.95/dim_max
                y_ = (y*1.0/dim_max - 0.5)*2.0
                x_ = (x*1.0/dim_max - 0.5)*2.0

                self.gui.translate(x_, -y_, 0.0)


                if (y == self.agent_y) and (x == self.agent_x):
                    self.gui.set_color(1.0, 1.0, 0.0)
                elif self.rewards[y][x] < 0.0:
                    self.gui.set_color(1.0, 0.0, 0.0)
                elif self.rewards[y][x] > 0.0:
                    self.gui.set_color(0.0, 1.0, 0.0)
                else:
                    self.gui.set_color(0.9, 0.9, 0.9)

                self.gui.paint_square(size)

                self.gui.pop()

        text = "move = " + str(self.get_move()) + "  "
        text+= "score = " + str(self.get_score()) + "  "
        text+= "normalised score = " + str(self.get_normalised_score()) + "  "
        self.gui.push()
        self.gui.set_color(1.0, 1.0, 1.0)
        self.gui._print(-1.0, -0.5, 0.0, text);
        self.gui.pop()

        self.gui.finish()

        time.sleep(0.05)
