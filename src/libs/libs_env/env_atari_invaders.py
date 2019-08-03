import libs.libs_env.env_atari_interface as env_atari_interface
import libs.libs_gl_gui.gl_gui as libs_gl_gui

import numpy
import time
import random

class EnvAtariInvaders(env_atari_interface.EnvAtariInterface):

    def __init__(self, size = 32):
        env_atari_interface.EnvAtariInterface.__init__(self, size)

        #3 actions for movements + 1 for fire
        self.actions_count  = 4

        #player paddle size
        self.player_size = 3

        self.game_height = (int)(self.height)
        self.game_width  = (int)(self.width)

        #init game
        self.reset()

        self.window_name = "SPACE INVADERS"


    def reset(self):
        self.clear_game_screen()

        self.__init_invaders()
        self.__init_player_shoot()
        self.__init_enemy_shoot()

        self.__respawn()

    def _print(self):
        print("move=", self.get_move(), "  score=", self.get_score(), " game=", self.get_games_count())

    def do_action(self, action):

        fire = False
        if action == 0:
            self.player_position+= 1
        elif action == 1:
            self.player_position-= 1
        elif action == 2:
            fire = True
        else:
            pass

        self.player_position = self.__saturate(self.player_position, 2, self.game_width - 3)

        self.__process_invaders()
        player_shoot_result = self.__process_player_shoot(fire)
        enemy_hit_result = self.__process_enemy_shoot()

        player_hit_result = self.__process_player_hit()

        self.clear_game_screen()

        if enemy_hit_result > 0.0:
            for y in range(0, self.game_height):
                for x in range(0, self.game_width):
                    color = [0.6, 0.0, 0.0]
                    self.set_game_screen_element(x, y, color)


        for i in range(0, len(self.invaders)):
            color = self.item_to_color(5)
            self.set_game_screen_element(self.invaders[i][0], self.invaders[i][1], color)
            self.set_game_screen_element(self.invaders[i][0]+1, self.invaders[i][1], color)


        for i in range(0, len(self.player_shoot)):
            color = self.item_to_color(3)
            self.set_game_screen_element(self.player_shoot[i][0], self.player_shoot[i][1], color)

        for i in range(0, len(self.enemy_shoot)):
            color = self.item_to_color(6)
            self.set_game_screen_element(self.enemy_shoot[i][0], self.enemy_shoot[i][1], color)

        color = self.item_to_color(11)
        x = self.player_position + 0
        y = self.game_height-1
        self.set_game_screen_element(x - 1, y, color)
        self.set_game_screen_element(x + 0, y, color)
        self.set_game_screen_element(x + 1, y, color)



        self.reward = 0.0
        self.set_no_terminal_state()

        if enemy_hit_result > 0.0:
            self.reward = -5.0
            self.__respawn()
        elif player_hit_result > 0.0:
            self.reward = 1.0
        elif player_shoot_result < 0.0:
            self.reward = -0.001

        if self.__count_remaining() == 0:
            self.reward = 4.0
            self.set_terminal_state()
            self.reset()
            self.next_game()

        if (self.get_iterations()+1)%8000 == 0:
            self.set_terminal_state()
            self.reset()
            self.next_game()

        self.update_state()
        self.next_move()


    def __respawn(self):
        self.__init_player_shoot()

        #init player position
        self.player_position = self.game_width//2


    def __init_player_shoot(self):
        self.player_shoot = []



    def __process_player_shoot(self, fire = False):
        if fire:
            player_nearest_shoot_distance = 0
            for i in range(0, len(self.player_shoot)):
                if self.player_shoot[i][1] > player_nearest_shoot_distance:
                    player_nearest_shoot_distance = self.player_shoot[i][1]

            if player_nearest_shoot_distance < self.game_height*0.75:
                shoot = [self.player_position, self.game_height-2]
                self.player_shoot.append(shoot)

        result = 0.0

        for i in range(0, len(self.player_shoot)):
            self.player_shoot[i][1]-= 1
            if self.player_shoot[i][1] <= 0:
                result = -1.0
                del self.player_shoot[i]
                break

        return result




    def __init_invaders(self):
        self.invaders_dx = 1
        self.invaders = []
        step = 4
        for i in range(0, self.game_width//step - step//2):
            for line in range(0, 6):
                x = i*step + 2*step
                y = 3*line + 2
                self.invaders.append([x, y])

    def __process_invaders(self):
        for i in range(0, len(self.invaders)):
            if self.invaders[i][0] > self.game_width - 2:
                self.invaders_dx = -1

            if self.invaders[i][0] <= 1:
                self.invaders_dx = 1

        for i in range(0, len(self.invaders)):
            self.invaders[i][0]+= self.invaders_dx

    def __process_player_hit(self):
        for j in range(0, len(self.player_shoot)):
            for i in range(0, len(self.invaders)):
                x = self.player_shoot[j][0]
                y = self.player_shoot[j][1]
                tx = self.invaders[i][0]
                ty = self.invaders[i][1]
                if x == tx and y == ty or x == tx+1 and y == ty:
                    del self.invaders[i]
                    del self.player_shoot[j]
                    return 1.0

        return 0

    def __init_enemy_shoot(self):
        if len(self.invaders) == 0:
            return

        self.enemy_shoot = []
        for i in range(0, 3):
            idx = random.randint(0, len(self.invaders)-1)
            pos = [self.invaders[idx][0], self.invaders[idx][1]]
            self.enemy_shoot.append(pos)



    def __process_enemy_shoot(self):
        if len(self.invaders) == 0:
            return 0

        result = 0
        for i in range(0, len(self.enemy_shoot)):
            self.enemy_shoot[i][1]+= 1

            if self.enemy_shoot[i][1] >= self.game_height-3:
                pos = int(self.enemy_shoot[i][0])
                player = self.player_position
                if player == pos or player+1 == pos or player-1 == pos:
                    result+= 1.0

            if self.enemy_shoot[i][1] >= self.game_height:
                idx = random.randint(0, len(self.invaders)-1)
                pos = [self.invaders[idx][0], self.invaders[idx][1]]
                self.enemy_shoot[i] = pos

        return result



    def __saturate(self, value, min, max):
        if value > max:
            value = max

        if value < min:
            value = min

        return value

    def __count_remaining(self):
        return len(self.invaders)
