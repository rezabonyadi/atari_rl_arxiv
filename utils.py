from player.player import Player
from environments.simulator import Atari
import numpy as np
import datetime
import os
import json
import csv
import imageio
from skimage.transform import resize

MAX_EPISODE_LENGTH= 18000
NO_OP_STEPS= 10
MAX_EPISODES= 3000
AGENT_HISTORY_LENGTH= 4
UPDATE_FREQ= 4
NETW_UPDATE_FREQ= 10000
REPLAY_MEMORY_START_SIZE = 50000
DISCOUNT_FACTOR= 0.99
MEMORY_SIZE = 1000000
BS= 32
LEARNING_RATE= 0.0001
PUNISH= 0.0
INI_EPSILON= 1.0
END_EPSILON= 0.1
MIN_OBSERVE_EPISODE= 200
GAME_ENV= "BreakoutDeterministic-v4"
REWARD_EXTRAPOLATION_EXPONENT = -5.0
frame_height = 84
frame_width = 84
LINEAR_EXPLORATION_EXPONENT = False
USE_DOUBLE_MODEL = True


class HandleResults:

    folder_to_use = ''
    settings_file_name = 'settings.jsn'
    time = datetime.datetime.now()

    def __init__(self, game_env, out_folder):
        # GAME_ENV = settings['GAME_ENV']
        d = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        self.folder_to_use = ''.join([out_folder, game_env, '/results_', d, '/'])

        if not os.path.exists(self.folder_to_use):
            os.makedirs(self.folder_to_use)

        self.results_file_name = ''.join([self.folder_to_use, 'results_', game_env, '_', d, '.csv'])

    def save_settings(self, settings, player= None):
        settings_dict = settings

        with open(''.join([self.folder_to_use, 'settings.jsn']), 'wt') as outfile:
            json.dump(settings_dict, outfile, indent=4)

        if player is not None:
            player.save_player_learner(self.folder_to_use)

    def load_settings_default(self, GAME_ENV):

        settings_dict = {}
        file_name = './default_settings.jsn'  # default_settings.jsn i in the root
        with open(file_name, 'rt') as json_file:
            settings_dict = json.load(json_file)

        return self.load_settings_dictionary(GAME_ENV, settings_dict)

    def load_settings_dictionary(self, GAME_ENV, settings_dict):

        settings_dict['GAME_ENV'] = GAME_ENV
        game_env = Atari(settings_dict['GAME_ENV'], settings_dict['frame_height'], settings_dict['frame_width'],
                         agent_history_length=settings_dict['AGENT_HISTORY_LENGTH'],
                         no_op_steps=settings_dict['NO_OP_STEPS'])

        player = self.build_player(settings_dict, game_env)

        return player, game_env, settings_dict['MAX_EPISODE_LENGTH'], settings_dict['MAX_EPISODES'], settings_dict

    def load_default_settings_constants(self, GAME_ENV):

        settings_dict = {}

        settings_dict['GAME_ENV'] = GAME_ENV
        settings_dict['AGENT_HISTORY_LENGTH'] = AGENT_HISTORY_LENGTH
        settings_dict['MEMORY_SIZE'] = MEMORY_SIZE
        settings_dict['BS'] = BS
        settings_dict['LEARNING_RATE'] = LEARNING_RATE
        settings_dict['INI_EPSILON'] = INI_EPSILON
        settings_dict['END_EPSILON'] = END_EPSILON
        settings_dict['MIN_OBSERVE_EPISODE'] = MIN_OBSERVE_EPISODE
        settings_dict['NETW_UPDATE_FREQ'] = NETW_UPDATE_FREQ
        settings_dict['UPDATE_FREQ'] = UPDATE_FREQ
        settings_dict['DISCOUNT_FACTOR'] = DISCOUNT_FACTOR
        settings_dict['REPLAY_MEMORY_START_SIZE'] = REPLAY_MEMORY_START_SIZE
        settings_dict['PUNISH'] = PUNISH
        settings_dict['REWARD_EXTRAPOLATION_EXPONENT'] = REWARD_EXTRAPOLATION_EXPONENT
        settings_dict['LINEAR_EXPLORATION_EXPONENT'] = LINEAR_EXPLORATION_EXPONENT
        settings_dict['USE_DOUBLE_MODEL'] = USE_DOUBLE_MODEL
        settings_dict['frame_height'] = frame_height
        settings_dict['frame_width'] = frame_width
        settings_dict['NO_OP_STEPS'] = NO_OP_STEPS
        settings_dict['MAX_EPISODE_LENGTH'] = MAX_EPISODE_LENGTH
        settings_dict['MAX_EPISODES'] = MAX_EPISODES

        return self.load_settings_dictionary(GAME_ENV, settings_dict)

    def load_settings_folder(self, folder, load_model):
        settings_dict = {}
        with open(''.join([folder, self.settings_file_name]), 'rt') as json_file:
            settings_dict = json.load(json_file)

        player, game_env, _, _, _ = self.load_settings_dictionary(GAME_ENV, settings_dict)

        if load_model:
            player.load_player_learner(folder)

        return player, game_env, settings_dict['MAX_EPISODE_LENGTH'], settings_dict['MAX_EPISODES'], settings_dict

    def save_res(self, res_dict):
        if not os.path.isfile(self.results_file_name):
            with open(self.results_file_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                headings = list(res_dict.keys())
                writer.writerow(headings)
            file.close()

        with open(self.results_file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            values = list(res_dict.values())
            writer.writerow(values)
        file.close()

        print(res_dict)

    def generate_gif(self, episode, frames_for_gif, reward, string_to_add):

        path = self.folder_to_use
        for idx, frame_idx in enumerate(frames_for_gif):
            frames_for_gif[idx] = resize(frame_idx, (420, 320, 3),
                                         preserve_range=True, order=0).astype(np.uint8)

        imageio.mimsave(f'{path}{"ATARI_episode_{0}_reward_{1}_{2}.gif".format(episode, reward, string_to_add)}',
                        frames_for_gif, duration=1 / 30)

    def build_player(self, settings_dict, game_env):
        player = Player(game_env, settings_dict['AGENT_HISTORY_LENGTH'], settings_dict['MEMORY_SIZE'],
                        settings_dict['BS'],
                        settings_dict['LEARNING_RATE'], settings_dict['INI_EPSILON'], settings_dict['END_EPSILON'],
                        settings_dict['MIN_OBSERVE_EPISODE'], settings_dict['NETW_UPDATE_FREQ'],
                        settings_dict['UPDATE_FREQ'], settings_dict['DISCOUNT_FACTOR'],
                        settings_dict['REPLAY_MEMORY_START_SIZE'], settings_dict['PUNISH'],
                        settings_dict['REWARD_EXTRAPOLATION_EXPONENT'], settings_dict['LINEAR_EXPLORATION_EXPONENT'],
                        settings_dict['USE_DOUBLE_MODEL'])
        return player