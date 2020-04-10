from player.player import Player
from environments.simulator import Atari
import numpy as np
import datetime
from utils import HandleResults
import numba

'''
Set the main settings in the default_settings.jsn
* PUNISH controls the positive punishment, to be set to non-zero to apply, set to zero to ignore
* REWARD_EXTRAPOLATION_EXPONENT controls the \lambda for backfilling. Set to -1.0 to turn this off (i.e., use the 
actual reward values only)
'''

def run_episode(max_episode_length, episode, game_env, player, total_frames, evaluation=False):
    terminal_life_lost = game_env.reset()
    episode_reward = 0
    life_seq = 0
    frame_number = 0
    gif_frames = []
    while True:
        # Get state, make action, get next state (rewards, terminal, ...), record the experience, train if necessary
        current_state = game_env.get_current_state()
        action = player.take_action(current_state, total_frames, evaluation)
        processed_new_frame, reward, terminal, terminal_life_lost, original_frame = game_env.step(action)

        if frame_number >= max_episode_length:
            terminal = True
            terminal_life_lost = True

        # if evaluation:
        #     gif_frames.append(original_frame)

        if not evaluation:
            player.updates(total_frames, episode, action, processed_new_frame, reward, terminal_life_lost, life_seq)

        episode_reward += reward
        life_seq += 1

        if terminal_life_lost:
            life_seq = 0

        # game_env.env.render()
        total_frames += 1
        frame_number += 1

        if terminal:
            break

    return episode_reward, total_frames


def learn_by_game(results_handler, load_folder='', load_model=False):

    if load_folder is not '':
        player, game_env, max_episode_length, max_number_of_episodes, all_settings = \
            results_handler.load_settings_folder(load_folder, load_model)
    else:
        player, game_env, max_episode_length, max_number_of_episodes, all_settings = \
            results_handler.load_settings_default(GAME_ENV)

    for k, v in all_settings.items():
        print(k, ': ', v)

    print('****************************')

    results_handler.save_settings(all_settings, player)
    res_dict = {}

    highest_reward = 0
    total_frames = 0.0
    prev_frames = 0.0
    all_rewards = []
    time = datetime.datetime.now()
    prev_time = time
    best_evaluation = 0

    for episode in range(max_number_of_episodes):
        episode_reward, total_frames = run_episode(max_episode_length, episode, game_env, player, total_frames)

        # all_rewards[episode] = episode_reward
        all_rewards.append(episode_reward)

        if episode_reward>highest_reward:
            highest_reward = episode_reward

        if episode % 10 == 0:
            # evaluation_reward, _ = run_episode(max_episode_length, episode, game_env, player, 0, evaluation=True)

            # if evaluation_reward > best_evaluation:
            #     best_evaluation = evaluation_reward
                # print('Best eval: ', str(best_evaluation))

            now = datetime.datetime.now()
            res_dict['time'] = str(now - time)
            res_dict['episode'] = episode
            res_dict['total_frames'] = total_frames
            res_dict['epsilon'] = format(player.epsilon, '.3f')
            res_dict['highest_reward'] = highest_reward
            # res_dict['best_eval'] = best_evaluation
            res_dict['mean_rewards'] = np.mean(all_rewards[-10:])
            res_dict['mean_loss'] = format(np.mean(player.losses[-10:]), '.5f')
            # res_dict['memory_vol'] = player.memory.count
            # res_dict['fps'] = (total_frames - prev_frames) / ((now - prev_time).total_seconds())
            # res_dict['sparsity'] = np.mean(player.memory.sparsity_lengths[-10:])
            res_dict['estimating_reward'] = player.memory.use_estimated_reward
            res_dict['reward_exponent'] = player.memory.reward_extrapolation_exponent

            results_handler.save_res(res_dict)

            prev_time = now
            prev_frames = total_frames

OUT_FOLDER = './output/test_res/'

games = ['BreakoutDeterministic-v4', 'AsterixDeterministic-v4', 'CarnivalDeterministic-v4', 'MsPacmanDeterministic-v4',
    'UpNDownDeterministic-v4', 'AssaultDeterministic-v4']

for GAME_ENV in games:
    handler = HandleResults(GAME_ENV, OUT_FOLDER)
    learn_by_game(handler)
