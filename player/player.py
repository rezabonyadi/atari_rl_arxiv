import numpy as np
import os
import datetime
from player.player_components.memory import ReplayMemory
from player.player_components.learner import QLearner
from keras.models import model_from_json
# from numba import *


class Player:
    def __init__(self, game_env, agent_history_length, total_memory_size, batch_size,
                 learning_rate, init_epsilon, end_epsilon, minimum_observe_episode,
                 update_target_frequency, train_frequency, gamma, exploratory_memory_size,
                 punishment, reward_extrapolation_exponent, linear_exploration_exponent, use_double):

        self.n_actions = game_env.action_space_size
        self.init_epsilon = init_epsilon
        self.epsilon = init_epsilon
        self.end_epsilon = end_epsilon
        self.minimum_observe_episodes = minimum_observe_episode
        self.update_target_frequency = update_target_frequency
        self.game_env = game_env
        self.train_frequency = train_frequency
        self.exploratory_memory_size = exploratory_memory_size
        self.linear_exploration_exponent = linear_exploration_exponent
        self.use_double_model = use_double
        self.punishment = punishment

        if reward_extrapolation_exponent < 0:
            use_estimated_reward = False
        else:
            use_estimated_reward = True

        self.memory = ReplayMemory(self.game_env.frame_height, self.game_env.frame_width,
                                   agent_history_length, total_memory_size,
                                   batch_size, self.game_env.is_graphical,use_estimated_reward=use_estimated_reward,
                                   reward_extrapolation_exponent=reward_extrapolation_exponent,
                                   linear_exploration_exponent=self.linear_exploration_exponent,
                                   use_spotlight=False)

        self.learner = QLearner(self.n_actions, learning_rate, self.game_env.frame_height, self.game_env.frame_width,
                                agent_history_length, gamma=gamma, use_double_model=self.use_double_model)
        self.losses = []
        self.q_values = []

        # self.actuator = ???

    # @jit
    def take_action(self, current_state, total_frames, evaluation=False):
        if (np.random.rand() <= self.epsilon) or (total_frames < self.exploratory_memory_size) and (not evaluation):
            action = np.random.randint(0, self.n_actions)
        else:
            current_state = np.expand_dims(current_state, axis=0)
            q_values = self.learner.predict(current_state)

            action, q_value = self.learner.action_selection_policy(q_values)
            self.q_values.append(q_value)

        return action

    # @jit
    def learn(self, no_passed_frames):
        # This is a constant approach, learn after a given frequency, but maybe it can be improved?
        if no_passed_frames % self.train_frequency == 0:
            current_state_batch, actions, rewards, next_state_batch, terminal_flags = self.memory.get_minibatch()

            punishment = self.calculate_punishment()

            loss = self.learner.train(current_state_batch, actions, rewards, next_state_batch, terminal_flags, punishment)
            self.losses.append(loss)

        if no_passed_frames % self.update_target_frequency == 0:
            self.learner.update_target_network()

    def calculate_punishment(self):
        # punishment = 0.0
        # if abs(self.punishment) > 0:
        #     punishment = min(self.memory.min_reward - 1, -1)
        #
        punishment = -self.punishment
        return punishment

    # @jit
    def updates(self, no_passed_frames, episode, action, processed_new_frame, reward, terminal_life_lost, episode_seq):
        self.memory.add_experience(action, processed_new_frame, reward, terminal_life_lost, episode_seq, episode)

        if no_passed_frames > self.exploratory_memory_size:
            self.update_epsilon(episode)
            self.learn(no_passed_frames)

    # @jit
    def update_epsilon(self, episode):
        self.epsilon -= 0.00001
        self.epsilon = max(self.epsilon, self.end_epsilon)
        # print('Epsilon: ', str(self.epsilon))

    # @jit
    def save_player_learner(self, folder):
        model_json = self.learner.main_learner.model.to_json(indent=4)
        with open(''.join([folder, 'model_structure.jsn']), "w") as json_file:
            json_file.write(model_json)

        self.learner.main_learner.model.save_weights(''.join([folder, 'model_weights.wts']))

    # @jit
    def load_player_learner(self, folder):
        json_file = open(''.join([folder, 'model_structure.jsn']), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(''.join([folder,'model_weights.wts']))

        self.learner.main_learner.model = loaded_model
