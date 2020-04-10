import numpy as np
import gym
import tensorflow as tf
from skimage.transform import resize
from skimage.color import rgb2gray

class Atari:
    """Wrapper for the environment provided by gym"""

    def __init__(self, env_name, frame_height, frame_width, agent_history_length=4, no_op_steps=10):
        self.env_name = env_name
        self.env = gym.make(env_name)
        print("The environment has the following {} actions: {}".format(self.env.action_space.n,
                                                                        self.env.unwrapped.get_action_meanings()))

        # self.frame_processor = ProcessFrame()
        self.current_state = np.empty((frame_height, frame_width, agent_history_length), dtype=np.uint8)
        self.last_lives = 0
        self.no_op_steps = no_op_steps
        self.agent_history_length = agent_history_length
        self.action_space_size = self.env.action_space.n
        self.game_shape = self.env.observation_space.shape
        self.is_graphical = True if len(self.game_shape) > 1 else False
        self.frame_height = frame_height if self.is_graphical else self.game_shape[0]
        self.frame_width = frame_width if self.is_graphical else 1

    def reset(self, evaluation=False):
        """
        Args:
            evaluation: A boolean saying whether the agent is evaluating or training
        Resets the environment and stacks four frames ontop of each other to
        create the first state
        """
        frame = self.env.reset()
        self.last_lives = 0
        terminal_life_lost = True  # Set to true so that the agent starts
        # with a 'FIRE' action when evaluating
        if evaluation:
            for _ in range(np.random.randint(1, self.no_op_steps)):
                frame, _, _, _ = self.env.step(1)  # Action 'Fire'
        else:
            frame, _, _, _ = self.env.step(1)

        processed_frame = self.process(frame)
        for i in range(self.agent_history_length):
            self.update_current_state(processed_frame)

        return terminal_life_lost

    def process(self, frame):
        # returns a (height, width, 1) array
        if self.is_graphical:
            # frame = frame[34:200, 0:160, :]
            frame = resize(frame, (self.frame_height, self.frame_width))
            frame = rgb2gray(frame)
            frame = np.uint8(frame*255)
        return frame

    def step(self, action):
        """
        Args:
            action: Integer, action the agent performs
        Performs an action and observes the reward and terminal state from the environment
        """
        new_frame, reward, terminal, info = self.env.step(action)  # (5★)

        if info['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.last_lives = info['ale.lives']

        processed_new_frame = self.process(new_frame)
        self.update_current_state(processed_new_frame)

        return processed_new_frame, reward, terminal, terminal_life_lost, new_frame

    def get_current_state(self):
        return self.current_state

    def update_current_state(self, frame):
        self.current_state = np.append(self.current_state[:, :, 1:], np.expand_dims(frame, axis=2), axis=2)
