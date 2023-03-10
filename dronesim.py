import numpy as np
import pygame
import gym

from gym import spaces
import tensorflow as tf


class DroneSim(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 256  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int), 
                "velocity": spaces.Box(-1,1, shape=(2,), dtype=float)
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Box(-1,1,shape=(2,), dtype=float)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        # self._action_to_direction = {
        #     0: np.array([1, 0]),
        #     1: np.array([0, 1]),
        #     2: np.array([-1, 0]),
        #     3: np.array([0, -1]),
        # }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    
    def _get_obs(self):
        low,high = 0,self.window_size - 1
        mean = (high + low) / 2
        std = ((high - low)** 2 / 12)**(1/2)
        agent_loc = (self.agent_location - mean) / std
        target_loc = (self.target_location - mean) / std

        low,high = -10,10
        mean =(high + low) / 2
        std = ((high - low)** 2  / 12 )**(1/2)
        agent_vel = (self.agent_velocity - mean) / std


        return {"agent": agent_loc,  "target": target_loc, "velocity": agent_vel}
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self.agent_location - self.target_location, ord=1
            )
        }
    
    def reset(self, seed=None, options=None):
    # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self.agent_location = self.np_random.integers(0, self.window_size, size=2)
        self.agent_velocity = np.zeros((2,))
        # self.agent_location = np.random.randint(0, self.window_size -1, size=(2,)).astype(np.float32)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self.target_location = np.random.randint(0,self.window_size - 1, size=(2,)).astype(int) 
        # self.target_location = self.agent_location
        # self.target_location = np.zeros((2,)) + 100
        distance = np.linalg.norm(self.agent_location - self.target_location)
        while distance < 15:
            self.target_location = self.np_random.integers(
                0, self.window_size, size=2, dtype=int
            )
            distance = np.linalg.norm(self.agent_location - self.target_location)
        # currently using action as direction displacement's 0th derivative
        self.agent_location = self.agent_location.astype(float)
        self.target_location = self.target_location.astype(float)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        self.out_of_bounds_counter = 0

        return observation, info
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        # direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        action = action.numpy()
        self.agent_location += self.agent_velocity 
        self.agent_velocity += action
        self.agent_velocity = np.clip(self.agent_velocity,-10,10)
        #self.agent_location = np.clip(self.agent_location,0,self.window_size -1)
        
        # An episode is done iff the agent has reached the target
        distance = np.linalg.norm(self.target_location - self.agent_location)
        print(distance)
        if (self.agent_location < 0).any() or (self.agent_location > self.window_size).any():
            terminated = False 
            self.out_of_bounds_counter+=1
            if self.out_of_bounds_counter > 5:
                terminated = True
            reward = -25000
        elif distance < 15:
            reward =250
            terminated = True
        else:
            reward = -1 * distance
            terminated = False
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        # First we draw the target
        pygame.draw.rect(canvas, (255, 0, 0), pygame.Rect(self.target_location, (10,10),))
        # Now we draw the agent
        pygame.draw.rect(canvas, (0, 0, 255), pygame.Rect(self.agent_location, (10,10)))
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()