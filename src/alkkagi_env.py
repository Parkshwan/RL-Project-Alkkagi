import gym
from gym import spaces
import numpy as np
import pymunk
import pymunk.pygame_util
import pygame
import math
import random

class AlkkagiEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agent_discs=1, num_opponent_discs=1):
        super(AlkkagiEnv, self).__init__()
        self.screen_width = 600
        self.screen_height = 600
        self.agent_radius = 15
        self.max_force = 1000

        self.num_agent_discs = num_agent_discs
        self.num_opponent_discs = num_opponent_discs
        self.discs = []
        self.agent_discs = []
        self.opponent_discs = []
        
        # Observation Space: (x, y, team) * (num_agent + num_opponent)
        single_space = spaces.Tuple(
            (
                spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32), # x, y position
                spaces.Discrete(3) # 0 agent, 1 opponent, -1 dead
            )
        )
        self.observation_space = spaces.Tuple([single_space] * (self.num_agent_discs + self.num_opponent_discs))

        # Action Space: Each step, pick one disk and flick with fx, fy
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(self.num_agent_discs), # disc index
                spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # (fx, fy) (will be scaled by multiplying max_force)
            )
        )

        # pymunk physics
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)

        # damping for friction
        self.space.damping = 0.2

        # pygame rendering
        self.screen = None
        self.clock = None
        self.draw_options = None

    def _add_disc(self, position):
        mass = 1
        radius = self.agent_radius
        inertia = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Circle(body, radius)
        shape.elasticity = 0.5
        self.space.add(body, shape)
        self.discs.append(body)

        return body

    def _remove_out_of_bounds_discs(self):
        new_discs = []
        for disc in self.discs:
            x, y = disc.position
            if 0 <= x <= self.screen_width and 0 <= y <= self.screen_height:
                new_discs.append(disc)
            else:
                for shape in disc.shapes:
                    self.space.remove(shape)
                self.space.remove(disc)

                # remove disc in agent/opponent list
                if disc in self.agent_discs:
                    self.agent_discs.remove(disc)
                elif disc in self.opponent_discs:
                    self.opponent_discs.remove(disc)
                    
        self.discs = new_discs
    
    def _all_discs_stopped(self, threshold=5.0):
        all_stopped = True
        for disc in self.discs:
            speed = disc.velocity.length
            if speed < threshold:
                # stop the disk when the velocity is less than threshold
                disc.velocity = pymunk.Vec2d(0, 0)
                disc.angular_velocity = 0
            else:
                all_stopped = False
        return all_stopped

    def _random_pos(self, y_low: int, y_high: int, placed: list[tuple[int, int]]):
        r = self.agent_radius
        margin = 2
        max_try = 1000
        for _ in range(max_try):
            x = random.randint(r + 5, self.screen_width - r - 5)
            y = random.randint(y_low + r, y_high - r)
            if all(math.hypot(x - px, y - py) >= 2 * r + margin for px, py in placed):
                return x, y
        raise RuntimeError("Failed adding disk")

    def reset(self, random=False):
        # pymunk physics
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)

        # damping for friction
        self.space.damping = 0.2

        self.discs = []
        self.agent_discs = []
        self.opponent_discs = []
        placed_xy = []

        spacing = 2 * self.agent_radius + 50

        # Add agent's discs
        for i in range(self.num_agent_discs):
            if not random:
                x = self.screen_width // 2 + (i - self.num_agent_discs // 2) * spacing
                y = self.screen_height - 100
            else:
                x, y = self._random_pos(self.screen_height // 2, self.screen_height, placed_xy)
            placed_xy.append((x, y))
            disc = self._add_disc((x, y))
            self.agent_discs.append(disc)
        
        # Add opponent's discs
        for i in range(self.num_opponent_discs):
            if not random:
                x = self.screen_width // 2 + (i - self.num_opponent_discs // 2) * spacing
                y = 100
            else:
                x, y = self._random_pos(0, self.screen_height // 2, placed_xy)
            placed_xy.append((x, y))
            disc = self._add_disc((x, y))
            self.opponent_discs.append(disc)

        return self._get_obs()

    def step(self, action, who, render=False):
        # whose turn?
        if who == 0:
            moving_discs = self.agent_discs
        else:
            moving_discs = self.opponent_discs
        
        disc_index = int(action[0])
        force = (
            np.clip(action[1], -1, 1) * self.max_force,
            np.clip(action[2], -1, 1) * self.max_force
        )

        # if invalid index input, just ignore
        if disc_index >= len(moving_discs) or disc_index < 0:
            if who == 1:
                print("INVALID CHOICE FLAG")
                print(disc_index)
            obs   = self._get_obs()
            reward = -1.0
            done   = False
            info   = {"invalid_action": True,
                        "action_mask": self.get_action_mask(who)}
            return obs, reward, done, info

        agent_before = len(self.agent_discs)
        opponent_before = len(self.opponent_discs)

        moving_disc = moving_discs[disc_index]
        moving_disc.apply_impulse_at_local_point(force, (0, 0))

        while True:
            if render:
                self.render()
            self.space.step(1 / 60.0)
            self._remove_out_of_bounds_discs()
            if self._all_discs_stopped(threshold=10):
                break

        self._remove_out_of_bounds_discs()

        obs = self._get_obs()
        reward = self._compute_reward(agent_before, opponent_before, who)
        done = self._check_done()
        mask = self.get_action_mask(who)
        info = {"action_mask": mask}
        
        return obs, reward, done, info

    def _get_obs(self):
        obs = []
        for disc in self.agent_discs:
            pos = disc.position
            obs.extend(
                [
                    (pos[0] - self.screen_width / 2) / (self.screen_width / 2),
                    (pos[1] - self.screen_height / 2) / (self.screen_height / 2),
                    0
                ]
            )
        
        # padding
        while len(obs) < self.num_agent_discs * 3:
            obs.extend([0.0, 0.0, -1])
        
        for disc in self.opponent_discs:
            pos = disc.position
            obs.extend(
                [
                    (pos[0] - self.screen_width / 2) / (self.screen_width / 2),
                    (pos[1] - self.screen_height / 2) / (self.screen_height / 2),
                    1
                ]
            )

        # padding
        while len(obs) < (self.num_agent_discs + self.num_opponent_discs) * 3:
            obs.extend([0.0, 0.0, -1])

        return np.array(obs, dtype=np.float32)
    
    def get_action_mask(self, who):
        if who == 0:
           mask = np.zeros(self.num_agent_discs, dtype=bool)
           mask[:len(self.agent_discs)] = True
        else:
            mask = np.zeros(self.num_opponent_discs, dtype=bool)
            mask[:len(self.opponent_discs)] = True
        return mask
    
    def _compute_reward(self, agent_before, opponent_before, who):
        if who == 0:
            reward = - (len(self.opponent_discs) - opponent_before + 1 * (agent_before - len(self.agent_discs)))
        if who == 1:
            reward = (1 * (len(self.opponent_discs) - opponent_before) + agent_before - len(self.agent_discs))
        return reward

    def _check_done(self):
        return len(self.agent_discs) == 0 or len(self.opponent_discs) == 0

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            
            self.board_img = pygame.image.load("img.png").convert()
            self.board_img = pygame.transform.scale(self.board_img, (self.screen_width, self.screen_height))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill((255, 255, 255))
        self.screen.blit(self.board_img, (0, 0))

        pygame.draw.rect(
            self.screen,
            (220, 220, 220),
            pygame.Rect(0, 0, self.screen_width, self.screen_height),
            width=5
        )

        for body in self.agent_discs:
            for shape in body.shapes:
                if isinstance(shape, pymunk.Circle):
                    pos = int(body.position.x), int(body.position.y)
                    radius = int(shape.radius)
                    color = (0, 0, 0)
                    pygame.draw.circle(self.screen, color, pos, radius)
        for body in self.opponent_discs:
            for shape in body.shapes:
                if isinstance(shape, pymunk.Circle):
                    pos = int(body.position.x), int(body.position.y)
                    radius = int(shape.radius)

                    color = (230, 230, 230)
                    pygame.draw.circle(self.screen, color, pos, radius)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen:
            pygame.quit()