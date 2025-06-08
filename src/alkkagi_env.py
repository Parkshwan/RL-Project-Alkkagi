import gym
from gym import spaces
import numpy as np
import pymunk
import pymunk.pygame_util
import pygame
import math

class AlkkagiEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_discs_per_player=1):
        super(AlkkagiEnv, self).__init__()
        self.screen_width = 600
        self.screen_height = 600
        self.agent_radius = 15
        self.max_force = 1000

        self.num_discs_per_player = num_discs_per_player
        self.num_total_discs = 2 * num_discs_per_player
        self.discs = []

        single_space = spaces.Tuple((
            spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),  # pos
            spaces.Discrete(2),  # 0 agent, 1 opponent
            spaces.Discrete(2),  # 0 not removed, 1 removed
        ))
        self.observation_space = spaces.Tuple([single_space] * self.num_total_discs)

        self.action_space = spaces.Tuple((
            spaces.Discrete(self.num_total_discs),
            spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        ))

        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.6

        self.screen = None
        self.clock = None
        self.draw_options = None

    def _add_disc(self, position, index, team):
        mass = 1
        radius = self.agent_radius
        inertia = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Circle(body, radius)
        shape.elasticity = 0.7
        
        body.index = index
        body.team = team
        body.removed = False
        
        self.space.add(body, shape)
        self.discs.append(body)

        return body

    def _remove_out_of_bounds_discs(self):
        for disc in self.discs:
            if disc.removed: continue
            
            x, y = disc.position
            if not (0 <= x <= self.screen_width and 0 <= y <= self.screen_height):
                for shape in disc.shapes:
                    self.space.remove(shape)
                self.space.remove(disc)
                disc.removed = True
    
    def _all_discs_stopped(self, threshold=5.0):
        all_stopped = True
        for disc in self.discs:
            speed = disc.velocity.length
            # print(disc.velocity.length)
            if speed < threshold:
                # threshold 이하이면 속도를 0으로 만들어 정지시킴
                disc.velocity = pymunk.Vec2d(0, 0)
                disc.angular_velocity = 0
            else:
                all_stopped = False
        return all_stopped

    def reset(self, fixed=False):
        self.space = pymunk.Space()  # 충돌 방지 위해 space도 새로 생성
        self.space.gravity = (0, 0)
        self.space.damping = 0.6
        self.discs = []  # 이전 디스크 초기화
        
        radius = self.agent_radius
        spacing = 2 * radius + 5

        def random_position(team):
            attempts = 0
            while attempts < 100:
                x = np.random.uniform(radius, self.screen_width - radius)
                y_range = (radius, self.screen_height / 2 - radius) if team == 1 else (self.screen_height / 2 + radius, self.screen_height - radius)
                y = np.random.uniform(*y_range)
                pos = (x, y)
                if all(np.linalg.norm(np.array(pos) - np.array(d.position)) >= spacing for d in self.discs):
                    return pos
                attempts += 1
            raise ValueError("Failed to find non-overlapping position")

        indices = np.random.permutation(self.num_total_discs)

        for i in range(self.num_total_discs):
            team = 0 if i < self.num_discs_per_player else 1
            if fixed:
                x = self.screen_width // 2 + ((i % self.num_discs_per_player) - self.num_discs_per_player // 2) * spacing
                y = self.screen_height - 100 if team == 0 else 100
                pos = (x, y)
            else:
                pos = random_position(team)
            self._add_disc(pos, index=int(indices[i]), team=team)

        return self._get_obs()

    def step(self, action, who):
        target_index = int(action[0])
        direction = np.clip(action[1], -1, 1) * self.max_force

        target = next((d for d in self.discs if d.index == target_index),None)

        target.apply_impulse_at_local_point(direction, (0, 0))
        
        while True:
            self.render()
            self.space.step(1 / 60.0)
            self._remove_out_of_bounds_discs()
            if self._all_discs_stopped(threshold=0.1):
                break

        self._remove_out_of_bounds_discs()

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()
        info = {"action_mask": self.get_action_mask(who)}
        return obs, reward, done, info

    def _get_obs(self):
        obs = []
        sorted_discs = sorted(self.discs, key=lambda d: d.index)
        for disc in sorted_discs:
            pos = (
                (disc.position[0] - self.screen_width / 2) / (self.screen_width / 2),
                (disc.position[1] - self.screen_height / 2) / (self.screen_height / 2),
            )
            obs.extend([pos[0], pos[1], disc.team, int(disc.removed)])
        return np.array(obs, dtype=np.float32)


    
    def get_action_mask(self, who):
        mask = np.zeros(self.num_total_discs, dtype=bool)

        for d in self.discs:
            if d.team == who and not d.removed:
                mask[d.index] = True
        return mask

    
    def _compute_reward(self):
        num_agent_removed = sum(1 for d in self.discs if d.team == 0 and d.removed)
        num_opponent_removed = sum(1 for d in self.discs if d.team == 1 and d.removed)
        return num_opponent_removed - num_agent_removed


    def _check_done(self):
        all_agent_out = all(d.removed for d in self.discs if d.team == 0)
        all_opponent_out = all(d.removed for d in self.discs if d.team == 1)
        return all_agent_out or all_opponent_out

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill((255, 255, 255))
        pygame.draw.rect(self.screen, (220, 220, 220), pygame.Rect(0, 0, self.screen_width, self.screen_height), width=5)

        for body in self.discs:
            for shape in body.shapes:
                if isinstance(shape, pymunk.Circle):
                    pos = int(body.position.x), int(body.position.y)
                    radius = int(shape.radius)
                    color = (20, 20, 20) if body.team == 0 else (235, 235, 235)
                    pygame.draw.circle(self.screen, color, pos, radius)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen:
            pygame.quit()