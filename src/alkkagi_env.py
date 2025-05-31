import gym
from gym import spaces
import numpy as np
import pymunk
import pymunk.pygame_util
import pygame
import math

class AlkkagiEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agent_discs=1, num_opponent_discs=1):
        super(AlkkagiEnv, self).__init__()
        self.screen_width = 600
        self.screen_height = 600
        self.agent_radius = 15

        self.num_agent_discs = num_agent_discs
        self.num_opponent_discs = num_opponent_discs
        self.discs = []
        self.agent_discs = []
        self.opponent_discs = []

        # 관측 공간: (x, y, vx, vy) * (num_agent + num_opponent)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=((num_agent_discs + num_opponent_discs) * 4,),
            dtype=np.float32
        )

         # 행동 공간: 각 step에 agent가 한 개의 디스크만 조작한다고 가정 (x, y 방향)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)  # (디스크 인덱스, x, y)

        # pymunk physics
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)

        # 공간 전체에 약한 감속 적용
        self.space.damping = 0.9  # 1.0이면 감속 없음

        # pygame rendering
        self.screen = None
        self.clock = None
        self.draw_options = None

        # self.reset()

    def _add_disc(self, position):
        mass = 1
        radius = self.agent_radius
        inertia = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Circle(body, radius)
        shape.elasticity = 0.9
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
        self.discs = new_discs
    
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

    def reset(self):
        spacing = 2 * self.agent_radius + 5

        # Add agent's discs
        for i in range(self.num_agent_discs):
            x = self.screen_width // 2 + (i - self.num_agent_discs // 2) * spacing
            y = self.screen_height - 100
            disc = self._add_disc((x, y)) # add to discs list
            self.agent_discs.append(disc) # add to agent_discs list
        
        # Add opponent's discs
        for i in range(self.num_opponent_discs):
            x = self.screen_width // 2 + (i - self.num_opponent_discs // 2) * spacing
            y = 100
            disc = self._add_disc((x, y))
            self.opponent_discs.append(disc)

        return self._get_obs()

    def step(self, action):
        disc_index = int(np.clip(action[0], 0, len(self.agent_discs) - 1))
        direction = np.clip(action[1:], -1, 1)

        if np.linalg.norm(direction) > 1e-6:
            direction = direction / np.linalg.norm(direction)
        else:
            direction = np.array([0.0, -1.0])  # 기본 방향

        agent_before = len(self.agent_discs)
        opponent_before = len(self.opponent_discs)

        force = tuple(direction * 25)
        agent_disc = self.agent_discs[disc_index]
        agent_disc.apply_impulse_at_local_point(force, (0, 0))

        while True:
            self.space.step(1 / 60.0)
            self._remove_out_of_bounds_discs()
            if self._all_discs_stopped(threshold=0.1):
                break
        
        # TODO
        # Opponent's action with policy

        self._remove_out_of_bounds_discs()

        obs = self._get_obs()
        reward = self._compute_reward(agent_before, opponent_before)
        done = self._check_done()
        return obs, reward, done, {}

    def _get_obs(self):
        obs = []
        for disc in self.discs:
            pos = disc.position
            vel = disc.velocity
            obs.extend([
                (pos[0] - self.screen_width / 2) / (self.screen_width / 2),
                (pos[1] - self.screen_height / 2) / (self.screen_height / 2),
                vel[0] / 1000,
                vel[1] / 1000
            ])
        # 패딩: 디스크가 사라졌을 경우 0으로 채움
        while len(obs) < (self.num_agent_discs + self.num_opponent_discs) * 4:
            obs.extend([0.0, 0.0, 0.0, 0.0])
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, agent_before, opponent_before):
        # reward = num(removed opponent) - num(removed agent)
        return len(self.opponent_discs) - opponent_before + agent_before - len(self.agent_discs)

    def _check_done(self):
        return len(self.agent_discs) == 0 or len(self.opponent_discs) == 0

    def render(self, mode='human'):
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

        # 보드 경계 사각형 그리기
        pygame.draw.rect(
            self.screen,
            (220, 220, 220),
            pygame.Rect(0, 0, self.screen_width, self.screen_height),
            width=5
        )

        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen:
            pygame.quit()
