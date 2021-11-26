#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-10 15:28:30
@LastEditor: John
LastEditTime: 2021-09-16 00:52:30
@Discription: 
@Environment: python 3.7.7
'''
import gym
import numpy as np


class NormalizedActions(gym.ActionWrapper):
    ''' 将action范围重定在[0.1]之间
    '''

    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action

    def reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        return action


class OUNoise(object):
    '''Ornstein–Uhlenbeck噪声
    '''

    def __init__(self, action_space=1, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu  # OU噪声的参数
        self.theta = theta  # OU噪声的参数
        self.sigma = max_sigma  # OU噪声的参数
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = 1
        self.low = -1
        self.high = 1
        self.reset()

    def reset(self):
        self.obs = np.ones(self.action_dim) * self.mu

    def evolve_obs(self):
        x = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.obs = x + dx
        return self.obs

    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)  # sigma会逐渐衰减
        return np.clip(action + ou_obs, self.low, self.high)  # 动作加上噪声后进行剪切


class Area1Env:
    # 系统状态
    __delta_f = 0
    __delta_Pg = 0
    __delta_Pm = 0
    __delta_PL = 0
    i_delta_ACE = 0
    d_delta_ACE = 0

    def __init__(self):
        self.cons_Tg = 0.1
        self.cons_Tt = 0.4
        self.cons_H = 0.0833
        self.cons_D = 0.0015
        self.cons_R = 0.33
        self.delta_t = 0.01

        self.observation_dim = 3
        self.action_dim = 1

    def reset(self):
        self.__delta_f = 0
        self.__delta_Pg = 0
        self.__delta_Pm = 0
        self.__delta_PL = 0
        self.i_delta_ACE = 0
        self.d_delta_ACE = 0
        return np.array([0, 0, 0])

    def step(self, action, time, state):

        if time < 2000:
            self.__delta_PL = 0
        else:
            if time < 5000:
                self.__delta_PL = -0.01
            else:
                self.__delta_PL = 0.01

        if time < 20000:
            done = False
        else:
            done = True

        delta_Pg_ = (1 / self.cons_Tg * (
                    action[0] - 1 / self.cons_R * self.__delta_f - self.__delta_Pg)) * self.delta_t + self.__delta_Pg
        self.__delta_Pg = delta_Pg_

        delta_Pm_ = (1 / self.cons_Tt * (self.__delta_Pg - self.__delta_Pm)) * self.delta_t + self.__delta_Pm
        self.__delta_Pm = delta_Pm_

        delta_f_ = (1 / (2 * self.cons_H) * (
                self.__delta_Pm - self.__delta_PL - self.cons_D * self.__delta_f)) * self.delta_t + self.__delta_f
        self.__delta_f = delta_f_

        s_ = (1/0.33+0.015)*self.__delta_f    #ACE
        self.i_delta_ACE += s_
        self.d_delta_ACE = (s_-state[0])/self.delta_t
        reward = -abs(self.__delta_f)
        # reward = -abs(s_)

        return np.array([s_, self.i_delta_ACE, self.d_delta_ACE]), reward, done

    def render(self):
        pass


def action1(action):
    low_bound = -1
    upper_bound = 1
    action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
    action = np.clip(action, low_bound, upper_bound)

    return action


def reverse_action(action):
    low_bound = -1
    upper_bound = 1
    action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
    action = np.clip(action, low_bound, upper_bound)
    return action