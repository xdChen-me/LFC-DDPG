#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 20:58:21
@LastEditor: John
LastEditTime: 2021-09-16 01:31:33
@Discription: 
@Environment: python 3.7.7
'''
import sys, os

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加父路径到系统路径sys.path

import datetime
import gym
import torch
import numpy as np

from DDPG.env import NormalizedActions, OUNoise, Area1Env, action1, reverse_action
from DDPG.agent import DDPG
from common.utils import save_results, make_dir
from common.plot import plot_rewards, plot_rewards_cn

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class DDPGConfig:
    def __init__(self):
        self.algo = 'DDPG'  # 算法名称
        self.env = 'Area1Env'  # 环境名称
        self.result_path = curr_path + "/outputs/" + self.env + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
        self.train_eps = 20  # 训练的回合数
        self.eval_eps = 1  # 测试的回合数
        self.gamma = 0.99  # 折扣因子
        self.critic_lr = 0.001  # 评论家网络的学习率
        self.actor_lr = 0.001  # 演员网络的学习率
        self.memory_capacity = 8000
        self.batch_size = 128
        self.target_update = 2
        self.hidden_dim = 256
        self.soft_tau = 1e-2  # 软更新参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def env_agent_config(cfg, seed=1):
    env = Area1Env()
    state_dim = env.observation_dim
    action_dim = env.action_dim
    agent = DDPG(state_dim, action_dim, cfg)
    return env, agent


def train(cfg, env, agent):
    print('开始训练！')
    print(f'环境：{cfg.env}，算法：{cfg.algo}，设备：{cfg.device}')
    ou_noise = OUNoise()  # 动作噪声
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录平均奖励
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ou_noise.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action = agent.choose_action(state)
            # action = action1(action)
            # action = np.array([action])
            action = ou_noise.get_action(action, i_step)
            # action = reverse_action(action)
            next_state, reward, done = env.step(action, i_step, state)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
        #if (i_ep + 1) % 10 == 0:
        print('回合：{}/{}，奖励：{:.2f}'.format(i_ep + 1, cfg.train_eps, ep_reward))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('完成训练！')
    return rewards, ma_rewards


def eval(cfg, env, agent):
    print('开始测试！')
    print(f'环境：{cfg.env}, 算法：{cfg.algo}, 设备：{cfg.device}')
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    state_save = []
    for i_ep in range(cfg.eval_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action = agent.choose_action(state)
            action = np.array([action])
            next_state, reward, done, = env.step(action, i_step, state)
            state_save.append(next_state[0])
            ep_reward += reward
            state = next_state
        print('回合：{}/{}, 奖励：{}'.format(i_ep + 1, cfg.train_eps, ep_reward))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('完成测试！')
    return rewards, ma_rewards, state_save


if __name__ == "__main__":
    cfg = DDPGConfig()
    '''
    # 训练
    perfect_path = curr_path + "/outputs/" + '/Area1Env/' + \
   '/' + '/20211125-175035/' + '/models/'  # 保存模型的路径

    env, agent = env_agent_config(cfg, seed=1)
    agent.load(path=perfect_path)

    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    '''
    # 测试
    perfect_path = curr_path + "/outputs/" + '/Area1Env/' + \
      '/' + '/20211125-202036/' + '/models/'  # 保存模型的路径
    env, agent = env_agent_config(cfg, seed=10)
    agent.load(path=perfect_path)
    rewards, ma_rewards, state_save = eval(cfg, env, agent)
    # save_results(rewards, ma_rewards,  tag='eval', path=cfg.result_path)



    plt.figure(0)
    plt.plot(state_save)

    #plt.xlim((0, L))
    # plt.ylim((min(feedback_list) - 0.5, max(feedback_list) + 0.5))
    plt.xlabel('time (s)')
    plt.ylabel('ACE (p.u.)')
    plt.title('TEST DDPG')
    # plt.ylim((1 - 0.5, 1 + 0.5))

    plt.grid(True)
    plt.show()
    # plot_rewards_cn(rewards, ma_rewards, tag="eval", env=cfg.env, algo=cfg.algo, path=cfg.result_path)
