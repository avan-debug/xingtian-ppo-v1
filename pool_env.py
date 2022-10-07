# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Make atari env for simulation."""
import random
import time
import envpool
import numpy as np

from xt.environment.environment import Environment
from xt.environment.gym import infer_action_type
from xt.environment.gym.atari_wrappers import make_atari
from zeus.common.util.register import Registers


@Registers.env
class SinglePool(Environment):
    def init_env(self, env_info):
        print('[GGLC] SinglePool created')
        self._env = envpool.make(
            env_info['name'].replace('NoFrameskip-v4', '-v5'),
            env_type='gym',
            num_envs=1,
            frame_skip=4,
            episodic_life=True,
            stack_num=4,
            noop_max=30,
            seed=random.randint(0, 1000),
        )
        self.dim = env_info.get('dim', 84)
        self.last_state = np.zeros((self.dim, self.dim, 4))
        self.init_state = None

        gym_env = make_atari(env_info)
        self.action_type = infer_action_type(gym_env.action_space)
        return gym_env

    def init_stack_obs(self, num):
        pass

    def reset(self):
        self.init_state = self.last_state
        return self.last_state

    def step(self, action: int, agent_index=0):
        obs, reward, done, info = self._env.step(np.array([action]))
        obs = obs.transpose(0, 2, 3, 1)
        _info = {'real_done': info['lives'][0] == 0, 'eval_reward': info['reward'][0], 'ale.lives': info['lives'][0]}
        self.last_state = obs[0]
        return obs[0], reward[0], done[0], _info


@Registers.env
class EnvPool(Environment):
    """use envpool to speedup."""
    def init_env(self, env_info):
        print('[GGLC] EnvPool created')
        self.size = env_info.get("size")
        self.wait_nums=env_info.get("wait_nums")
        self.thread_affinity_offset=env_info.get("thread_affinity_offset",-1)
        self.name = env_info.get("name").replace('NoFrameskip-v4', '-v5')
        assert self.size is not None and self.name is not None, "envpool must assign 'name' and 'size'."

        self.pool = envpool.make(
            task_id=self.name,
            env_type='gym',
            num_envs=self.size,
            frame_skip=4,
            episodic_life=True,
            stack_num=4,
            noop_max=30,
            seed=random.randint(0, 10000),
            repeat_action_probability=.0,
            batch_size=self.wait_nums,
            num_threads=self.size,
            thread_affinity_offset=self.thread_affinity_offset,
            max_episode_steps=108000,
        )

        self.spec = envpool.make_spec(self.name)
        self.dim = env_info.get('dim', 84)
        self.action_type = infer_action_type(self.spec.action_space)
        self.init_state = None
        self.last_state = [np.zeros((self.dim, self.dim, 4)) for _ in range(self.size)]
        self.env_ids=np.array(list(range(self.size)))

    def reset(self):
        self.init_state = self.last_state
        return self.last_state

    def step(self, action,agent_index=0):
        _start = time.time()
        self.pool.send(np.array(action),self.env_ids)
        obs, rew, done, info = self.pool.recv()
        obs = obs.transpose(0, 2, 3, 1)
        
        self.last_state = obs
        self.env_ids=info["env_id"]

        _info = []
        
        for env_id in range(self.size):
            _info.append({'real_done': info['lives'][env_id] == 0 and done[env_id], 'ale.lives': info['lives'][env_id]})

        return list(obs), list(rew), list(done), _info

    def get_env_info(self):
        self.reset()
        env_info = {
            "n_agents": 1,
            "api_type": 'standalone',
            "agent_ids": [0],
            "action_type": self.action_type
        }
        return env_info

    def close(self):
        self.pool.close()
