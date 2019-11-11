import argparse
import numpy as np
import tensorflow as tf
import os.path as osp
import os

from ppo import build_model
from utils import get_env_type, get_default_network

from baselines.common.cmd_util import make_vec_env
from baselines.common.tf_util import get_session
from baselines.common.vec_env import VecEnvWrapper
from baselines import logger
import datetime


class VecNormalize(VecEnvWrapper):
    class RMS:
        def __init__(self, mean, var):
            self.mean = mean
            self.var = var

        def update(self, *args, **kwargs):
            pass

    def __init__(self, venv, ob_rms, ret_rms, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = self.RMS(*ob_rms)
        self.ret_rms = self.RMS(*ret_rms)
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

        assert self.ob_rms.mean.shape == self.ob_rms.var.shape == self.observation_space.shape

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        for e, info in enumerate(infos):
            info['ob_unorm'] = obs[e]
            info['rew_unorm'] = rews[e]

        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)

        self.ret_rms.update(self.ret)
        rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.

        return obs, rews, news, infos

    def _obfilt(self, obs):
        self.ob_rms.update(obs)
        obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
        return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)

    def __del__(self):
        if self.ob_rms:
            np.save(osp.join(logger.get_dir(), 'ob_rms_mean.npy'), self.ob_rms.mean)
            np.save(osp.join(logger.get_dir(), 'ob_rms_var.npy'), self.ob_rms.var)
        if self.ret_rms:
            np.save(osp.join(logger.get_dir(), 'ret_rms_mean.npy'), self.ret_rms.mean)
            np.save(osp.join(logger.get_dir(), 'ret_rms_var.npy'), self.ret_rms.var)


def build_mujoco(env_id, ob_rms, ret_rms, num_env=4, seed=2019, reward_scale=1., ):
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    env = make_vec_env(env_id, 'mujoco', num_env, seed, reward_scale=reward_scale)

    env = VecNormalize(env, ob_rms=ob_rms, ret_rms=ret_rms)
    return env

def load_rms(rms_type):
    try:
        ob_rms_mean = np.load(osp.join(path, '{}_rms_mean.npy'.format(rms_type)))
    except FileNotFoundError as e:
        print(e)
        ob_rms_mean = 0.
    try:
        ob_rms_var = np.load(osp.join(path, '{}_rms_var.npy'.format(rms_type)))
    except FileNotFoundError as e:
        print(e)
        ob_rms_var = 1.
    return ob_rms_mean, ob_rms_var


def collect_one_trajectory(model, env, trajectory_length):
    assert hasattr(env, 'num_envs')
    for it in range(10):   # max try
        obs_buffer, acs_buffer, rews_buffer = [], [], []
        obs = env.reset()
        rets = 0.
        success = True
        for t in range(1, trajectory_length+1):
            acs, *_ = model.step(obs)
            obs_new, rews, dones, infos = env.step(acs)

            rets += rews[0]
            obs_buffer.append(infos[0]['ob_unorm'])
            acs_buffer.append(acs[0])
            rews_buffer.append(infos[0]['rew_unorm'])

            obs = obs_new
            if dones[0] and t < trajectory_length:
                print('a fail trajectory with length: {}'.format(t))
                success = False
                break
        if success:
            print('collect a trajectory with length:{}, normalized return:{:.2f}, true return:{:.2f}'.format
                  (trajectory_length, rets, np.sum(rews_buffer)))
            trajectory = {'obs': np.array(obs_buffer),
                          'acs': np.array(acs_buffer),
                          'rews': np.array(rews_buffer)}
            return trajectory
    print('Can not collect a trajectory with length {}'.format(trajectory_length))
    return None


def collect_many_trajectory(model, env, nbatch=50, trajectory_length=1000):
    num_env = env.num_envs if hasattr(env, 'num_envs') else 1

    assert num_env == 1
    obs_buffer = np.zeros(shape=[nbatch, trajectory_length, *env.observation_space.shape], dtype=env.observation_space.dtype)
    acs_buffer = np.zeros(shape=[nbatch, trajectory_length, *env.action_space.shape], dtype=env.action_space.dtype)
    rews_buffer = np.zeros(shape=[nbatch, trajectory_length], dtype=np.float32)
    eprets_buffer = np.zeros(shape=[nbatch], dtype=np.float32)

    pointer = 0
    while True:
        trajectory = collect_one_trajectory(model, env, trajectory_length)
        if trajectory:
            obs_buffer[pointer] = trajectory['obs']
            acs_buffer[pointer] = trajectory['acs']
            rews_buffer[pointer] = trajectory['rews']
            eprets_buffer[pointer] = np.sum(trajectory['rews'])
            pointer += 1
        if pointer % 10 == 0:
            print('************************Collect {}/{} trajectory*********************'.format(pointer, nbatch))
        if pointer == nbatch:
            break
    results = {
        'obs': obs_buffer,
        'acs': acs_buffer,
        'rews': rews_buffer,
        'ep_rets': eprets_buffer,
    }
    return results

def main(path):
    assert 'ppo' in path
    env_id = path.split('-')[1] + '-' + path.split('-')[2]
    logdir = osp.join('dataset', env_id, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))
    os.makedirs(logdir, exist_ok=True)
    logger.configure(dir=logdir)

    env = build_mujoco(env_id, num_env=1, ob_rms=load_rms('ob'), ret_rms=load_rms('ret'))

    model = build_model(
        env=env,
        network=get_default_network(env),
        scope='ppo2_model',
        seed=2019,
        load_path=osp.join(path, 'checkpoints', '10000000.model')
    )
    # sample_one_trajectory(model=model, env=env, trajectory_length=1000)
    trajectory = collect_many_trajectory(model=model, env=env, nbatch=50, trajectory_length=1000)
    savepath = osp.join(logger.get_dir(), 'deterministic.ppo.{}.0.00.npz'.format(env_id))
    np.save(savepath, trajectory)
    print('saving into: {}'.format(savepath))


if __name__ == '__main__':
    path = 'expert/ppo2-Ant-v2-2019-11-10-21-45-35-800647'
    main(path)
