import numpy as np
import gymnasium as gym
import easydict, random, json

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from envirs.warppers import Recorder, Monitor, wrap_deepmind_render

class _GymCompat(gym.Wrapper):
    """Adapts gymnasium 5-tuple API to the legacy 4-tuple used internally."""
    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs
    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        return obs, rew, terminated or truncated, info
    def seed(self, seed=None):
        pass

def env_maker(env_name, i, env_seed, args):
    def __make_env():
        if args.gameflag == 'atari':
            env = gym.make(env_name)
            env = _GymCompat(env)
            env.seed(i + env_seed)
            random.seed(i + env_seed)
            np.random.seed(i + env_seed)
            env = Recorder(env, i, args)
            if args.render:
                env = Monitor(env, i, args, 'org_')
            env = AtariWrapper(env)
            env = wrap_deepmind_render(env)
        else:
            env = _GymCompat(gym.make(env_name))
            env.seed(i + env_seed)
            random.seed(i + env_seed)
            np.random.seed(i + env_seed)
            env = Recorder(env, i, args)
        if args.render:
            env = Monitor(env, i, args)
        return env
    return __make_env

def fEnv(args):
    env_args = easydict.EasyDict()
    env_args.timer    = args.timer
    env_args.env_seed = args.env_seed
    env_args.env_name = args.env_name
    env_args.play_num = args.play_num
    env_args.type_num = args.type_num
    env_args.unit_num = args.unit_num
    env_args.envparas = args.envparas
    env_args.npcparas = args.npcparas
    env_args.agtparas = args.agtparas
    env_args.envonoff = args.envonoff
    env_args.learnflag= args.learnflag
    env_args.render   = args.render
    env_args.zoom_in  = args.zoom_in
    with open('./myenv/envinfo.json', 'w') as fenvinfo:
        print(json.dumps(env_args), file=fenvinfo)
    envs = [env_maker(args.env_name, i, args.env_seed, args) for i in range(args.env_num)]
    if len(envs) > 1:
        env = SubprocVecEnv(envs)
    else:
        env = DummyVecEnv(envs)
    envinfo = {}
    obs = env.reset()
    envinfo['obs'] = obs
    return env, envinfo
