import numpy as np
import gymnasium as gym
import ale_py
import easydict, random, json

gym.register_envs(ale_py)

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, WarpFrame, ClipRewardEnv,
)
from envirs.warppers import Recorder, Monitor, wrap_deepmind_render

def env_maker(env_name, i, env_seed, args):
    def __make_env():
        render_mode = 'rgb_array' if args.render else None
        if args.gameflag == 'atari':
            env = gym.make(env_name, render_mode=render_mode)
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            # Recorder sits after frame-skip but before reward-clip / episodic-life,
            # so it logs the TRUE per-episode game score with correct step counting.
            env = Recorder(env, i, args)
            if args.render:
                env = Monitor(env, i, args, 'org_')
            env = EpisodicLifeEnv(env)
            if 'FIRE' in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = WarpFrame(env)
            env = ClipRewardEnv(env)
            if args.render:
                env = wrap_deepmind_render(env)
                env = Monitor(env, i, args)
        else:
            env = gym.make(env_name, render_mode=render_mode)
            env = Recorder(env, i, args)
            if args.render:
                env = Monitor(env, i, args)
        try:
            env.action_space.seed(i + env_seed)
        except Exception:
            pass
        random.seed(i + env_seed)
        np.random.seed(i + env_seed)
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
