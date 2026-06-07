import numpy as np
import gym, ale_py; gym.register_envs(ale_py)
import gym, easydict, cv2, random, scipy, json
from envirs.warppers import Recorder, Monitor, wrap_deepmind_render
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
#from pysc2.env import sc2_env
def env_maker(env_name, i, env_seed, args):
    def __make_env():
        if args.gameflag=='atari':
            env = make_atari(env_name)
        #elif args.gameflag=='sc2':
        #    env = sc2_env.SC2Env(env_name)#,tep_mul=step_mul,visualize=True)
        else:
            env = gym.make(env_name)
        if hasattr(env,'attr'): env.spec._kwargs['attr']=env.attr
        env.seed(i+env_seed)
        random.seed(i+env_seed)
        np.random.seed(i+env_seed)
        env = Recorder(env, i, args)
        if args.gameflag=='atari':
            if args.render:
                env = Monitor(env, i, args, 'org_')
            env = wrap_deepmind(env)
            env = wrap_deepmind_render(env)
        if args.render:
            env = Monitor(env, i, args)
        return env
    return __make_env
from envirs.warppers import VecNormalize
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
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
        print(json.dumps(env_args),file=fenvinfo)
    env = [env_maker(args.env_name, i, args.env_seed, args) for i in range(args.env_num)]
    if len(env) > 1: env = ShmemVecEnv(env)
    else:            env = SubprocVecEnv(env)#DummyVecEnv(env) #self._save_obs(e, obs);could not broadcast input array from shape (3,9,9,3) into shape (9,9,3)
    #if len(env.observation_space.shape) == 1:
    #    env = VecNormalize(env, gamma=0.99)
    envinfo = {}
    obs = env.reset()
    #print('env obs shape: ',obs.shape)
    envinfo['obs'] = obs
    return env, envinfo
