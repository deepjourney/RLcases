import numpy as np
from algos import getAlgo
import agents
from agents.wrappers import Memo, Stack, Imagine
class MyAgent(agents.Agent):
    def __init__(self,env,envinfo,args):
        agents.Agent.__init__(self,args)
        self.attr.stack_shape = env.observation_space.shape
        self.attr.stack_dtype = envinfo['obs'].dtype
        obs_space = env.observation_space
        act_space = env.action_space
        print(args.env_name,obs_space,act_space)
        self.algo = getAlgo(obs_space,act_space,args)
    def memoexps(self, new_obs, rew, done, info):
        self.algo.memoexps(new_obs, rew, done, info)
    def getaction(self, obs, explore):
        #act, act_info = np.zeros([self.attr.args.env_num],dtype=np.int64), {}
        act, act_info = self.algo.get_action(obs,explore)
        return act, act_info
    def update(self, crt_step, max_step, info_in):
        info_in['mb_obs']     = np.array(info_in['mb_obs'])
        info_in['mb_act']     = np.array(info_in['mb_act'])
        info_in['mb_new_obs'] = np.array(info_in['mb_new_obs'])
        info_in['mb_rew']     = np.array(info_in['mb_rew'])
        info_in['mb_done']    = np.array(info_in['mb_done'])
        self.algo.update(crt_step=crt_step, max_step=max_step, info_in=info_in)
    def save(self,name):
        self.algo.save(name)
    def load(self):
        self.algo.load()

"""def agt_maker(agt_name, i, agt_seed, args):
    def __make_agt():
        agt = agents.make(agt_name)
        agt = Memo(agt, i, args)
        return agt
    return __make_agt"""
def fAgent(env,envinfo,args):
    #agt = [agt_maker(args.agt_name, i, args.agt_seed, args) for i in range(args.agt_num)]
    agt = MyAgent(env,envinfo,args)
    if args.memoplace == "agtcpu": agt = Memo(agt)
    agt = Stack(agt)
    agt = Imagine(agt)
    return agt
