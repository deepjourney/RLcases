import numpy as np
from models import getModel
import agents
from collections import deque
import copy
class ReconAgent(agents.Agent):
    def __init__(self,env,obs,args):
        agents.Agent.__init__(self,env,obs,args)
        obs_space = env.observation_space
        self.org_act_space = env.spec._kwargs['org_action_space']
        self.ext_act_space = env.spec._kwargs['ext_action_space']
        print(args.env_name,obs_space,self.org_act_space)
        self.model     = getModel(obs_space,self.org_act_space,args)
        self.model.load(folder=args.action_model)
        self.memo_ext_act     = deque(maxlen=args.memo_size)
        self.memo_ext_act_info= deque(maxlen=args.memo_size)
        print(args.env_name,obs_space,self.ext_act_space,'(ext_act)')

        ext_args = copy.deepcopy(args)
        ext_args.apfparas = "1=512=32^32"
        self.ext_model = getModel(obs_space,self.ext_act_space,ext_args)
    def memoexps(self, new_obs, rew, done, info):
        pass
    def getaction(self, obs, explore):
        act,     act_info     = self.model.get_action(obs,explore=False)
        ext_act, ext_act_info = self.ext_model.get_action(obs,explore)
        self.memo_ext_act.append(ext_act)
        self.memo_ext_act_info.append(ext_act_info)
        if self.org_act_space.__class__.__name__ == "Discrete":
            act = ext_act*self.org_act_space.n+act
        if self.org_act_space.__class__.__name__ == "Box":
            ext_act = np.expand_dims(ext_act,axis=-1)
            ext_act_norm = ext_act/(self.ext_act_space.n-1)*1.99-1
            act = np.concatenate((act,ext_act_norm),axis=-1)
        return act, act_info
    def update(self, crt_step, max_step, info_in):
        info_in =  {'mb_ext_act':       np.array(self.memo_ext_act),
                    'mb_ext_act_info':  self.memo_ext_act_info, **info_in}
        self.ext_model.update(mb_obs_stack=info_in['mb_obs'], mb_act=info_in['mb_ext_act'], mb_new_stack=info_in['mb_new_obs'], mb_rew=info_in['mb_rew'], mb_done=info_in['mb_done'], \
                info_es=info_in['mb_info'], info_ps=info_in['mb_ext_act_info'], crt_step=crt_step, max_step=max_step)
    def save(self,name):
        self.ext_model.save(name,'ext')
    def load(self):
        self.ext_model.load('ext')

def fAgent(env,obs,args):
    agt = ReconAgent(env,obs,args)
    agt = agents.Memo(agt)
    agt = agents.Stack(agt)
    agt = agents.Imagine(agt)
    return agt
