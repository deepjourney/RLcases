import numpy as np
from algos import getAlgo
import agents
from agents.wrappers import Memo, Stack, Imagine
import easydict, copy, time
from pprint import pprint
class TeamAgent(agents.Agent):
    def __init__(self,env,envinfo,args,dsgn):
        agents.Agent.__init__(self,env,envinfo,args,dsgn)
        obs_space = env.observation_space
        act_space = env.action_space
        print(args.env_name,obs_space,act_space)
        self.algo = getAlgo(obs_space,act_space,args)
        if self.attr.args.timer:
            self.getaction_time = 0
    def __del__(self):
        if self.attr.args.timer:
            print('TeamAgent getaction_time:',round(self.getaction_time/60,2),' minutes')
            print(self.attr.dsgns)
    def memoexps(self, new_obs, rew, done, info):
        self.algo.memoexps(new_obs, rew, done, info)
    def getaction(self, obs, explore):
        getaction_start = time.process_time()
        if self.attr.dsgn=='0_0_0': act, act_info = self.algo.get_action(obs,explore)
        else:                       act, act_info = np.zeros([self.attr.args.env_num,3],dtype=np.int64), {}# default is np.float64
        self.getaction_time += time.process_time()-getaction_start
        return act, act_info
    def update(self, crt_step, max_step, info_in):
        #print('TeamAgent update', self.attr.dsgn)
        if self.attr.dsgn=='0_0_0':
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

def fAgent(env,envinfo,args):
    envinfo['obs'] = np.array([envinfo['obs'] for i in range(3)]).swapaxes(0,1)
    master_dsgn = '0'
    squads = []
    for isquad in range(args.agt_num):
        squad_dsgn = master_dsgn+'_'+str(isquad)
        teams = []
        for iteam in range(args.type_num):
            team_dsgn = squad_dsgn+'_'+str(iteam)
            team = TeamAgent(env,envinfo,args,team_dsgn)
            if args.memoplace == "agtcpu": team = Memo(team)
            team = Stack(team)
            teams.append(team)
        squad = Combine_Teams(teams,squad_dsgn)
        squads.append(squad)
    master = Combine_Squads(squads,master_dsgn)
    return master

class Combine(object):
    def __init__(self, agts, dsgn):
        self.agts, self.attr = agts, easydict.EasyDict()
        self.attr.sub_num, self.attr.dsgn = len(self.agts), dsgn
        self.attr.dsgns = [int(number) for number in dsgn.split('_')]
        self.attr.i = self.attr.dsgns[-1]
        self.attr.args = agts[0].attr.args
        if self.attr.args.timer:
            self.getaction_time = 0
    def __del__(self):
        if self.attr.args.timer:
            print('Combine getaction_time:',round(self.getaction_time/60,2),' minutes')
            print(self.attr.dsgns)
    def memoexps(self, new_obs, rew, done, info, **kwargs):
        if self.attr.dsgn=='0':
            new_obs = np.array([new_obs for i in range(3)]).swapaxes(0,1)
            #new_obs = np.transpose(new_obs,(1,2,0,3,4,5))
            rew = np.array([rew for i in range(3)]).swapaxes(0,1)
            #rew = np.transpose(rew,(1,2,0,3,4,5))

        for i,agt in enumerate(self.agts):
            agt.memoexps(new_obs, rew, done, info, **kwargs)
        return
    def getaction(self, obs, explore, **kwargs):
        getaction_start = time.process_time()
        if self.attr.dsgn=='0':
            obs = np.array([obs for i in range(3)]).swapaxes(0,1)

        act, act_info = [], []
        for i,agt in enumerate(self.agts):
            acti, act_infoi = agt.getaction(obs, explore, **kwargs)
            act.append(acti)
            act_info.append(act_infoi)
        act = np.array(act).swapaxes(0,1)
        self.getaction_time += time.process_time()-getaction_start
        return act, act_info
    def update(self, crt_step, max_step, info_in={}, **kwargs):
        #print('Combine update', self.attr.dsgn)
        for i,agt in enumerate(self.agts):
            agt.update(crt_step, max_step, info_in, **kwargs)
        return
    def save(self, name, **kwargs):
        for i,agt in enumerate(self.agts):
            agt.save(name, **kwargs)
        return
    def load(self, **kwargs):
        for i,agt in enumerate(self.agts):
            agt.load(**kwargs)
        return
class Combine_Teams(Combine):
    def _actWrapper(self,act):
        pass
class Combine_Squads(Combine):
    def getaction(self, obs, explore, **kwargs):
        act, act_info = Combine.getaction(self,obs,explore,**kwargs)
        return act[:,0,0,0], {}
