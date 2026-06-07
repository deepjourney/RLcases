import numpy as np
from algos import getAlgo
import agents
from agents.wrappers import Memo, Stack, Imagine
class MyAgent(agents.Agent):
    def __init__(self,env,obs,args):
        agents.Agent.__init__(self,env,obs,args)
        obs_space = env.observation_space
        act_space = env.action_space
        print(args.env_name,obs_space,act_space)
        #self.algo = getAlgo(obs_space,act_space,args)

        self.a_actions = np.array([0], dtype=np.int32)

    def memoexps(self, new_obs, rew, done, info):
        self.a_actions = info[0]['a_actions']
        pass
        #self.algo.memoexps(new_obs, rew, done, info)
    def getaction(self, obs, explore):
        #act, act_info = self.algo.get_action(obs,explore)
        actions = []
        for obsi in obs:
            function_id = np.random.choice(self.a_actions)
            actions.append(function_id)

        return [actions], {}
        return act, act_info
    def update(self, crt_step, max_step, info_in):
        info_in =  {'mb_obs':       np.array(info_in['mb_obs']),
                    'mb_act':       np.array(info_in['mb_act']),
                    'mb_new_obs':   np.array(info_in['mb_new_obs']),
                    'mb_rew':       np.array(info_in['mb_rew']),
                    'mb_done':      np.array(info_in['mb_done']), **info_in}
        #self.algo.update(crt_step=crt_step, max_step=max_step, info_in=info_in)
    def save(self,name):
        pass
        #self.algo.save(name)
    def load(self):
        pass
        #self.algo.load()

def fAgent(env,obs,args):
    #agt = [agt_maker(args.agt_name, i, args.agt_seed, args) for i in range(args.agt_num)]
    agt = MyAgent(env,obs,args)
    if args.memoplace == "agtcpu": agt = Memo(agt)
    agt = Stack(agt)
    agt = Imagine(agt)
    return agt
