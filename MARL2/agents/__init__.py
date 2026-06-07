def add_arguments(parser):
    parser.add_argument('--agent', default='imagine', help='agent to use: imagine | reconimg (imagine)')#agent
    parser.add_argument('--agtonoff', default='', help='agent switches')
    parser.add_argument('--play-num', type=int, default=1, help='number of players(default: 1)')
    parser.add_argument('--type-num', type=int, default=1, help='number of types  (default: 1)')
    parser.add_argument('--unit-num', type=int, default=1, help='number of units  (default: 1)')
    parser.add_argument('--learnflag', default='', help='learn flag switches')
    parser.add_argument('--stack-num', type=int, default=1, help='number of observation stacks (default: 1)')
    parser.add_argument('--memo-size', type=int, default=5, help='size of memory (default: 5)')
    parser.add_argument('--imgparas', default='none,none,0.0', help='imagine parameters (ref: full,prev,0.0)')
    parser.add_argument('--action-model', default='', help='folder of action model')
    parser.add_argument('--memoplace', default='agtcpu', help='memoplace: agtcpu | algocpu | algogpu (agent)')
def add_strings(args):
    args.exp_dir=args.exp_dir+':'+args.agent+'_'+str(args.stack_num)+'_'+str(args.play_num)+'_'+str(args.type_num)+'_'+str(args.unit_num)+'_'+str(args.agtonoff)#+'_'+str(args.memo_size)#+'_'+args.imgparas+'_'+args.memoplace
def getAgent(env,envinfo,args):
    if args.agent=='imagine':
        from agents.agent_imagine import fAgent
    if args.agent=='reconimg':
        from agents.agent_reconimg import fAgent
    if args.agent=='imagineM':
        from agents.agent_imagineM import fAgent
    if args.agent=='csc2':
        from agents.agent_csc2 import fAgent
    if args.agent=='haliteM':
        from agents.agent_haliteM import fAgent
    return fAgent(env,envinfo,args)

import numpy as np
import easydict,time
#from pysc2.agents import base_agent
#from pysc2.lib import actions
#from pysc2.lib import features
class Agent(object):#base_agent.BaseAgent):
    metadata = {'render.modes': []}
    spec = None
    def __init__(self,args):#env,envinfo,args,dsgn):
        self.attr = easydict.EasyDict()
        self.attr.args = args
        #self.attr.env, self.attr.initobs = env, envinfo['obs']
        #self.attr.args, self.attr.dsgn = args, dsgn
        #self.attr.dsgns = [int(number) for number in dsgn.split('_')]
        #self.attr.i = self.attr.dsgns[-1]
    @property
    def unwrapped(self):
        return self
    def __str__(self):
        if self.spec is None: return '<{} instance>'.format(type(self).__name__)
        else:                 return '<{}<{}>>'.format(type(self).__name__, self.spec.id)
    def memoexps(self, new_obs, rew, done, info):
        raise NotImplementedError
    def getaction(self, obs, explore):
        raise NotImplementedError
    def update(self, crt_step, max_step, info_in):
        raise NotImplementedError
    def save(self,name):
        raise NotImplementedError
    def load(self):
        raise NotImplementedError
    #def step(self, obs):
    #    super(RandomAgent, self).step(obs)
    #    function_id = numpy.random.choice(obs.observation.available_actions)
    #    args = [[numpy.random.randint(0, size) for size in arg.sizes]
    #            for arg in self.action_spec.functions[function_id].args]
    #    return actions.FunctionCall(function_id, args)

class Wrapper(Agent):
    agt = None
    def __init__(self, agt):
        self.agt = agt
        self.metadata = self.agt.metadata
        self.attr = self.agt.attr
    @property
    def spec(self):
        return self.agt.spec
    @property
    def unwrapped(self):
        return self.agt.unwrapped
    @classmethod
    def class_name(cls):
        return cls.__name__
    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.agt)
    def __repr__(self):
        return str(self)
    def memoexps(self, new_obs, rew, done, info, **kwargs):
        return self.agt.memoexps(new_obs, rew, done, info, **kwargs)
    def getaction(self, obs, explore, **kwargs):
        return self.agt.getaction(obs, explore, **kwargs)
    def update(self, crt_step, max_step, info_in={}, **kwargs):
        return self.agt.update(crt_step, max_step, info_in, **kwargs)
    def save(self, name, **kwargs):
        return self.agt.save(name, **kwargs)
    def load(self, **kwargs):
        return self.agt.load(**kwargs)
    def process_time(self):
        return np.array([time.process_time(),time.perf_counter()])
