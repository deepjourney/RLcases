import numpy as np
from algos import getAlgo
import agents
from agents.wrappers import Memo, Stack, Imagine
import gym, gym.spaces
import easydict,random,scipy
from kaggle_environments.envs.halite.helpers import *
class TeamAgent(agents.Agent):
    def __init__(self,env,envinfo,args,dsgn,learnflag):
        agents.Agent.__init__(self,args)
        self.attr.stack_shape = [args.unit_num]+list(env.observation_space.shape[-3:])
        self.attr.stack_dtype = np.uint8
        self.attr.obs_space = gym.spaces.Box(low=0,high=255,shape=env.observation_space.shape[-3:],dtype=np.uint8)#env.observation_space
        self.attr.act_space = gym.spaces.Discrete(5)
        self.attr.dsgn, self.attr.dsgns = dsgn, [int(number) for number in dsgn.split('_')]
        self.attr.learnflag = learnflag
        self.algo = None
        if self.attr.learnflag>= 0: self.algo = getAlgo(self.attr.obs_space,self.attr.act_space,self.attr.args)
        if self.attr.learnflag>  0: self.load()
        print(self.attr.dsgn,self.attr.learnflag)
        print(self.attr.args.env_name,self.attr.obs_space,self.attr.act_space)
        self.step = 0
        self.assignon = bool(int(args.agtonoff))
    def save(self,name):
        self.algo.save(self.attr.dsgn+'_'+name)
    def load(self):
        self.algo.load(prefix=self.attr.dsgn+'_')
    def memoexps(self, new_obs, rew, done, info):
        pass#self.algo.memoexps(new_obs, rew, done, info)
    def getaction(self, obs, explore): #return np.zeros([self.attr.args.env_num,self.attr.args.unit_num],dtype=np.int64), {}
        #with np.printoptions(threshold=np.inf):
        #    print('TeamAgent',obs.shape)
        #    print('TeamAgent',np.transpose(obs,(0,1,2,5,3,4)))
        act, act_info = self.algo.get_action(obs,explore) # env_num, stack_num, unit_num, width, height, channel
        #with np.printoptions(threshold=np.inf):
        #    print('TeamAgent',act.shape)
        #    print('TeamAgent',act)
        #self.step+=1
        #if self.step==5: exit()
        if not self.assignon: return act, act_info # env_num, unit_num
        maps  = obs[:,0,:,:,:,-1]
        #print(maps.shape)
        probs = act_info['probs'].reshape(self.attr.args.env_num,self.attr.args.unit_num,-1)# env_num, unit_num*act_space.n
        #print(probs.shape)
        env_num = self.attr.args.env_num
        unit_num= self.attr.args.unit_num
        def xy(n):
            return n % 21, n // 21
        def c(n):
            return n % 21
        # 0 None 1 WEST 2 SOUTH 3 EAST 4 NORTH
        act = np.zeros([env_num,unit_num],dtype=np.uint8) ### uint8? int is enough?
        for i in range(env_num):
            C = -100*np.ones((unit_num,21*21+unit_num))
            for j in range(unit_num):
                x,y = np.nonzero(maps[i][j])
                if len(x)!=len(y) or len(x)>1 or len(y)>1:
                    print('marker layer not correct!')
                if len(x)==0: continue
                ship_pred_actions = probs[i][j]
                act_num = 5

                raw_ship_pred_actions = np.copy(ship_pred_actions)

                """PRED_PWR = 2.7
                restore_sum = np.sum(ship_pred_actions)
                if restore_sum > 0:
                    ship_pred_actions = ship_pred_actions ** PRED_PWR
                    ship_pred_actions *= restore_sum / sum(ship_pred_actions)"""

                ship_ranked_actions = np.zeros((act_num,), dtype = np.float32)
                for rank in range(0, np.sum(ship_pred_actions > 1e-6) ):
                    while True:
                        action = int(random.choice(np.flatnonzero(ship_pred_actions)))
                        if random.random() < ship_pred_actions[action]:
                            ship_ranked_actions[action] = act_num - rank + raw_ship_pred_actions[action];
                            ship_pred_actions[action] = 0
                            ship_pred_actions = ship_pred_actions / np.sum(ship_pred_actions)
                            break;

                C[j, x        + 21*y] = ship_ranked_actions[0]
                C[j, x + 21*c(y - 1)] = ship_ranked_actions[4]
                C[j, c(x + 1) + 21*y] = ship_ranked_actions[3]
                C[j, x + 21*c(y + 1)] = ship_ranked_actions[2]
                C[j, c(x - 1) + 21*y] = ship_ranked_actions[1]
                #if 1:#my_halite >= conf.convertCost or ship_info[CARGO] > conf.convertCost:
                #    C[j, 21*21 + j]= ship_ranked_actions[5] # conversion doesn't use any squares
            entity_idxs, assignments = scipy.optimize.linear_sum_assignment(C, maximize=True)
            #print([ ( xy(assignment) if assignment < 21 * 21 else assignment - 21 * 21 ) for assignment in assignments])
            assigned = dict(zip(entity_idxs, assignments))
            for j in range(unit_num):
                x,y = np.nonzero(maps[i][j])
                if len(x)!=len(y) or len(x)>1 or len(y)>1:
                    print('marker layer not correct!')
                if len(x)==0: continue
                
                xt, yt = xy(assigned[j])
                if x == xt and y == yt:
                    pass # no move
                else:
                    if   c(xt-x) == 1: act[i][j]=3#a = 'EAST'
                    elif c(yt-y) == 1: act[i][j]=2#a = 'SOUTH'
                    elif c(x-xt) == 1: act[i][j]=1#a = 'WEST'
                    elif c(y-yt) == 1: act[i][j]=4#a = 'NORTH'
                    else:print('says to move but where???')

        return act, act_info # env_num, unit_num
    def update(self, crt_step, max_step, info_in): #return
        info_in['mb_obs']     = np.array(info_in['mb_obs'])
        info_in['mb_act']     = np.array(info_in['mb_act'])
        info_in['mb_new_obs'] = np.array(info_in['mb_new_obs'])
        info_in['mb_rew']     = np.array(info_in['mb_rew'])
        info_in['mb_done']    = np.array(info_in['mb_done'])
        self.algo.update(crt_step=crt_step, max_step=max_step, info_in=info_in)
class Marker(agents.Wrapper):
    def __init__(self,agt):
        agents.Wrapper.__init__(self,agt)
        if self.attr.args.timer:
            self.Marker_getaction, self.Marker_memoexps = 0,0
    def __del__(self):
        if self.attr.args.timer:
            print('Marker_getaction:',np.round(self.Marker_getaction/60,2),' minutes')
            print('Marker_memoexps:', np.round(self.Marker_memoexps/60,2),' minutes')
    def save(self,name):
        if self.attr.learnflag== 0: self.agt.save(name)
    def load(self):
        if self.attr.learnflag>= 0: self.agt.load()
    def memoexps(self, new_obs, rew, done, info):
        if self.attr.args.timer: self.Marker_memoexps_start = self.process_time()
        if self.attr.learnflag== 0: self.agt.memoexps(self.obswrapper(new_obs), rew, done, info)
        if self.attr.args.timer: self.Marker_memoexps += self.process_time()-self.Marker_memoexps_start
    def getaction(self, obs, explore):
        if self.attr.learnflag>= 0:
            if self.attr.args.timer: self.Marker_getaction_start = self.process_time()
            act, act_info = self.agt.getaction(self.obswrapper(obs),explore)
            if self.attr.args.timer: self.Marker_getaction += self.process_time()-self.Marker_getaction_start
        else:
            act, act_info = np.zeros([self.attr.args.env_num,self.attr.args.unit_num],dtype=np.int64), {}
        return act, act_info
    def obswrapper(self, obs): #[:,0,0,:]
        return np.expand_dims(obs,axis=1)
        units_obs = np.zeros([self.attr.args.env_num,self.attr.args.unit_num]+list(self.attr.obs_space.shape))
        for i in range(self.attr.args.unit_num):
            obs[:,:,:,[-1,-1-i]] = obs[:,:,:,[-1-i,-1]]
            units_obs[:,i,:] = obs[:]
        units_obs = np.expand_dims(units_obs,axis=1) #print('Marker',units_obs.shape) #exit()
        return units_obs
    def update(self, crt_step, max_step, info_in):
        if self.attr.learnflag== 0: self.agt.update(crt_step=crt_step, max_step=max_step, info_in=info_in)
def fAgent(env,envinfo,args):
    glearnflag = []
    for learnflags in args.learnflag.split('_'):
        glearnflag.append([int(learnflag) for learnflag in learnflags.split(',')])
    master_dsgn = '0'
    squads = []
    for isquad in range(args.play_num):
        squad_dsgn = master_dsgn+'_'+str(isquad)
        teams = []
        for iteam in range(args.type_num):
            team_dsgn = squad_dsgn+'_'+str(iteam)
            team = TeamAgent(env,envinfo,args,team_dsgn,glearnflag[isquad][iteam])
            if args.memoplace == "agtcpu": team = Memo(team)
            team = Marker(team)
            teams.append(team)
        squad = Combine_Teams(teams,squad_dsgn,glearnflag[isquad])
        squads.append(squad)
    master = Combine_Squads(squads,master_dsgn,glearnflag)
    return master
class Combine(object):
    def __init__(self, agts, dsgn, learnflag):
        self.agts, self.attr = agts, easydict.EasyDict()
        self.attr.sub_num, self.attr.dsgn = len(self.agts), dsgn
        self.attr.dsgns = [int(number) for number in dsgn.split('_')]
        self.attr.i = self.attr.dsgns[-1]
        self.attr.learnflag = learnflag
    def memoexps(self, new_obs, rew, done, info, **kwargs):
        for i,agt in enumerate(self.agts):
            #agt.memoexps(new_obs, rew[:,i], done, info, **kwargs)
            agt.memoexps(new_obs[:,i], rew[:,i], done, info, **kwargs)
        return
    def getaction(self, obs, explore, **kwargs):
        act, act_info = [], []
        for i,agt in enumerate(self.agts):
            #acti, act_infoi = agt.getaction(obs, explore, **kwargs)
            acti, act_infoi = agt.getaction(obs[:,i], explore, **kwargs)
            act.append(acti)
            act_info.append(act_infoi)
        act = np.array(act).swapaxes(0,1)
        return act, act_info
    def update(self, crt_step, max_step, info_in={}, **kwargs):
        for i,agt in enumerate(self.agts):
            agt.update(crt_step, max_step, info_in, **kwargs)
    def save(self, name, **kwargs):
        for i,agt in enumerate(self.agts):
            agt.save(name, **kwargs)
    def load(self, **kwargs):
        for i,agt in enumerate(self.agts):
            agt.load(**kwargs)
class Combine_Teams(Combine):
    def _actWrapper(self,act):
        pass
class Combine_Squads(Combine):
    def _actWrapper(self,act):
        pass
