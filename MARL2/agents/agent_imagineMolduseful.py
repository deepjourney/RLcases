import numpy as np
from algos import getAlgo
import agents,copy,random
from agents.wrappers import Memo, Stack, Imagine
from collections import OrderedDict
from pprint import pprint
class MyAgent(agents.Agent):
    def __init__(self,env,obs,args):
        agents.Agent.__init__(self,env,obs,args)
        obs_space = env.observation_space
        act_space = env.action_space
        print(args.env_name,obs_space,act_space)
        print('Initobs shape:',obs.shape)
        obs_space = env.spec._kwargs['attr']['obs_spaces']
        act_space = env.spec._kwargs['attr']['act_spaces']
        print(args.env_name,obs_space,act_space)
        self.unit_nums = env.spec._kwargs['attr']['unit_num_alls']
        self.num_species = len(self.unit_nums)
        self.learnflag = [  int(cond) for cond in args.learnflag.split(',')]
        # need to expand to same size with num species
        print(self.learnflag)
        subargs = []
        for i in range(self.num_species):
            subargs.append(copy.deepcopy(args))
        #subargs[0].lr_M = 1000
        #subargs[0].apfparas = '3,3,1,1,1,1,0,0,0,256,1^3,3,1,1,1,1,0,0,0,256,1=256'
        for i in range(self.num_species):
            print(subargs[i])
        #try:
        if 1:
            lines = open(args.exp_dir+'cscores','r').read().splitlines()
            print(len(lines))
            records = OrderedDict()
            for i,line in enumerate(lines):
                data = line.split(',')
                #print(data)
                thread = data[0]
                key = data[1]+','+data[2]
                value = np.array([int(data[3]),int(data[4]),int(data[5]),int(data[6])])
                if key in records: records[key] += value
                else:              records[key] = value
                #if i%100==0:
                #    for key,value in records.items():
                #        records[key] = value*0.5
            #print('records')
            #pprint(dict(records.items()))
            fullrecords = OrderedDict()
            for key,value in records.items():
                value0 = round(value[0]/value[3],2)
                value1 = round(value[1]/value[3],2)
                value2 = round(value[2]/value[3],2)
                value = [(value),np.array([value0,value1,value2])]
                fullrecords[key] = value
            #print('fullrecords')
            #pprint(dict(fullrecords.items()))
            sorted_fullrecords = sorted(fullrecords.items(), key=lambda x: x[1][0][0]/x[1][0][3], reverse=True)
            #print('sorted_fullrecords')
            #for i in range(len(sorted_fullrecords)):
            #    print(sorted_fullrecords[i])
            sorted_fullrecords = sorted(fullrecords.items(), key=lambda x: x[1][1][0], reverse=True)
            print('sorted_fullrecords2')
            for i in range(len(sorted_fullrecords)):
                print(sorted_fullrecords[i])
            """
            agtrecords = {}
            npcrecords = {}
            for key,value in records.items():
                agtkey = key.split(',')[0]
                npckey = key.split(',')[1]
                if agtkey in agtrecords: agtrecords[agtkey]+=value.copy()
                else:                    agtrecords[agtkey]=value.copy()
                if npckey in npcrecords: npcrecords[npckey]+=value.copy()
                else:                    npcrecords[npckey]=value.copy()
                #pprint(dict(agtrecords.items()))
                #pprint(dict(npcrecords.items()))
            print('agt')
            fullagtrecords = OrderedDict()
            for key,value in agtrecords.items():
                value0 = round(value[0]/value[3],2)
                value1 = round(value[1]/value[3],2)
                value2 = round(value[2]/value[3],2)
                value = [(value),np.array([value0,value1,value2])]
                fullagtrecords[key] = value
            sorted_fullagtrecords = sorted(fullagtrecords.items(), key=lambda x: x[1][1][0], reverse=True)
            for i in range(len(sorted_fullagtrecords)):
                print(sorted_fullagtrecords[i])
            print('npc')
            fullnpcrecords = OrderedDict()
            for key,value in npcrecords.items():
                value0 = round(value[0]/value[3],2)
                value1 = round(value[1]/value[3],2)
                value2 = round(value[2]/value[3],2)
                value = [(value),np.array([value0,value1,value2])]
                fullnpcrecords[key] = value
            sorted_fullnpcrecords = sorted(fullnpcrecords.items(), key=lambda x: x[1][1][2], reverse=True)
            for i in range(len(sorted_fullnpcrecords)):
                print(sorted_fullnpcrecords[i])
            """
            agtrecordsfair = {}
            npcrecordsfair = {}
            for key,value in fullrecords.items():
                agtkey = key.split(',')[0]
                npckey = key.split(',')[1]
                if agtkey in agtrecordsfair: agtrecordsfair[agtkey].append(value.copy()[1])
                else:                        agtrecordsfair[agtkey]=[value.copy()[1]]
                if npckey in npcrecordsfair: npcrecordsfair[npckey].append(value.copy()[1])
                else:                        npcrecordsfair[npckey]=[value.copy()[1]]
            print('agtfair')
            #pprint(dict(agtrecordsfair.items()))
            fullagtrecordsfair = OrderedDict()
            for key,value in agtrecordsfair.items():
                newvalue = [np.zeros(1),np.round(np.mean(value,axis=0),2)]
                fullagtrecordsfair[key] = newvalue
            sorted_fullagtrecordsfair = sorted(fullagtrecordsfair.items(), key=lambda x: x[1][1][0], reverse=True)
            for i in range(len(sorted_fullagtrecordsfair)):
                print(sorted_fullagtrecordsfair[i])
            print('npcfair')
            #pprint(dict(npcrecordsfair.items()))
            fullnpcrecordsfair = OrderedDict()
            for key,value in npcrecordsfair.items():
                newvalue = [np.zeros(1),np.round(np.mean(value,axis=0),2)]
                fullnpcrecordsfair[key] = newvalue
            sorted_fullnpcrecordsfair = sorted(fullnpcrecordsfair.items(), key=lambda x: x[1][1][2], reverse=True)
            for i in range(len(sorted_fullnpcrecordsfair)):
                print(sorted_fullnpcrecordsfair[i])
            
            sorted_fullagtrecords = sorted_fullagtrecordsfair
            sorted_fullnpcrecords = sorted_fullnpcrecordsfair

            potential_nums,potential_weights=[],[]
            for i in range(self.num_species):
                if i==0:
                    potential_nums.append([int(record[0]) for record in sorted_fullagtrecords])
                    potential_weights.append([(record[1][1][0]) for record in sorted_fullagtrecords])
                if i==1:
                    potential_nums.append([int(record[0]) for record in sorted_fullnpcrecords])
                    potential_weights.append([(record[1][1][2]) for record in sorted_fullnpcrecords])
            print('potential_nums')
            print(potential_nums)
            print('potential_weights')
            print(potential_weights)
        #except:
        #    print('No past cscores or past cscores read error!')

        self.model_num_pools,self.model_weight_pools = [],[]
        for i in range(self.num_species):
            num_pool,weight_pool=[],[]
            if self.learnflag[i]>0:
                #num_pool    = potential_nums[i][:self.learnflag[i]]#[j for j in range(self.learnflag[i])]
                #weight_pool = potential_weights[i][:self.learnflag[i]]#[1 for j in range(self.learnflag[i])]
                try:
                    num_pool    = potential_nums[i][:self.learnflag[i]]#[j for j in range(self.learnflag[i])]
                    weight_pool = potential_weights[i][:self.learnflag[i]]#[1 for j in range(self.learnflag[i])]
                    num_pool    = num_pool    + [self.learnflag[i]-1-j for j in range(self.learnflag[i]-len(num_pool))]
                    weight = np.mean(weight_pool)
                    if np.isnan(weight):
                        print('nan weight',weight)
                        weight=1
                    weight_pool = weight_pool + [weight for j in range(self.learnflag[i]-len(weight_pool))]
                    print('Use pool from cscores')
                except:
                    num_pool    = [j for j in range(self.learnflag[i])]
                    weight_pool = [1 for j in range(self.learnflag[i])]
                    print('Use initial pool')
            self.model_num_pools.append(num_pool)
            self.model_weight_pools.append(weight_pool)
        print('self.model_num_pools')
        print(self.model_num_pools)
        print('self.model_weight_pools')
        print(self.model_weight_pools)

        self.loaded_model_num = -self.attr.args.env_seed #
        self.attr.model_num_list,self.pickedj_list = [],[]
        for i in range(self.num_species):
            model_num,pickedj = self.loaded_model_num,None
            if self.learnflag[i]>0:
                pickedj = random.choices(range(len(self.model_num_pools[i])),weights=self.model_weight_pools[i])[0]
                model_num = self.model_num_pools[i][pickedj]
            self.pickedj_list.append(pickedj)
            self.attr.model_num_list.append(model_num)

        self.algos = []
        for i in range(self.num_species):
            if self.learnflag[i]>0:
                algopool = []
                for model_num in self.model_num_pools[i]:
                    algo = getAlgo(obs_space[i],act_space[i],subargs[i])
                    self.model_filename = algo.load(prefix='algo'+str(i)+'_'+str(model_num)+'_')
                    algopool.append(algo)
                self.algos.append(algopool)
            else:
                algo = getAlgo(obs_space[i],act_space[i],subargs[i])
                self.algos.append(algo)

        self.fscores = open(args.exp_dir+'cscores','a')
    def memoexps(self, new_obs, rew, done, info):
        # need to upgrade, if want to use memoplace=algogpu or algocpu
        #for i in range(self.num_species):
        #    self.algos[i].memoexps(new_obs, rew, done, info)
        #if done.any(): # if one env done, change the picking of algopool no matter other envs are on the way...
        for idonei,donei in enumerate(done):
            if donei:
                if self.attr.args.to_test:
                    print(idonei,end=',',file=self.fscores)
                    for model_num in self.attr.model_num_list:
                        print(model_num,end=',',file=self.fscores)
                    print(info[idonei]['gain'],end=',',file=self.fscores)
                    print(info[idonei]['miss'],end=',',file=self.fscores)
                    print(info[idonei]['loss'],end=',',file=self.fscores)
                    print(info[idonei]['epsd'],file=self.fscores,flush=True)
                self.attr.model_num_list.clear()
                self.pickedj_list.clear()
                for i in range(self.num_species):
                    model_num,pickedj = self.loaded_model_num,None
                    if self.learnflag[i]>0:
                        pickedj = random.choices(range(len(self.model_num_pools[i])),weights=self.model_weight_pools[i])[0]
                        model_num = self.model_num_pools[i][pickedj]
                    self.pickedj_list.append(pickedj)
                    self.attr.model_num_list.append(model_num)
    def getaction(self, obs, explore):
        obss,start,end=[],0,0
        for i in range(self.num_species):
            end += self.unit_nums[i]
            obss.append(obs[:,:,start:end,:,:,:])
            start=end
        acts,act_infos=[],[]
        for i in range(self.num_species):
            if self.learnflag[i]>0:
                acti,act_infoi=self.algos[i][self.pickedj_list[i]].get_action(obss[i],explore)
            else:
                acti,act_infoi=self.algos[i].get_action(obss[i],explore)
            acts.append(acti)
            act_infos.append(act_infoi)
        act = np.concatenate(acts,axis=-1)
        act_info = {}
        return act, act_info
    def update(self, crt_step, max_step, info_in):
        info_ins,start,end=[],0,0
        for i in range(self.num_species):
            end += self.unit_nums[i]
            info_ini = {'mb_obs':       info_in['mb_obs'][:,:,:,start:end,:,:,:],
                        'mb_act':       info_in['mb_act'][:,:,start:end],
                        'mb_new_obs':   info_in['mb_new_obs'][:,:,:,start:end,:,:,:],
                        'mb_rew':       info_in['mb_rew'][:,:,start:end],
                        'mb_done':      info_in['mb_done']}
            start=end
            info_ins.append(info_ini)
        for i in range(self.num_species):
            if self.learnflag[i]!=0: continue
            self.algos[i].update(crt_step=crt_step, max_step=max_step, info_in=info_ins[i])
    def save(self,name):
        for i in range(self.num_species):
            if self.learnflag[i]!=0: continue
            self.algos[i].save(name,prefix='algo'+str(i)+'_')#+str(self.attr.args.env_seed%self.poolsize)+'_')
    def load(self):
        for i in range(self.num_species):
            if self.learnflag[i]!=0: continue
            else: self.model_filename = self.algos[i].load(prefix='algo'+str(i)+'_')
            self.loaded_model_num = int(self.model_filename.split('/')[-1].split('_')[1])
            print('agent load',self.loaded_model_num)
            self.attr.model_num_list[i] = self.loaded_model_num

class ExtraInfoToEnv(agents.Wrapper):
    def __init__(self,agt):
        agents.Wrapper.__init__(self,agt)
        self.turn = 0
    def getaction(self, obs, explore):
        act, act_info = self.agt.getaction(obs,explore)
        if len(self.attr.model_num_list)!=2:
            print('error')
            exit()
        extrainfo = [self.attr.model_num_list]*self.attr.args.env_num
        extrainfo = np.array(extrainfo)
        act = np.concatenate([act,extrainfo],axis=-1)
        if act.dtype!=np.int64:
            print(act.dtype)
            print('error')
            exit()
        return act, act_info

"""def agt_maker(agt_name, i, agt_seed, args):
    def __make_agt():
        agt = agents.make(agt_name)
        agt = Memo(agt, i, args)
        return agt
    return __make_agt"""
def fAgent(env,obs,args):
    #agt = [agt_maker(args.agt_name, i, args.agt_seed, args) for i in range(args.agt_num)]
    agt = MyAgent(env,obs,args)
    if args.memoplace == "agtcpu": agt = Memo(agt)
    agt = Stack(agt)
    agt = Imagine(agt)
    agt = ExtraInfoToEnv(agt)
    return agt
