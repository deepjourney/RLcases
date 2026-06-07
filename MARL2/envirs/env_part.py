import numpy as np
import gym, easydict, cv2, random, scipy, json
from envirs.warppers import Recorder, Monitor, wrap_deepmind_render
class Mask2D(gym.Wrapper):#reconnaissance recon
    def __init__(self, env, i, args):
        gym.Wrapper.__init__(self, env=env)
        pobparas = args.pobparas.split('=')[0].split('^')
        print(pobparas)
        self.mask_size, self.num_move, self.unit_move = [], [], []
        for pobpara in pobparas:
            if pobpara=='': continue
            maskparas = pobpara.split(',')
            self.mask_size.append(int(maskparas[0]))
            self.num_move.append(int(maskparas[1]))
            self.unit_move.append(int(maskparas[2]))
        if len(self.num_move)!=0:
            self.org_action_space = self.env.action_space
            self.action_space = gym.spaces.Discrete(np.prod(self.num_move)*self.env.action_space.n)
            self.spec._kwargs['org_action_space'] = self.org_action_space
            self.spec._kwargs['ext_action_space'] = gym.spaces.Discrete(np.prod(self.num_move))
        self.masks, self.poss = [], []
    def reset(self):
        obs = self.env.reset()
        self.masks.clear()
        self.poss.clear()
        for mask_sizei in self.mask_size:
            mask, pos = self.getmask(size=mask_sizei)
            self.masks.append(mask)
            self.poss.append(pos)
        self.mask_all = np.zeros(self.observation_space.shape,dtype=np.uint8)
        self.mask_step = 0
        return obs
    def getmask(self,size):
        mask = np.ones(self.observation_space.shape,dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if i>=mask.shape[0]-size and j>=mask.shape[1]-size: mask[i][j]=np.zeros(1,dtype=np.uint8)
        return mask, [mask.shape[0]-size,mask.shape[1]-size]
    def movemask(self, mask, pos, masksize, unit_of_move, move):
        if move==0: pass
        if move==1: #print(-min(self.unit_of_move,self.pos[1]))
            mask = np.roll(mask, shift=-min(unit_of_move,pos[1]), axis=1)
            pos[1] = max(pos[1]-unit_of_move,0)
        if move==2: #print(-min(self.unit_of_move,self.pos[0]))
            mask = np.roll(mask, shift=-min(unit_of_move,pos[0]), axis=0)
            pos[0] = max(pos[0]-unit_of_move,0)
        if move==3: #print(min(self.unit_of_move,self.mask.shape[0]-self.args.masksize-self.pos[0]))
            mask = np.roll(mask, shift= min(unit_of_move,mask.shape[0]-masksize-pos[0]), axis=0)
            pos[0] = min(pos[0]+unit_of_move, mask.shape[0]-masksize)
        if move==4: #print(min(self.unit_of_move,self.mask.shape[1]-self.args.masksize-self.pos[1]))
            mask = np.roll(mask, shift= min(unit_of_move,mask.shape[1]-masksize-pos[1]), axis=1)
            pos[1] = min(pos[1]+unit_of_move, mask.shape[1]-masksize)
        return mask, pos
    def step(self, act):
        if len(self.num_move)!=0:
            move = act//self.org_action_space.n
            act  = act%self.org_action_space.n
            moves, npprod = [], np.prod(self.num_move)
            for i,num_movei in enumerate(self.num_move[::-1]):
                npprod=npprod/num_movei
                moves.append(move//npprod)
                move = move%npprod
            moves.reverse()
        obs, rew, done, info = self.env.step(act)
        #self.mask_step += 1#reconnaissance recon for atari wrappers FireResetEnv
        #if self.mask_step <=2:
        #    return obs, rew, done, info
        for i in range(len(self.mask_size)):
            self.masks[i], self.poss[i] = self.movemask(self.masks[i], self.poss[i], self.mask_size[i], self.unit_move[i], moves[i])
        self.mask_all = np.ones(self.observation_space.shape,dtype=np.uint8)
        for mask in self.masks:
            self.mask_all = np.where(mask==0, 0, self.mask_all)
        #info = {'mask': self.mask_all}
        obs = np.where(self.mask_all==0, obs, None)
        return obs, rew, done, info
    def render(self,mode):
        frame = self.env.render(mode)
        fmask = np.tile(self.mask_all,3)
        frame = np.where(fmask==0, frame, (0.5*frame+0.5*255*fmask).astype(np.uint8))
        #scipy.misc.imsave(str(self.g_step)+'.jpg', frame)
        return frame
import copy
class Mask1D(gym.Wrapper):
    def __init__(self, env, i, args):
        gym.Wrapper.__init__(self, env=env)
        pobparas = args.pobparas.split('=')[1].split('^')
        print(pobparas)
        self.mask_size, self.num_move, self.unit_move = [], [], []
        for pobpara in pobparas:
            if pobpara=='': continue
            maskparas = pobpara.split(',')
            self.mask_size.append(int(maskparas[0]))
            self.num_move.append(self.env.observation_space.shape)
            self.unit_move.append(None)
            #self.num_move.append(int(maskparas[1]))
            #self.unit_move.append(int(maskparas[2]))
        if len(self.num_move)!=0:
            if self.env.action_space.__class__.__name__ == "Discrete":
                self.org_action_space = self.env.action_space
                self.action_space = gym.spaces.Discrete(np.prod(self.num_move)*self.env.action_space.n)
                self.spec._kwargs['org_action_space'] = self.org_action_space
                self.spec._kwargs['ext_action_space'] = gym.spaces.Discrete(np.prod(self.num_move))
            if self.env.action_space.__class__.__name__ == "Box":
                self.org_action_space = self.env.action_space
                low   = copy.deepcopy(self.env.action_space.low)
                high  = copy.deepcopy(self.env.action_space.high)
                shape = copy.deepcopy(self.env.action_space.shape)
                dtype = copy.deepcopy(self.env.action_space.dtype)
                low   = np.append(low,-1)
                high  = np.append(high,1)
                self.action_space = gym.spaces.Box(low=low,high=high,shape=None,dtype=dtype)
                self.spec._kwargs['org_action_space'] = self.org_action_space
                self.spec._kwargs['ext_action_space'] = gym.spaces.Discrete(np.prod(self.num_move))
        self.masks, self.poss = [], []
    def reset(self):
        obs = self.env.reset()
        self.masks.clear()
        self.poss.clear()
        for mask_sizei in self.mask_size:
            mask, pos = self.getmask(size=mask_sizei)
            self.masks.append(mask)
            self.poss.append(pos)
        self.mask_all = np.zeros(self.observation_space.shape,dtype=np.uint8)
        self.mask_step = 0
        return obs
    def getmask(self,size):
        mask = np.ones(self.observation_space.shape,dtype=np.uint8)
        mask[-1] = 0
        return mask, 0
    def movemask(self, mask, pos, masksize, unit_of_move, move):
        move = int(move)
        mask = np.ones(self.observation_space.shape,dtype=np.uint8)
        mask[move] = 0
        pos = move
        return mask, pos
    def step(self, act):
        if len(self.num_move)!=0:
            if self.env.action_space.__class__.__name__ == "Discrete":
                move = act//self.org_action_space.n
                act  = act%self.org_action_space.n
                moves, npprod = [], np.prod(self.num_move)
                for i,num_movei in enumerate(self.num_move[::-1]):
                    npprod=npprod/num_movei
                    moves.append(move//npprod)
                    move = move%npprod
                moves.reverse()
            if self.env.action_space.__class__.__name__ == "Box":
                move = int((np.clip(act[-1],-1,0.99)+1.0)/2.0*(np.prod(self.num_move)))
                act  = act[:-1]
                moves, npprod = [], np.prod(self.num_move)
                for i,num_movei in enumerate(self.num_move[::-1]):
                    npprod=npprod/num_movei
                    moves.append(move//npprod)
                    move = move%npprod
                moves.reverse()
        obs, rew, done, info = self.env.step(act)
        #self.mask_step += 1#reconnaissance recon for atari wrappers FireResetEnv
        #if self.mask_step <=2:
        #    return obs, rew, done, info
        for i in range(len(self.mask_size)):
            self.masks[i], self.poss[i] = self.movemask(self.masks[i], self.poss[i], self.mask_size[i], self.unit_move[i], moves[i])
        self.mask_all = np.ones(self.observation_space.shape,dtype=np.uint8)
        for mask in self.masks:
            self.mask_all = np.where(mask==0, 0, self.mask_all)
        #info = {'mask': self.mask_all}
        #obs = np.where(self.mask_all==0, obs, None)
        obs = np.where(self.mask_all==1, obs, None)
        return obs, rew, done, info
    def render(self,mode):
        return self.env.render(mode)

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
def env_maker(env_name, i, env_seed, args):
    def __make_env():
        if args.atariflag: env = make_atari(env_name)
        else:              env = gym.make(env_name)
        env.seed(i+env_seed)
        random.seed(i+env_seed)
        np.random.seed(i+env_seed)
        env = Recorder(env, i, args)
        if args.atariflag:
            if args.render:
                env = Monitor(env, i, args, 'org_')
            env = wrap_deepmind(env)
            env = wrap_deepmind_render(env)
        if len(env.observation_space.shape) == 3: env = Mask2D(env, i, args)
        else:                                     env = Mask1D(env, i, args)
        if args.render:
            env = Monitor(env, i, args)
        return env
    return __make_env
from envirs.warppers import VecNormalize
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
def fEnv(args):
    env_args = easydict.EasyDict()
    env_args.envparas = args.envparas
    with open('./myenv/envinfo.json', 'w') as fenvinfo:
        print(json.dumps(env_args),file=fenvinfo)
    env = [env_maker(args.env_name, i, args.env_seed, args) for i in range(args.env_num)]
    env = SubprocVecEnv(env)
    #env = DummyVecEnv(env)
    #if len(env.observation_space.shape) == 1:
    #    env = VecNormalize(env, gamma=0.99)
    return env, env.reset()
