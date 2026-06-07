import numpy as np
import gym, myenv, roboschool, easydict, cv2, random, scipy, json
class Monitor(gym.Wrapper):
    def __init__(self, env, i, args): #whole game information here
        gym.Wrapper.__init__(self, env=env)
        self.args = args
        self.observation_space, self.action_space = self.env.observation_space, self.env.action_space
        self.g_step, self.reward, self.length, self.last_epreward, self.last_eplength, self.rewlist, self.actlist = 0, 0, 0, -1, -1, [], []
        self.frewards, self.flengths = open(args.rewardsname+str(i),'a'), open(args.lengthsname+str(i),'a')
        self.frewlist, self.factlist = open(args.rewardsname+'rewlist_'+str(i),'a'), open(args.rewardsname+'actlist_'+str(i),'a')
        if args.to_test:
            videoname    = args.output_dir+str(args.env_seed)+'_'+str(i)+'.mp4'
            fps, fourcc  = 30, cv2.VideoWriter_fourcc(*'mp4v')#'M','J','P','G')
            if args.atariflag: args.times = 1###
            width, height= self.env.observation_space.shape[0]*args.times, self.env.observation_space.shape[1]*args.times
            if args.atariflag:  self.encoder = ImageEncoder(videoname, (width, height, 3), fps)
            else:               self.vWriter = cv2.VideoWriter(videoname, fourcc, fps, (width, height))
        if self.args.masksize!=0:#reconnaissance recon
            self.mask, self.pos = self.getmask(size=self.args.masksize)
            self.num_of_move, self.unit = 5, 10
            self.action_space = gym.spaces.Discrete(self.num_of_move*self.action_space.n)
    def getmask(self,size):
        mask = np.ones(self.observation_space.shape,dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if i>=mask.shape[0]-size and j>=mask.shape[1]-size: mask[i][j]=np.zeros(3,dtype=np.uint8)
        return mask, [mask.shape[0]-size,mask.shape[1]-size]
    def movemask(self,move):
        #print('m',move)
        #print('p',self.pos)
        if move==0: 
            #print(-min(self.unit,self.pos[1]))
            self.mask = np.roll(self.mask, shift=-min(self.unit,self.pos[1]), axis=1)
            self.pos[1] = max(self.pos[1]-self.unit,0)
        if move==1: 
            #print(-min(self.unit,self.pos[0]))
            self.mask = np.roll(self.mask, shift=-min(self.unit,self.pos[0]), axis=0)
            self.pos[0] = max(self.pos[0]-self.unit,0)
        if move==2: 
            pass
        if move==3: 
            #print(min(self.unit,self.mask.shape[0]-self.args.masksize-self.pos[0]))
            self.mask = np.roll(self.mask, shift= min(self.unit,self.mask.shape[0]-self.args.masksize-self.pos[0]), axis=0)
            self.pos[0] = min(self.pos[0]+self.unit, self.mask.shape[0]-self.args.masksize)
        if move==4: 
            #print(min(self.unit,self.mask.shape[1]-self.args.masksize-self.pos[1]))
            self.mask = np.roll(self.mask, shift= min(self.unit,self.mask.shape[1]-self.args.masksize-self.pos[1]), axis=1)
            self.pos[1] = min(self.pos[1]+self.unit, self.mask.shape[1]-self.args.masksize)
        #print('p',self.pos)
        #if move>5: exit()
    def __del__(self):
        for file in [self.frewards,self.flengths,self.frewlist,self.factlist]:
            print('',file=file,flush=True)
            file.close()
        if self.args.to_test and not self.args.atariflag: self.vWriter.release()
    def reset(self):
        self.reward, self.length, self.last_epreward, self.last_eplength, self.rewlist, self.actlist = 0, 0, -1, -1, [], []
        obs = self.env.reset()
        #self.observation = obs
        if self.args.masksize!=0:
            self.mask, self.pos = self.getmask(size=self.args.masksize)
        return obs
    def step(self, act):
        if self.args.masksize!=0:
            move = act%self.num_of_move
            act = act//self.num_of_move
        obs, rew, done, info = self.env.step(act)
        self.reward+=rew
        self.length+=1
        #self.rewlist.append(rew)
        #self.actlist.append(act)
        #self.observation = obs
        self.g_step+=self.args.env_num
        if done:
            print(int(self.g_step),',',int(self.reward),end='|',file=self.frewards,flush=True)
            print(int(self.g_step),',',int(self.length),end='|',file=self.flengths,flush=True)
            #self.last_epreward = int(self.reward)
            #self.last_eplength = int(self.length)
            #info = {'last_epreward':self.last_epreward,'last_eplength':self.last_eplength,**info}
            #obs = self.reset()
        if self.args.masksize!=0:
            #print('g',self.g_step)
            #self.movemask(self.g_step-1)
            self.movemask(move)
            obs = np.where(self.mask==0, obs, 0)
            #frame = np.add(frame,self.mask*100)
            #frame = self.env.render(mode='rgb_array')
            #frame = np.where(self.mask==0, frame, (0.5*frame+0.5*255*self.mask).astype(np.uint8))
            #scipy.misc.imsave(str(self.g_step)+'.jpg', frame)
        return obs, rew, done, info#self.observation
    def monitor(self,mode):
        frame = self.env.render(mode)#mode='rgb_array'
        if self.args.masksize!=0:
            frame = np.where(self.mask==0, frame, (0.5*frame+0.5*255*self.mask).astype(np.uint8))
        if self.args.atariflag: self.encoder.capture_frame(frame)
        else:                   self.vWriter.write(frame)
        return frame
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
def env_maker(env_name, i, env_seed, args):
    def __make_env():
        if args.atariflag: env = make_atari(env_name)
        else:              env = gym.make(env_name)
        env.seed(i+env_seed)
        random.seed(i+env_seed)
        np.random.seed(i+env_seed)
        env = Monitor(env, i, args)
        if args.atariflag:
            env = wrap_deepmind(env)
            #env = TransposeImage(env, op=[2, 0, 1])
            env.render = env.env.env.env.env.monitor
        return env
    return __make_env
from envirs.warppers import VecNormalize, ImageEncoder
def fEnv(args):
    env_args = easydict.EasyDict()
    env_args.envparas = args.envparas
    with open('./myenv/envinfo.json', 'w') as fenvinfo:
        print(json.dumps(env_args),file=fenvinfo)
    env = [env_maker(args.env_name, i, args.env_seed, args) for i in range(args.env_num)]
    env = SubprocVecEnv(env)
    #env = DummyVecEnv(env)
    if len(env.observation_space.shape) == 1:
        env = VecNormalize(env, gamma=0.99)
    return env, env.reset()
    """envparas = args.envparas.split('_')
    env_args = easydict.EasyDict()
    env_args.generatefreq   = int(envparas[0])
    env_args.endcondition   = [int(cond) for cond in envparas[1].split(',')]
    env_args.bombdamage     = int(envparas[2])
    env_args.penalize       = float(envparas[3])
    with open('./myenv/envinfo.json', 'w') as fenvinfo:
        print(json.dumps(env_args),file=fenvinfo)
    args.times = 10"""
