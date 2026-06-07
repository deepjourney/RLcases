def add_arguments(parser):
    parser.add_argument('--algo', default='PTa2c1', help='algo to use: TFa2c1 | PTa2c1 | PTppo (tensorflow or pytorch)')#algo
    parser.add_argument('--lr-M', default='700', help='learning rate (default: 700e-6)')
    parser.add_argument('--decay', default='cosdec', help='decay to use: linear | exp | cos | coscos | cosdec')
    parser.add_argument('--decayparas', default='0.01,,', help='decay parameters')#0.01,55556,0.8
    parser.add_argument('--opt', default='RMSprop', help='optimizer to use: RMSprop | Adam | ooooo ()')
    parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop optimizer apha (default: 0.99)')

    parser.add_argument('--lossfunc', default='one', help='lossfunc to use: one | xxx | xxx (one)')#loss function
    parser.add_argument('--input-dtype', default='tf.uint8', help='input dtype (default: tf.uint8)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--vlossratio', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--entropycoef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')

    parser.add_argument('--aprxfunc', default='cnnmlp', help='approximate function to use: cnnmlp | mlp | ooooo (cnnmlp)')#approximate function
    parser.add_argument('--apfparas', default='8,8,4,4,32,1^4,4,2,2,64,1^3,3,1,1,64,1=512=64', help='approximate function parameters')

    parser.add_argument('--use-proper-time-limits', action='store_true', default=False, help='compute returns taking into account time limits')#other
    parser.add_argument('--use-gae', action='store_true', default=False, help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--ppo-epoch', type=int, default=4, help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32, help='number of batches for ppo (default: 32)')
    parser.add_argument('--recurrent-policy', action='store_true', default=False, help='use a recurrent policy')
def add_strings(args):
    args.exp_dir=args.exp_dir+':'+args.algo+'_'+str(args.lr_M)+'_'+args.decay+'_'+args.decayparas+'_'+args.opt
    #args.exp_dir=args.exp_dir+':'+args.lossfunc+'_'+str(args.gamma)+'_'+str(args.vlossratio)+'_'+str(args.entropycoef)+'_'+str(args.max_grad_norm)
    args.exp_dir=args.exp_dir+':'+args.aprxfunc+'_'+args.apfparas
def getAlgo(obs_space,act_space,args):
    if args.algo=='TFa2c1':
        from algos.algo_TFa2c1 import fAlgo
    if args.algo=='PTa2c1':
        from algos.algo_PTa2c1 import fAlgo
    if args.algo=='PTppo':
        from algos.algo_PTppo import fAlgo
    if args.algo=='PTa2c1M':
        from algos.algo_PTa2c1M import fAlgo
    return fAlgo(obs_space,act_space,args)

import torch,math,os,glob
from PTparts.kfac import KFACOptimizer
from PTparts.cyclicLR import CyclicCosAnnealingLR

def get_device():
    """Pick a training device that works on the current machine.
    Priority: CUDA (NVIDIA GPU) -> CPU. Set RLCASES_DEVICE to override,
    e.g. RLCASES_DEVICE=mps to try Apple Silicon, or RLCASES_DEVICE=cpu to force CPU."""
    forced = os.environ.get('RLCASES_DEVICE', '').strip().lower()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')

class PTAlgo():
    def __init__(self,obs_space,act_space,args):
        self.obs_space, self.act_space, self.args = obs_space, act_space, args
        torch.manual_seed(args.env_seed)
        torch.cuda.manual_seed_all(args.env_seed)
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    def create_optimizer(self,model):
        if self.args.opt=='RMSprop': optimizer = torch.optim.RMSprop(model.parameters(), self.args.lr, eps=self.args.eps, alpha=self.args.alpha)
        if self.args.opt=='Adam':    optimizer = torch.optim.Adam(model.parameters(), self.args.lr, eps=self.args.eps)
        if self.args.opt=='Acktr':   optimizer = KFACOptimizer(model, self.args.lr)
        return optimizer
    def _doublestones(self,max_step,unit):
        num_loop = int(math.log2(max_step//unit+1))+1
        stones = [unit*(pow(2,i+1)-1) for i in range(num_loop)]
        return stones
    def create_scheduler(self,optimizer):
        scheduler = None
        decayparas = self.args.decayparas.split(',') # eta_min_ratio, T_2, exp_gamma
        def _get(idx, default, cast):
            return cast(decayparas[idx]) if idx < len(decayparas) and decayparas[idx] != '' else default
        eta_ratio = _get(0, 0.0, float)
        T_2       = _get(1, self.args.max_train_steps, int)  # default: single cycle over whole training
        gamma     = _get(2, 1.0, float)                      # default: no per-cycle decay
        if self.args.decay=='const':  pass
        if self.args.decay=='linear': self.eta_min_ratio = eta_ratio
        if self.args.decay=='exp':    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        if self.args.decay=='cos':    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_2, eta_min=eta_ratio*self.args.lr)
        if self.args.decay=='coscos': scheduler = CyclicCosAnnealingLR(optimizer,milestones=self._doublestones(self.args.max_train_steps, T_2),
                                    decay_milestones=None,                                                          eta_min=eta_ratio*self.args.lr)
        if self.args.decay=='cosdec': scheduler = CyclicCosAnnealingLR(optimizer,milestones=self._doublestones(self.args.max_train_steps, T_2),
                                    decay_milestones=self._doublestones(self.args.max_train_steps, T_2), eta_min=eta_ratio*self.args.lr, gamma=gamma)
        return scheduler
    def update_scheduler(self,crt_step,max_step,scheduler,optimizer):
        if self.args.decay=='const': return
        if self.args.decay=='linear':
            #lr = self.args.lr*((1-self.eta_min_ratio)/2*np.cos((crt_step+1)/self.T_2*np.pi)+(1+self.eta_min_ratio)/2)
            lr = self.args.lr*((1-self.eta_min_ratio)*(1-crt_step/max_step)+self.eta_min_ratio)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return
        scheduler.step()
    def save_model(self,model,prefix,name):
        torch.save(model, self.args.checkpoint_dir+prefix+name)
    def load_model(self,folder,prefix):
        print('load_model',folder)
        print('load_model',prefix)
        if folder=='': flist = glob.glob(self.args.checkpoint_dir+prefix+'*')
        else:          flist = glob.glob(folder+prefix+'*')
        print('load_model',[fname.split('/')[-1] for fname in flist])
        flist = [ffile for ffile in flist if os.path.isfile(ffile)]
        print('load_model',[fname.split('/')[-1] for fname in flist])
        ffile = max(flist, key=os.path.getmtime)#ctime)
        print('load_model',ffile.split('/')[-1])
        return torch.load(ffile, map_location=get_device()),ffile
    def memoexps(self, new_obs, rew, done, info):
        pass
    def save(self,name,prefix=''):
        self.save_model(self.model,prefix+'',name)
    def load(self,prefix='',folder=''):
        try:
            self.model,self.model_filename = self.load_model(folder,prefix)#+'model')
            self.model.eval()
            torch.save(self.model.state_dict(),'./pymodel')
            return self.model_filename
        except:
            print('Error when trying to load model...Skipped.')
            print(prefix)
            return None
"""def doublestones(max_step,num_loop):
    num_unit = pow(2,num_loop)-1
    unit = max_step//num_unit
    stones = [unit*pow(2,i) for i in range(num_loop)]
    return stones"""
class PTWrapper():#Don't need to be an algo?
    algo = None
    def __init__(self, algo):
        self.algo = algo
    def memoexps(self, new_obs, rew, done, info, **kwargs):
        return self.algo.memoexps(new_obs, rew, done, info, **kwargs)
    def get_action(self, obs, explore=False, **kwargs):
        return self.algo.get_action(obs, explore, **kwargs)
    def get_value(self, obs, **kwargs):
        return self.algo.get_value(obs, **kwargs)
    def update(self, crt_step, max_step, info_in={}, **kwargs):
        return self.algo.update(crt_step, max_step, info_in, **kwargs)
    def save(self, name, **kwargs):
        return self.algo.save(name, **kwargs)
    def load(self, **kwargs):
        return self.algo.load(**kwargs)

import numpy as np
from collections import deque
class Memo(PTWrapper):
    def __init__(self,algo):
        PTWrapper.__init__(self,algo)
        self.obs_space, self.act_space, self.args = algo.obs_space, algo.act_space, algo.args
        self.device = algo.device
        if self.args.memoplace == "algocpu":
            self.memo_obs       = deque(maxlen=self.args.memo_size)
            self.memo_act       = deque(maxlen=self.args.memo_size)
            self.memo_nob       = deque(maxlen=self.args.memo_size)
            self.memo_rew       = deque(maxlen=self.args.memo_size)
            self.memo_done      = deque(maxlen=self.args.memo_size)
            self.memo_info      = deque(maxlen=self.args.memo_size)
            self.memo_act_info  = deque(maxlen=self.args.memo_size)
        if self.args.memoplace == "algogpu":
            self.memg_obs  = torch.zeros(self.args.memo_size, self.args.env_num, self.args.stack_num, *self.obs_space.shape).to(self.device).float()
            if self.act_space.__class__.__name__ == 'Discrete': self.memg_act  = torch.zeros(self.args.memo_size, self.args.env_num).to(self.device).long()
            else:                                               self.memg_act  = torch.zeros(self.args.memo_size, self.args.env_num, self.act_space.shape[0]).to(self.device)
            self.memg_nob  = torch.zeros(self.args.memo_size, self.args.env_num, self.args.stack_num, *self.obs_space.shape).to(self.device).float()
            self.memg_rew  = torch.zeros(self.args.memo_size, self.args.env_num, 1).to(self.device).float()
            self.memg_mask = torch.zeros(self.args.memo_size, self.args.env_num, 1).to(self.device).float()
            self.memg_size = 0
            self.memg_info     = deque(maxlen=self.args.memo_size)
            self.memg_act_info = deque(maxlen=self.args.memo_size)
    def memoexps(self, new_obs, rew, done, info):
        if self.args.memoplace == "algocpu":
            self.memo_nob.append(new_obs)
            self.memo_rew.append(rew)
            self.memo_done.append(done)
            self.memo_info.append(info)
        if self.args.memoplace == "algogpu":
            self.memg_nob[self.memg_size].copy_(torch.from_numpy(new_obs))
            self.memg_rew[self.memg_size].copy_(torch.from_numpy(rew[:,np.newaxis]))
            done_int = done.astype(int)
            done_inv = np.ones(done_int.shape)-done_int
            self.memg_mask[self.memg_size].copy_(torch.from_numpy(done_inv[:,np.newaxis]))
            self.memg_size = (self.memg_size+1)%self.args.memo_size
            self.memg_info.append(info)
    def get_action(self, inputs, explore):
        if self.args.memoplace == "algocpu":
            self.memo_obs.append(inputs)
        inputs = torch.from_numpy(inputs).float().to(self.device)
        actgpu, info_p = self.algo.get_action(inputs, explore)
        #{'value':value.cpu().numpy(),'actprob':actprob.cpu().numpy(),'entropy':entropy.cpu().numpy()}
        actcpu = actgpu.cpu().numpy()
        if self.args.memoplace == "algocpu":
            self.memo_act.append(actcpu)
            self.memo_act_info.append(info_p)
        if self.args.memoplace == "algogpu":
            self.memg_obs[self.memg_size].copy_(inputs)
            self.memg_act[self.memg_size].copy_(actgpu)
            self.memg_act_info.append(info_p)
        return actcpu, info_p
    def update(self, crt_step, max_step, info_in):
        if self.args.memoplace == "algogpu":
            info_in['mb_obs']    = self.memg_obs
            info_in['mb_act']    = self.memg_act
            info_in['mb_new_obs']= self.memg_nob
            info_in['mb_rew']    = self.memg_rew
            #dones  = info_in['mb_done'].astype(int)
            #mb_masks = np.ones(dones.shape)-dones
            info_in['mb_done']   = self.memg_mask
            info_in['mb_info']   = self.memg_info
            info_in['mb_act_info']= self.memg_act_info
        if self.args.memoplace == "algocpu":
            info_in['mb_obs']    = torch.from_numpy(np.array(self.memo_obs)).to(self.device).float()
            info_in['mb_act']    = torch.from_numpy(np.array(self.memo_act)).to(self.device)
            if self.act_space.__class__.__name__ == 'Discrete': info_in['mb_act'] = info_in['mb_act'].long()
            info_in['mb_new_obs']= torch.from_numpy(np.array(self.memo_nob)).to(self.device).float()
            info_in['mb_rew']    = torch.from_numpy(np.array(self.memo_rew)).to(self.device).float().unsqueeze(-1)
            memo_done_int = np.array(self.memo_done).astype(int)
            memo_done_inv = np.ones(memo_done_int.shape)-memo_done_int
            info_in['mb_done']   = torch.from_numpy(np.expand_dims(memo_done_inv,-1)).to(self.device).float()
            #self.mb_mask   = torch.from_numpy(memo_done).float().unsqueeze(-1).to(self.device)
            info_in['mb_info']   = self.memo_info
            info_in['mb_act_info']= self.memo_act_info
        return self.algo.update(crt_step, max_step, info_in)
