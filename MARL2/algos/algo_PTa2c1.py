import numpy as np
import torch,algos,time
import torch.nn as nn
from PTparts import PTnetwork,resnet3d,resnet2d
class Policy(nn.Module):
    def __init__(self, obs_space, act_space, args, device):
        super(Policy, self).__init__()
        self.obs_space, self.act_space, self.args = obs_space, act_space, args
        self.device = device
        if args.aprxfunc=='cnnmlp':
            apfparas = args.apfparas.split('=')
            cnncnnparas = apfparas[0].split('^')
            cnnmlpparas = apfparas[1].split('^')
            if len(obs_space.shape) == 3:   # image: (H, W, C)
                n_ch = args.stack_num * obs_space.shape[-1]
                self.base = PTnetwork.CNNBase(num_inputs=n_ch, num_outputs=int(cnnmlpparas[-1]), paraslist=cnncnnparas).to(self.device)
            elif len(obs_space.shape) == 1: # vector
                mlpmlpparas = apfparas[2].split('^') if len(apfparas) > 2 else cnnmlpparas
                self.base = PTnetwork.MLPBase(num_inputs=args.stack_num * obs_space.shape[0], paraslist=mlpmlpparas).to(self.device)
            else: raise NotImplementedError
        elif args.aprxfunc=='cnn2d':
            apfparas = args.apfparas.split('=')
            cnncnnparas = apfparas[0].split('^')
            cnnmlpparas = apfparas[1].split('^')
            self.base = PTnetwork.CNNBase2D(num_inputs=args.stack_num, num_outputs=int(cnnmlpparas[-1]), paraslist=cnncnnparas).to(self.device)
        elif args.aprxfunc=='res2d':
            apfparas = args.apfparas.split('=')
            cnncnnparas = apfparas[0].split('^')
            cnnmlpparas = apfparas[1].split('^')
            self.base = resnet2d.ResNet2D(num_inputs=args.stack_num, num_outputs=int(cnnmlpparas[-1]), paraslist=cnncnnparas).to(self.device)
        elif args.aprxfunc=='cnn3d':
            apfparas = args.apfparas.split('=')
            cnncnnparas = apfparas[0].split('^')
            cnnmlpparas = apfparas[1].split('^')
            inputs_shape = [obs_space.shape[0], obs_space.shape[1], args.stack_num, obs_space.shape[2]]
            self.base = PTnetwork.CNNBase3D(num_inputs=inputs_shape, num_outputs=int(cnnmlpparas[-1]), paraslist=cnncnnparas).to(self.device)
        elif args.aprxfunc=='res3d':
            apfparas = args.apfparas.split('=')
            cnncnnparas = apfparas[0].split('^')
            cnnmlpparas = apfparas[1].split('^')
            inputs_shape = [obs_space.shape[0], obs_space.shape[1], args.stack_num, obs_space.shape[2]]
            #self.base = ResNet(num_inputs=int(cnncnnparas[0]), num_outputs=int(cnnmlpparas[-1]), paraslist=cnncnnparas[1:]).to(self.device)
            self.base = resnet3d.ResNet3D(num_inputs=inputs_shape, num_outputs=int(cnnmlpparas[-1]), paraslist=cnncnnparas).to(self.device)
        elif args.aprxfunc=='mlp':
            apfparas = args.apfparas.split('=')
            mlpmlpparas = apfparas[0].split('^')
            self.base = PTnetwork.MLPBase(num_inputs=args.stack_num*obs_space.shape[0], paraslist=mlpmlpparas).to(self.device)
        else: raise NotImplementedError
        if act_space.__class__.__name__ == "Discrete":
            self.dist = PTnetwork.Categorical(self.base.num_outputs, act_space.n).to(self.device)
        elif act_space.__class__.__name__ == "Box":
            self.dist = PTnetwork.DiagGaussian(self.base.num_outputs, act_space.shape[0]).to(self.device)
        elif act_space.__class__.__name__ == "MultiBinary":
            self.dist = PTnetwork.Bernoulli(self.base.num_outputs, act_space.shape[0]).to(self.device)
        else: raise NotImplementedError
        args.numparas = int(sum([np.prod(p.size()) for p in self.parameters() if p.requires_grad]))#p.numel()
        print(args.numparas)
        for p in self.parameters():
            if p.requires_grad:
                print(np.prod(p.size()))
    def forward(self, inputs):
        raise NotImplementedError
    def get_action(self, inputs, explore):
        with torch.no_grad():
            value, actor_features = self.base(inputs)
            dist = self.dist(actor_features)
            if not explore: action = dist.mode()
            else:           action = dist.sample()
        info_p = {}
        return action, info_p
    def get_value(self, inputs):
        with torch.no_grad():
            value, _ = self.base(inputs)
        return value
    def get_loss(self, inputs, actions):
        values, actor_features = self.base(inputs)
        dist = self.dist(actor_features)
        if self.act_space.__class__.__name__ == "Discrete": action_log_probs = dist.log_probs(actions.squeeze(-1)).unsqueeze(-1)
        else:                                               action_log_probs = dist.log_probs(actions)
        dist_entropy = dist.entropy().mean()
        return values, action_log_probs, dist_entropy

class Algo(algos.PTAlgo):
    def __init__(self,obs_space,act_space,args):
        algos.PTAlgo.__init__(self,obs_space,act_space,args)
        self.obs_space, self.act_space, self.args = obs_space, act_space, args
        self.device = algos.get_device()
        self.model = Policy(obs_space,act_space,args,self.device)
        self.optimizer = self.create_optimizer(self.model)
        self.scheduler = self.create_scheduler(self.optimizer)
        if self.act_space.__class__.__name__ == 'Discrete': self.action_shape = 1
        else:                                               self.action_shape = self.act_space.shape[0]
        if self.args.timer:
            self.Algo_get_action_pre, self.Algo_get_action_crt, self.Algo_get_action_pst = 0,0,0
            self.Algo_get_update_pre, self.Algo_get_update_crt = 0,0
            self.Algo_get_update_rew, self.Algo_get_update_los = 0,0
            self.Algo_get_action_syn = 0
            self.Algo_get_update_syn = 0
    def __del__(self):
        if self.args.timer:
            print('Algo_get_action_pre:',round(self.Algo_get_action_pre/60,2),' minutes')
            print('Algo_get_action_crt:',round(self.Algo_get_action_crt/60,2),' minutes')
            print('Algo_get_action_pst:',round(self.Algo_get_action_pst/60,2),' minutes')
            print('Algo_get_update_pre:',round(self.Algo_get_update_pre/60,2),' minutes')
            print('Algo_get_update_crt:',round(self.Algo_get_update_crt/60,2),' minutes')
            print('Algo_get_update_rew:',round(self.Algo_get_update_rew/60,2),' minutes')
            print('Algo_get_update_los:',round(self.Algo_get_update_los/60,2),' minutes')
            print('Algo_get_action_syn:',round(self.Algo_get_action_syn/60,2),' minutes')
            print('Algo_get_update_syn:',round(self.Algo_get_update_syn/60,2),' minutes')
    def get_action(self, inputs, explore):
        if self.args.timer: self.Algo_get_action_pre_start = time.process_time()
        if self.args.memoplace == "algocpu" or self.args.memoplace == "algogpu":
            if self.args.timer: self.Algo_get_action_pre += time.process_time()-self.Algo_get_action_pre_start
            if self.args.timer: self.Algo_get_action_crt_start = time.process_time()
            action, info_p = self.model.get_action(inputs, explore)
            if self.args.timer: self.Algo_get_action_crt += time.process_time()-self.Algo_get_action_crt_start
            if self.args.timer: self.Algo_get_action_pst_start = time.process_time()
        else:
            inputs = torch.from_numpy(inputs).float().to(self.device)
            if self.args.timer: self.Algo_get_action_pre += time.process_time()-self.Algo_get_action_pre_start
            if self.args.timer: self.Algo_get_action_crt_start = time.process_time()
            action, info_p = self.model.get_action(inputs, explore)
            if self.args.timer: self.Algo_get_action_syn_start = time.process_time()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            if self.args.timer: self.Algo_get_action_syn += time.process_time()-self.Algo_get_action_syn_start
            if self.args.timer: self.Algo_get_action_crt += time.process_time()-self.Algo_get_action_crt_start
            if self.args.timer: self.Algo_get_action_pst_start = time.process_time()
            #print('***action shape:',action.shape)
            action = action.cpu().numpy()
            #exit()
        if self.args.timer: self.Algo_get_action_pst += time.process_time()-self.Algo_get_action_pst_start
        return action, info_p
    def get_value(self, inputs):
        value = self.model.get_value(inputs)
        return value
    def update(self, crt_step, max_step, info_in):
        if self.args.timer: self.Algo_get_update_pre_start = time.process_time()
        if self.args.memoplace == "algocpu" or self.args.memoplace == "algogpu":
            self.mb_obs    = info_in['mb_obs']
            self.mb_act    = info_in['mb_act']
            self.mb_nob    = info_in['mb_new_obs']
            self.mb_rew    = info_in['mb_rew']
            self.mb_mask   = info_in['mb_done']
        else:
            self.mb_obs    = torch.from_numpy(info_in['mb_obs']).to(self.device).float()
            self.mb_act    = torch.from_numpy(info_in['mb_act']).to(self.device)
            if self.act_space.__class__.__name__ == 'Discrete': self.mb_act = self.mb_act.long()
            self.mb_nob    = torch.from_numpy(info_in['mb_new_obs']).to(self.device).float()
            self.mb_rew    = torch.from_numpy(info_in['mb_rew']).to(self.device).float().unsqueeze(-1)
            mb_done_int = info_in['mb_done'].astype(int)
            mb_done_inv = np.ones(mb_done_int.shape)-mb_done_int
            self.mb_mask   = torch.from_numpy(np.expand_dims(mb_done_inv,-1)).to(self.device).float()
            #self.mb_mask   = torch.from_numpy(memo_done).float().unsqueeze(-1).to(self.device)
        if self.args.timer: self.Algo_get_update_rew_start = time.process_time()
        self.returns   = torch.zeros(self.args.memo_size+1, self.args.env_num, 1).to(self.device).float()
        if 0:
            print('update')
            print('self.mb_obs',self.mb_obs.shape)
            print('self.mb_act',self.mb_act.shape)
            print('self.mb_nob',self.mb_nob.shape)
            print('self.mb_rew',self.mb_rew.shape)
            print('self.mb_mask',self.mb_mask.shape)
            print('self.returns',self.returns.shape)
            exit()
        self.returns[-1] = self.get_value(self.mb_nob[-1]).detach()
        for step in reversed(range(self.mb_rew.size(0))):
            self.returns[step] = self.returns[step+1]*self.args.gamma*self.mb_mask[step] + self.mb_rew[step]
        returns = self.returns[:-1]
        if self.args.timer: self.Algo_get_update_rew += time.process_time()-self.Algo_get_update_rew_start

        if self.args.timer: self.Algo_get_update_los_start = time.process_time()
        self.mb_obs_batch = self.mb_obs.view(-1, self.args.stack_num, *self.obs_space.shape)
        self.mb_act_batch = self.mb_act.view(-1, self.action_shape)
        values, action_log_probs, dist_entropy = self.model.get_loss(self.mb_obs_batch,self.mb_act_batch)
        if 0:
            print('values',values.shape)
            print('returns',returns.shape)
            exit()
        returns = returns.view(-1,1)
        advantages = returns - values
        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()
        if self.args.timer: self.Algo_get_update_los += time.process_time()-self.Algo_get_update_los_start
        if self.args.timer: self.Algo_get_update_pre += time.process_time()-self.Algo_get_update_pre_start

        if self.args.timer: self.Algo_get_update_crt_start = time.process_time()
        if self.args.opt=='Acktr' and self.optimizer.steps % self.optimizer.Ts == 0:# Sampled fisher, see Martens 2014
            self.model.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda: value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        (value_loss*self.args.vlossratio + action_loss - dist_entropy*self.args.entropycoef).backward()
        if self.args.timer: self.Algo_get_action_syn_start = time.process_time()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        if self.args.timer: self.Algo_get_action_syn += time.process_time()-self.Algo_get_action_syn_start
        if self.args.opt!='Acktr': nn.utils.clip_grad_norm_(self.model.parameters(),self.args.max_grad_norm)
        self.optimizer.step()
        self.update_scheduler(crt_step,max_step,self.scheduler,self.optimizer)
        if self.args.timer: self.Algo_get_update_crt += time.process_time()-self.Algo_get_update_crt_start
        return value_loss.item(), action_loss.item(), dist_entropy.item()

def fAlgo(obs_space,act_space,args):
    algo = Algo(obs_space,act_space,args)
    if args.memoplace == "algocpu" or args.memoplace == "algogpu": algo = algos.Memo(algo)
    return algo
