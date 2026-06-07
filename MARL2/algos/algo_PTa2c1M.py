import numpy as np
import torch,algos,time
import torch.nn as nn
from PTparts import PTnetwork,resnet3d,resnet2d
import torchvision
debug = 0
class Policy(nn.Module):
    def __init__(self, obs_space, act_space, args, device):
        super(Policy, self).__init__()
        self.obs_space, self.act_space, self.args = obs_space, act_space, args
        self.device = device
        if args.aprxfunc=='cnn2d':
            apfparas = args.apfparas.split('=')
            cnncnnparas = apfparas[0].split('^')
            cnnmlpparas = apfparas[1].split('^')
            inputs_shape = [obs_space.shape[0], obs_space.shape[1], args.stack_num*obs_space.shape[2]]
            self.base = PTnetwork.CNNBase2D(num_inputs=inputs_shape, num_outputs=int(cnnmlpparas[-1]), paraslist=cnncnnparas).to(self.device)
        elif args.aprxfunc=='res2d':
            apfparas = args.apfparas.split('=')
            cnncnnparas = apfparas[0].split('^')
            cnnmlpparas = apfparas[1].split('^')
            inputs_shape = [obs_space.shape[0], obs_space.shape[1], args.stack_num*obs_space.shape[2]]
            self.base = resnet2d.ResNet2D(num_inputs=inputs_shape, num_outputs=int(cnnmlpparas[-1]), paraslist=cnncnnparas).to(self.device)
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
        #return torch.ones([inputs.shape[0],1], dtype=torch.int, device="cuda:0"),{}
        with torch.no_grad():
            value, actor_features = self.base(inputs)
            dist = self.dist(actor_features)
            if not explore: action = dist.mode()
            else:           action = dist.sample()
        #actprob = dist.log_probs(action)
        info_p = {'probs':dist.probs}#'actprob':actprob
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
            self.Algo_get_action_rsp, self.Algo_get_action_cpu = 0,0
            self.Algo_get_action_syn = 0
            self.Algo_get_update_syn = 0
    def process_time(self):
        return np.array([time.process_time(),time.perf_counter()])
    def __del__(self):
        if self.args.timer:
            print('Algo_get_action_pre:',np.round(self.Algo_get_action_pre/60,2),' minutes')
            print('Algo_get_action_crt:',np.round(self.Algo_get_action_crt/60,2),' minutes')
            print('Algo_get_action_pst:',np.round(self.Algo_get_action_pst/60,2),' minutes')
            print('Algo_get_update_pre:',np.round(self.Algo_get_update_pre/60,2),' minutes')
            print('Algo_get_update_crt:',np.round(self.Algo_get_update_crt/60,2),' minutes')
            print('Algo_get_update_rew:',np.round(self.Algo_get_update_rew/60,2),' minutes')
            print('Algo_get_update_los:',np.round(self.Algo_get_update_los/60,2),' minutes')
            print('Algo_get_action_rsp:',np.round(self.Algo_get_action_rsp/60,2),' minutes')
            print('Algo_get_action_cpu:',np.round(self.Algo_get_action_cpu/60,2),' minutes')
            print('Algo_get_action_syn:',np.round(self.Algo_get_action_syn/60,2),' minutes')
            print('Algo_get_update_syn:',np.round(self.Algo_get_update_syn/60,2),' minutes')
    def get_action(self, inputsall, explore):
        if self.args.timer: self.Algo_get_action_pre_start = self.process_time()
        if debug: print('Algo get_action:',inputsall.shape,self.obs_space.shape,self.obs_space,self.act_space)
        inputsall_transpose = np.transpose(inputsall,(0,2,1,3,4,5))
        if debug: print('Algo get_action:',inputsall_transpose.shape,self.obs_space.shape,self.obs_space,self.act_space)
        inputs = inputsall.reshape(-1, self.args.stack_num, *self.obs_space.shape)
        if debug: print('Algo get_action:',inputs.shape,self.obs_space.shape,self.obs_space,self.act_space)
        #if self.args.timer: self.Algo_get_action_pre += self.process_time()-self.Algo_get_action_pre_start

        if self.args.memoplace == "algocpu" or self.args.memoplace == "algogpu":
            if self.args.timer: self.Algo_get_action_pre += self.process_time()-self.Algo_get_action_pre_start
            if self.args.timer: self.Algo_get_action_crt_start = self.process_time()
            action, info_p = self.model.get_action(inputs, explore)
            if self.args.timer: self.Algo_get_action_crt += self.process_time()-self.Algo_get_action_crt_start
            if self.args.timer: self.Algo_get_action_pst_start = self.process_time()
        else:
            inputs = torch.from_numpy(inputs).float().to(self.device)
            if self.args.timer: self.Algo_get_action_pre += self.process_time()-self.Algo_get_action_pre_start
            if self.args.timer: self.Algo_get_action_crt_start = self.process_time()
            action_gpu, info_p_gpu = self.model.get_action(inputs, explore)
            if self.args.timer: self.Algo_get_action_syn_start = self.process_time()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            if self.args.timer: self.Algo_get_action_syn += self.process_time()-self.Algo_get_action_syn_start
            if self.args.timer: self.Algo_get_action_crt += self.process_time()-self.Algo_get_action_crt_start
            if self.args.timer: self.Algo_get_action_pst_start = self.process_time()
            if self.args.timer: self.Algo_get_action_cpu_start = self.process_time()
            action = action_gpu.cpu().numpy()
            info_p = {}
            for key,value in info_p_gpu.items():
                info_p[key]=value.cpu().numpy()
            if self.args.timer: self.Algo_get_action_cpu += self.process_time()-self.Algo_get_action_cpu_start

        #if self.args.timer: self.Algo_get_action_pst_start = self.process_time()
        if self.args.timer: self.Algo_get_action_rsp_start = self.process_time()
        if debug: print('action',action.shape)
        actionall = action.reshape(self.args.env_num,-1)
        for key,value in info_p.items():
            info_p[key]=value.reshape(self.args.env_num,-1)
        if debug: print('actionall',actionall.shape)
        if self.args.timer: self.Algo_get_action_rsp += self.process_time()-self.Algo_get_action_rsp_start
        if self.args.timer: self.Algo_get_action_pst += self.process_time()-self.Algo_get_action_pst_start
        return actionall, info_p
    def get_value(self, inputsall):
        if debug: print('Algo get_value:',inputsall.shape,self.obs_space.shape,self.obs_space,self.act_space)
        inputsall_transpose = inputsall.permute(0,2,1,3,4,5) #np.transpose(inputsall,(0,2,1,3,4,5))
        if debug: print('Algo get_value:',inputsall_transpose.shape,self.obs_space.shape,self.obs_space,self.act_space)
        inputs = inputsall.view(-1, self.args.stack_num, *self.obs_space.shape)
        if debug: print('Algo get_value:',inputs.shape,self.obs_space.shape,self.obs_space,self.act_space)

        value = self.model.get_value(inputs)

        if debug: print('value',value.shape)
        valueall = value.view(self.args.env_num,-1)
        if debug: print('valueall',valueall.shape)
        return valueall
    def update(self, crt_step, max_step, info_in):
        #return
        if self.args.timer: self.Algo_get_update_pre_start = self.process_time()
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
        if debug: print('update')
        if debug: print('self.mb_obs',self.mb_obs.shape)
        if debug: print('self.mb_act',self.mb_act.shape)
        if debug: print('self.mb_nob',self.mb_nob.shape)
        if debug: print('self.mb_rew',self.mb_rew.shape)
        if debug: print('self.mb_mask',self.mb_mask.shape)
        self.mb_mask = self.mb_mask.repeat(1,1,self.mb_rew.shape[-2]).unsqueeze(-1)
        if debug: print('self.mb_mask',self.mb_mask.shape)

        if self.args.timer: self.Algo_get_update_rew_start = self.process_time()
        self.returns   = torch.zeros(self.args.memo_size+1, self.args.env_num, 1).to(self.device).float()
        if debug: print('self.returns',self.returns.shape)
        self.returns = self.returns.repeat(1,1,self.mb_rew.shape[-2]).unsqueeze(-1)
        if debug: print('self.returns',self.returns.shape)
        last_value = self.get_value(self.mb_nob[-1]).detach().unsqueeze(-1)
        if debug: print('last_value',last_value.shape)
        self.returns[-1] = last_value
        for step in reversed(range(self.mb_rew.size(0))):
            self.returns[step] = self.returns[step+1]*self.args.gamma*self.mb_mask[step] + self.mb_rew[step]
        returns = self.returns[:-1]
        if debug: print('returns',returns.shape)
        if self.args.timer: self.Algo_get_update_rew += self.process_time()-self.Algo_get_update_rew_start

        if self.args.timer: self.Algo_get_update_los_start = self.process_time()
        mb_obs = self.mb_obs.permute(0,1,3,2,4,5,6)
        mb_act = self.mb_act
        if debug: print('mb_obs',mb_obs.shape)
        if debug: print('mb_act',mb_act.shape)
        if debug: print('action_shape',self.action_shape)
        self.mb_obs_batch = self.mb_obs.view(-1, self.args.stack_num, *self.obs_space.shape)
        self.mb_act_batch = self.mb_act.view(-1, self.action_shape)
        if debug: print('self.mb_obs_batch',self.mb_obs_batch.shape)
        if debug: print('self.mb_act_batch',self.mb_act_batch.shape)
        values, action_log_probs, dist_entropy = self.model.get_loss(self.mb_obs_batch,self.mb_act_batch)
        if debug: print('values',values.shape)
        if debug: print('action_log_probs',action_log_probs.shape)
        if debug: print('dist_entropy',dist_entropy.shape)
        returns = returns.view(-1,1)
        if debug: print('returns',returns.shape)
        advantages = returns - values
        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()
        if self.args.timer: self.Algo_get_update_los += self.process_time()-self.Algo_get_update_los_start

        if debug: print('advantages',advantages.shape)
        if debug: print('value_loss',value_loss.shape)
        if debug: print('action_loss',action_loss.shape)

        if debug: print('before parameters update')
        if debug: print('self.mb_obs',self.mb_obs.shape)
        if debug: print('self.mb_act',self.mb_act.shape)
        if debug: print('self.mb_nob',self.mb_nob.shape)
        if debug: print('self.mb_rew',self.mb_rew.shape)
        if debug: print('self.mb_mask',self.mb_mask.shape)
        if debug: exit()
        if self.args.timer: self.Algo_get_update_pre += self.process_time()-self.Algo_get_update_pre_start

        if self.args.timer: self.Algo_get_update_crt_start = self.process_time()
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
        if self.args.timer: self.Algo_get_update_syn_start = self.process_time()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        if self.args.timer: self.Algo_get_update_syn += self.process_time()-self.Algo_get_update_syn_start
        if self.args.opt!='Acktr': nn.utils.clip_grad_norm_(self.model.parameters(),self.args.max_grad_norm)
        self.optimizer.step()
        self.update_scheduler(crt_step,max_step,self.scheduler,self.optimizer)
        if self.args.timer: self.Algo_get_update_crt += self.process_time()-self.Algo_get_update_crt_start
        return value_loss.item(), action_loss.item(), dist_entropy.item()

def fAlgo(obs_space,act_space,args):
    algo = Algo(obs_space,act_space,args)
    if args.memoplace == "algocpu" or args.memoplace == "algogpu": algo = algos.Memo(algo)
    return algo
