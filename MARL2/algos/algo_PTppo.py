import numpy as np
import torch,algos
import torch.nn as nn
from PTparts import PTnetwork
class Policy(nn.Module):
    def __init__(self, obs_space, act_space, args, device):
        super(Policy, self).__init__()
        self.obs_space, self.act_space, self.args = obs_space, act_space, args
        self.device = device
        if args.aprxfunc=='cnnmlp':
            apfparas = args.apfparas.split('=')
            cnncnnparas = apfparas[0].split('^')
            cnnmlpparas = apfparas[1].split('^')
            mlpmlpparas = apfparas[2].split('^')
            if len(obs_space.shape) == 3:
                self.base = PTnetwork.CNNBase(num_inputs=args.stack_num, num_outputs=int(cnnmlpparas[-1]), paraslist=cnncnnparas).to(self.device)
            elif len(obs_space.shape) == 1:
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
        if debug: self.params = list(self.base.parameters()) + list(self.dist.parameters())
    def forward(self, inputs):
        raise NotImplementedError
    def get_action(self, inputs, explore):
        with torch.no_grad():
            value, actor_features = self.base(inputs)
            dist = self.dist(actor_features)
            if not explore: action = dist.mode()
            else:           action = dist.sample()
            actprob = dist.log_probs(action)
            entropy = dist.entropy().mean()
            #if self.act_space.__class__.__name__ == "Discrete":
            #    action  = action.unsqueeze(-1)
            #    actprob = actprob.unsqueeze(-1)
        info_p = {'value':value.cpu().numpy(),'actprob':actprob.cpu().numpy(),'entropy':entropy.cpu().numpy()}
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

import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
debug = 0
class Algo(algos.PTAlgo):
    def __init__(self,obs_space,act_space,args):
        algos.PTAlgo.__init__(self,obs_space,act_space,args)
        self.obs_space, self.act_space, self.args = obs_space, act_space, args
        self.device = torch.device("cuda:0")
        self.model = Policy(obs_space,act_space,args,self.device)
        self.optimizer = self.create_optimizer(self.model)
        self.scheduler = self.create_scheduler(self.optimizer)
        if self.act_space.__class__.__name__ == 'Discrete': self.action_shape = 1
        else:                                               self.action_shape = self.act_space.shape[0]
    def get_action(self, inputs, explore):
        if self.args.memoplace == "algocpu" or self.args.memoplace == "algogpu":
            action, info_p = self.model.get_action(inputs, explore)
        else:
            inputs = torch.from_numpy(inputs).float().to(self.device)
            action, info_p = self.model.get_action(inputs, explore)
            action = action.cpu().numpy()
        return action, info_p
    def get_value(self, inputs):
        value = self.model.get_value(inputs)
        return value
    def feed_forward_generator(self, num_mini_batch, mini_batch_size=None):
        batch_size = self.args.env_num * self.args.memo_size
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)
        for indices in sampler:
            obs_batch         = self.mb_obs_batch[indices]#*self.mb_obs.size()[2:])[indices]#self.args.stack_num, *self.obs_space.shape)[indices]
            actions_batch     = self.mb_act_batch[indices]#self.mb_act.size()[-1])[indices]#if self.act_space.__class__.__name__ == 'Discrete': self.action_shape = 1
            masks_batch       = self.mb_mask_batch[indices]
            values_batch      = self.values_batch[indices]
            old_alp_batch     = self.action_log_probs_batch[indices]
            return_batch      = self.returns_batch[indices]
            adv_batch         = self.advantages_batch[indices]
            yield obs_batch, actions_batch, masks_batch, values_batch, old_alp_batch, return_batch, adv_batch
    def update(self, crt_step, max_step, info_in):
        if debug: print('update')
        if debug: fdebug = open('debuglogs','w')
        self.update_scheduler(crt_step,max_step,self.scheduler,self.optimizer)
        if self.args.memoplace == "algocpu" or self.args.memoplace == "algogpu":
            self.mb_obs    = info_in['mb_obs']
            self.mb_act    = info_in['mb_act']
            self.mb_nob    = info_in['mb_new_obs']
            self.mb_rew    = info_in['mb_rew']
            self.mb_mask   = info_in['mb_done']
            mb_value   = np.array([info_p['value'] for info_p in info_in['mb_act_info']])
            mb_actprob = np.array([info_p['actprob'] for info_p in info_in['mb_act_info']])
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
            mb_value   = np.array([info_p['value'] for info_p in info_in['mb_act_info']])
            mb_actprob = np.array([info_p['actprob'] for info_p in info_in['mb_act_info']])
        self.values           = torch.from_numpy(mb_value).to(self.device).float()
        self.action_log_probs = torch.from_numpy(mb_actprob).to(self.device).float()
        self.returns   = torch.zeros(self.args.memo_size+1, self.args.env_num, 1).to(self.device).float()
        next_value = self.get_value(self.mb_nob[-1]).detach()
        if self.args.use_gae:
            value_preds= torch.zeros(self.args.memo_size+1, self.args.env_num, 1).to(self.device)
            value_preds[-1], gae = next_value, 0
            for step in reversed(range(self.mb_rew.size(0))):
                value_preds[step] = self.values[step]
                delta = self.mb_rew[step] + self.args.gamma*value_preds[step+1]*self.mb_mask[step] - value_preds[step]
                gae = delta + self.args.gamma*self.args.gae_lambda*self.mb_mask[step]*gae
                self.returns[step] = gae + value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.mb_rew.size(0))):
                self.returns[step] = self.returns[step+1]*self.args.gamma*self.mb_mask[step] + self.mb_rew[step]
        self.advantages = self.returns[:-1] - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-5)

        self.mb_obs_batch          = self.mb_obs.view(-1, self.args.stack_num, *self.obs_space.shape)
        self.mb_act_batch          = self.mb_act.view(-1, self.action_shape)
        self.mb_mask_batch         = self.mb_mask.view(-1, 1)
        self.values_batch          = self.values.view(-1, 1)
        self.action_log_probs_batch= self.action_log_probs.view(-1, 1)
        self.returns_batch         = self.returns.view(-1, 1)
        self.advantages_batch      = self.advantages.view(-1, 1)

        self.clip_param = self.args.clip_param
        self.vlossratio = self.args.vlossratio
        self.entropycoef= self.args.entropycoef
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        if debug: print('self.mb_obs',self.mb_obs,file=fdebug)
        if debug: print('self.mb_act',self.mb_act,file=fdebug)
        if debug: print('self.mb_mask',self.mb_mask,file=fdebug)
        if debug: print('self.values',self.values,file=fdebug)
        if debug: print('self.action_log_probs',self.action_log_probs,file=fdebug)
        if debug: print('self.returns',self.returns,file=fdebug)
        if debug: print('self.advantages',self.advantages,file=fdebug)
        for e in range(self.args.ppo_epoch):
            data_generator = self.feed_forward_generator(self.args.num_mini_batch)
            sami = 0
            for sample in data_generator:
                if debug: print('sami',sami,file=fdebug)
                obs_batch, actions_batch, masks_batch, values_batch, old_alp_batch, return_batch, adv_targ = sample
                if debug: print('obs_batch',obs_batch, 'actions_batch',actions_batch,file=fdebug)
                values_new, action_log_probs_new, dist_entropy_new = self.model.get_loss(obs_batch,actions_batch)
                if debug: print('values_new',values_new, 'action_log_probs_new',action_log_probs_new, 'dist_entropy_new',dist_entropy_new,file=fdebug)

                ratio = torch.exp(action_log_probs_new - old_alp_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()
                if 1:#self.use_clipped_value_loss: #important!!!!!!!
                    values_new_clipped = values_batch + (values_new - values_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values_new - return_batch).pow(2)
                    value_losses_clipped = (values_new_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values_new).pow(2).mean()

                self.optimizer.zero_grad()
                if debug: print('allparas',self.model.params,file=fdebug)
                if debug: print(value_loss,action_loss,dist_entropy_new,file=fdebug)
                if debug: print(self.vlossratio,self.entropycoef,file=fdebug)
                if debug: print('optimizer',self.optimizer,file=fdebug)
                if debug: 
                    for param in self.model.parameters():
                        print('allparasgradbeforebefore',param.grad,file=fdebug)
                (value_loss*self.vlossratio + action_loss - dist_entropy_new*self.entropycoef).backward()
                if debug:
                    for param in self.model.parameters():
                        print('allparasgradbefore',param.grad,file=fdebug)
                nn.utils.clip_grad_norm_(self.model.parameters(),self.args.max_grad_norm)
                if debug:
                    for param in self.model.parameters():
                        print('allparasgradafter',param.grad,file=fdebug)
                self.optimizer.step()
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy_new.item()
                sami += 1
                if debug: 
                    if sami == 3: exit()
        num_updates = self.args.ppo_epoch * self.args.num_mini_batch
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

def fAlgo(obs_space,act_space,args):
    algo = Algo(obs_space,act_space,args)
    if args.memoplace == "algocpu" or args.memoplace == "algogpu": algo = algos.Memo(algo)
    return algo
