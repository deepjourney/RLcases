import numpy as np
import agents, time
from collections import deque
class Memo(agents.Wrapper):
    def __init__(self,agt):
        agents.Wrapper.__init__(self,agt)
        agt.attr.memo_obs       = deque(maxlen=agt.attr.args.memo_size)
        agt.attr.memo_act       = deque(maxlen=agt.attr.args.memo_size)
        agt.attr.memo_new_obs   = deque(maxlen=agt.attr.args.memo_size)
        agt.attr.memo_rew       = deque(maxlen=agt.attr.args.memo_size)
        agt.attr.memo_done      = deque(maxlen=agt.attr.args.memo_size)
        agt.attr.memo_info      = deque(maxlen=agt.attr.args.memo_size)
        agt.attr.memo_act_info  = deque(maxlen=agt.attr.args.memo_size)
        if self.attr.args.timer:
            self.Memo_pre, self.Memo_crt, self.Memo_pst = 0,0,0
            self.Memo_update_pre, self.Memo_update_crt = 0,0
            self.Memo_memoexps_pre, self.Memo_memoexps_crt = 0,0
    def __del__(self):
        if self.attr.args.timer:
            print('Memo_pre:',np.round(self.Memo_pre/60,2),' minutes')
            print('Memo_crt:',np.round(self.Memo_crt/60,2),' minutes')
            print('Memo_pst:',np.round(self.Memo_pst/60,2),' minutes')
            print('Memo_update_pre:',np.round(self.Memo_update_pre/60,2),' minutes')
            print('Memo_update_crt:',np.round(self.Memo_update_crt/60,2),' minutes')
            print('Memo_memoexps_pre:',np.round(self.Memo_memoexps_pre/60,2),' minutes')
            print('Memo_memoexps_crt:',np.round(self.Memo_memoexps_crt/60,2),' minutes')
    def memoexps(self, new_obs, rew, done, info):
        if self.attr.args.timer: self.Memo_memoexps_crt_start = self.process_time()
        self.agt.memoexps(new_obs, rew, done, info)
        if self.attr.args.timer: self.Memo_memoexps_crt += self.process_time()-self.Memo_memoexps_crt_start
        if self.attr.args.timer: self.Memo_memoexps_pre_start = self.process_time()
        self.agt.attr.memo_new_obs.append(new_obs)
        self.agt.attr.memo_rew.append(rew)
        self.agt.attr.memo_done.append(done)
        self.agt.attr.memo_info.append(info)
        if self.attr.args.timer: self.Memo_memoexps_pre += self.process_time()-self.Memo_memoexps_pre_start
    def getaction(self, obs, explore):
        if self.attr.args.timer: self.Memo_crt_start = self.process_time()
        act, act_info = self.agt.getaction(obs,explore)
        if self.attr.args.timer: self.Memo_crt += self.process_time()-self.Memo_crt_start
        if self.attr.args.timer: self.Memo_pst_start = self.process_time()
        self.agt.attr.memo_obs.append(obs)
        self.agt.attr.memo_act.append(act)
        self.agt.attr.memo_act_info.append(act_info)
        if self.attr.args.timer: self.Memo_pst += self.process_time()-self.Memo_pst_start
        return act, act_info
    def update(self, crt_step, max_step, info_in={}):
        if self.attr.args.timer: self.Memo_update_pre_start = self.process_time()
        info_in =  {'mb_obs':       self.agt.attr.memo_obs,#np.array(self.agt.attr.memo_obs),
                    'mb_act':       self.agt.attr.memo_act,#np.array(self.agt.attr.memo_act),
                    'mb_new_obs':   self.agt.attr.memo_new_obs,#np.array(self.agt.attr.memo_new_obs),
                    'mb_rew':       self.agt.attr.memo_rew,#np.array(self.agt.attr.memo_rew),
                    'mb_done':      self.agt.attr.memo_done,#np.array(self.agt.attr.memo_done),
                    'mb_info':      self.agt.attr.memo_info,
                    'mb_act_info':  self.agt.attr.memo_act_info, **info_in}
        if self.attr.args.timer: self.Memo_update_pre += self.process_time()-self.Memo_update_pre_start
        if self.attr.args.timer: self.Memo_update_crt_start = self.process_time()
        update_result = self.agt.update(crt_step=crt_step, max_step=max_step, info_in=info_in)
        if self.attr.args.timer: self.Memo_update_crt += self.process_time()-self.Memo_update_crt_start
        return update_result

class Stack(agents.Wrapper):
    def __init__(self,agt):
        agents.Wrapper.__init__(self,agt)
        agt.attr.obs_stack = np.zeros([agt.attr.args.env_num]+[agt.attr.args.stack_num]+list(agt.attr.stack_shape), dtype=agt.attr.stack_dtype)
        if self.attr.args.timer:
            self.Stack_pre, self.Stack_crt, self.Stack_pst = 0,0,0
    def __del__(self):
        if self.attr.args.timer:
            print('Stack_pre:',np.round(self.Stack_pre/60,2),' minutes')
            print('Stack_crt:',np.round(self.Stack_crt/60,2),' minutes')
            print('Stack_pst:',np.round(self.Stack_pst/60,2),' minutes')
    def obs_stack_update(self, new_obs, old_obs_stack):
        updated_obs_stack = np.roll(old_obs_stack, shift=-1, axis=1)
        updated_obs_stack[:,-1,:] = new_obs#[:]###
        return updated_obs_stack
    def memoexps(self, new_obs, rew, done, info):
        new_stack = self.obs_stack_update(new_obs, self.agt.attr.obs_stack)
        self.agt.memoexps(new_stack, rew, done, info)
        for i,donei in enumerate(done):
            if donei:
                self.agt.attr.obs_stack[i]*=0#[:-1]*=0
    def getaction(self, obs, explore):
        if self.attr.args.timer: self.Stack_pre_start = self.process_time()
        self.agt.attr.obs_stack = self.obs_stack_update(obs, self.agt.attr.obs_stack)
        obs_stack_copy = np.copy(self.agt.attr.obs_stack)###must copy!
        if self.attr.args.timer: self.Stack_pre += self.process_time()-self.Stack_pre_start
        if self.attr.args.timer: self.Stack_crt_start = self.process_time()
        act, act_info = self.agt.getaction(obs_stack_copy,explore)
        if self.attr.args.timer: self.Stack_crt += self.process_time()-self.Stack_crt_start
        return act, act_info

class Imagine(agents.Wrapper):
    def __init__(self,agt):
        agents.Wrapper.__init__(self,agt)
        imgparas = agt.attr.args.imgparas.split(',')
        self.imagine_method = imgparas[0]
        self.imagine_start  = imgparas[1]
        self.imagine_reward = float(imgparas[2])
        if self.attr.args.timer:
            self.Imagine_pre, self.Imagine_crt, self.Imagine_pst = 0,0,0
    def __del__(self):
        if self.attr.args.timer:
            print('Imagine_pre:',np.round(self.Imagine_pre/60,2),' minutes')
            print('Imagine_crt:',np.round(self.Imagine_crt/60,2),' minutes')
            print('Imagine_pst:',np.round(self.Imagine_pst/60,2),' minutes')
    def predict(self): # to do
        return self.agt.attr.obs_stack[:,-1,:]
    def imagine(self, obs):
        if self.imagine_method == 'none':
            return obs
        if self.imagine_method == 'prev':
            img = self.predict()
        else: # 'zero'
            img = 0
        fix = np.where(obs==None, img, obs)
        return fix
    def intrinsic_reward(self, obs, act, new_obs): # to do
        return 0
    def memoexps(self, new_obs, rew, done, info):
        imagineobs = self.imagine(new_obs)
        self.agt.memoexps(imagineobs, rew, done, info)
        if self.imagine_start == 'full':
            for i,donei in enumerate(done):
                if donei:
                    self.agt.attr.obs_stack[i][-1] = self.agt.attr.initobs[i]
    def getaction(self, obs, explore):
        if self.attr.args.timer: self.Imagine_pre_start = self.process_time()
        imagineobs = self.imagine(obs)
        if self.attr.args.timer: self.Imagine_pre += self.process_time()-self.Imagine_pre_start
        if self.attr.args.timer: self.Imagine_crt_start = self.process_time()
        act, act_info = self.agt.getaction(imagineobs, explore)
        if self.attr.args.timer: self.Imagine_crt += self.process_time()-self.Imagine_crt_start
        return act, act_info
    def update(self, crt_step, max_step, info_in={}):
        if self.imagine_reward != 0:
            mb_intrew = self.intrinsic_reward(np.array(self.agt.attr.memo_obs), np.array(self.agt.attr.memo_act), np.array(self.agt.attr.memo_new_obs))######
            info_in =  {'mb_intrew': mb_intrew, **info_in}
        return self.agt.update(crt_step=crt_step, max_step=max_step, info_in=info_in)

"""
class Space(object):
    def __init__(self, shape=None, dtype=None):
        import numpy as np # takes about 300-400ms to import, so we load lazily
        self.shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else np.dtype(dtype)

    def sample(self):
        raise NotImplementedError

    def contains(self, x):
        raise NotImplementedError

    __contains__ = contains

    def to_jsonable(self, sample_n):
        # Convert a batch of samples from this space to a JSONable data type.
        # By default, assume identity is JSONable
        return sample_n

    def from_jsonable(self, sample_n):
        # Convert a JSONable data type to a batch of samples from this space.
        # By default, assume identity is JSONable
        return sample_n

class ObservationWrapper(Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def observation(self, observation):
        deprecated_warn_once("%s doesn't implement 'observation' method. Maybe it implements deprecated '_observation' method." % type(self))
        return self._observation(observation)


class RewardWrapper(Wrapper):
    def reset(self):
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        deprecated_warn_once("%s doesn't implement 'reward' method. Maybe it implements deprecated '_reward' method." % type(self))
        return self._reward(reward)


class ActionWrapper(Wrapper):
    def step(self, action):
        action = self.action(action)
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def action(self, action):
        deprecated_warn_once("%s doesn't implement 'action' method. Maybe it implements deprecated '_action' method." % type(self))
        return self._action(action)

    def reverse_action(self, action):
        deprecated_warn_once("%s doesn't implement 'reverse_action' method. Maybe it implements deprecated '_reverse_action' method." % type(self))
        return self._reverse_action(action)
"""
