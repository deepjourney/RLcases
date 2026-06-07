import numpy as np
import tensorflow as tf
from TFparts import TFnetwork#, kfac
#from tensorflow.python.client import device_lib
#print(len(device_lib.list_local_devices()))
def noise_and_argmax(logits):
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(noise)), 1)
def openai_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)
def softmax_entropy(p0):
    return - tf.reduce_sum(p0 * tf.log(p0 + 1e-6), axis=1)
def mse(predicted, ground_truth):
    return tf.square(predicted - ground_truth) / 2.
class Model:
    def __init__(self,obs_space,act_space,args):
        self.obs_space, self.act_space, self.args = obs_space, act_space, args
        tf.reset_default_graph()
        tfconfig = tf.ConfigProto()#device_count = {'GPU': 0})
        #tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
        tfconfig.gpu_options.allow_growth = True
        tfconfig.allow_soft_placement = True
        #tfconfig.log_device_placement=True
        self.sess = tf.Session(config=tfconfig)
        tf.set_random_seed(args.env_seed)
        self.actions    = tf.placeholder(tf.int32, [None])
        self.rewards    = tf.placeholder(tf.float32, [None])
        self.advantages = tf.placeholder(tf.float32, [None])
        self.learning_rate  = tf.placeholder(tf.float32, [])

        input_shape = [args.stack_num]+list(obs_space.shape)
        self.input = tf.placeholder(tf.float32, [None]+input_shape)#tf.placeholder(eval(args.input_dtype), [None]+input_shape)###
        #self.shaped_input = tf.transpose(self.input[:,:,:,:,0],[0,2,3,1])
        self.shaped_input = self.input#tf.cast(self.shaped_input, tf.float32) / 255.0
        self.train = tf.placeholder(tf.bool, name='phase')
        if args.aprxfunc == 'cnnmlp':
            apfparas = args.apfparas.split('=')
            cnncnnparas = apfparas[0].split('^')
            cnnmlpparas = apfparas[1].split('^')
            mlpmlpparas = apfparas[2].split('^')
            if len(obs_space.shape) == 3:
                convs = [[int(para) for para in paras.split(',')] for paras in cnncnnparas]
                denss = [                              int(paras) for paras in cnnmlpparas]
            elif len(obs_space.shape) == 1:
                convs = []
                denss = [                              int(paras) for paras in mlpmlpparas]
            if act_space.__class__.__name__ == "Discrete":
                self.ay_value, self.cy_value = TFnetwork.cnnmlp(self.shaped_input,act_space.n,self.train,convs,denss)
                self.action_s = noise_and_argmax(self.ay_value)
                self.maxact_s = tf.argmax(self.ay_value, 1)
                #dist = tf.distributions.Categorical(probs=tf.nn.softmax(policy))
                negative_log_prob_action  = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.ay_value, labels=self.actions)
            elif act_space.__class__.__name__ == "Box":
                self.ay_value, self.cy_value = TFnetwork.cnnmlp(self.shaped_input,act_space.shape[0],self.train,convs,denss)
                logstd = tf.get_variable(name='logstd', shape=[1, act_space.shape[0]], initializer=tf.zeros_initializer())
                std = tf.zeros_like(self.ay_value) + tf.exp(logstd)
                dist = tf.distributions.Normal(loc=self.ay_value, scale=std)
                self.action_s = step_dist.sample(1)[0]
                negative_log_prob_action = step_dist.log_prob(self.action_s)
                #self.action_s = dist.sample(self.ay_value)
                #self.maxact_s = tf.argmax(self.ay_value, 1)
        self.value_s  = self.cy_value[:, 0]

        self.policy_gradient_loss = tf.reduce_mean(self.advantages * negative_log_prob_action)
        self.value_function_loss  = tf.reduce_mean(mse(tf.squeeze(self.cy_value), self.rewards))
        self.entropy              = tf.reduce_mean(openai_entropy(self.ay_value))
        self.loss = self.policy_gradient_loss + self.value_function_loss*args.vlossratio - self.entropy*args.entropycoef
        with tf.variable_scope("policy"):
            params = tf.trainable_variables()
            grads  = tf.gradients(self.loss, params)
            grads, grad_norm = tf.clip_by_global_norm(grads, args.max_grad_norm)
            grads  = list(zip(grads, params))
            if args.opt=='RMSprop': self.optimize = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=args.alpha, epsilon=args.eps).apply_gradients(grads)
            if args.opt=='Adam':    self.optimize = tf.train.AdamOptimizer(learning_rate=self.learning_rate).apply_gradients(grads)
        decayparas = args.decayparas.split(',') # eta_min_ratio, T/2, gamma
        if args.decay=='linear':
            self.eta_min_ratio = float(decayparas[0])
        if args.decay=='exp':
            self.exp_gamma = float(decayparas[2])
        if args.decay=='cos':
            self.eta_min_ratio = float(decayparas[0])
            self.T_2 = int(decayparas[1])
        self.saver, self.checkpoint_dir = tf.train.Saver(max_to_keep=10), args.checkpoint_dir
        self.sess.run(tf.global_variables_initializer())
        args.numparas = int(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    def get_action(self, s, explore):
        if explore: return self.sess.run(self.action_s, feed_dict={self.input: s, self.train: False}), None
        else:       return self.sess.run(self.maxact_s, feed_dict={self.input: s, self.train: False}), None
    def getvalue(self, s):
        return self.sess.run(self.value_s,  feed_dict={self.input: s, self.train: False})
    def __discount_with_dones(self, rewards, dones):
        discounted, r = [], 0 # Start from downwards to upwards like Bellman backup operation.
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + self.args.gamma * r * (1.0 - done)
            discounted.append(r)
        return discounted[::-1]
    def update(self, mb_obs_stack, mb_act, mb_new_stack, mb_rew, mb_done, info_es, info_ps, crt_step, max_step):
        if self.args.decay=='linear':
            lr = self.args.lr*((1-self.eta_min_ratio)*(1-crt_step/max_step)+self.eta_min_ratio)
        if self.args.decay=='exp':
            lr = self.args.lr*pow(self.exp_gamma,crt_step+1)
        if self.args.decay=='cos':
            lr = self.args.lr*((1-self.eta_min_ratio)/2*np.cos((crt_step+1)/self.T_2*np.pi)+(1+self.eta_min_ratio)/2)
        mb_val = []
        for i in range(mb_obs_stack.shape[0]):
            value = self.getvalue(mb_obs_stack[i])
            mb_val.append(value)
        mb_val = np.array(mb_val)
        # Conversion from (time_steps, num_envs) to (num_envs, time_steps)
        mb_obs_stack = np.asarray(mb_obs_stack).swapaxes(1, 0)
        mb_val       = np.asarray(mb_val      ).swapaxes(1, 0)
        mb_act       = np.asarray(mb_act      ).swapaxes(1, 0)
        mb_rew       = np.asarray(mb_rew      ).swapaxes(1, 0)
        mb_done      = np.asarray(mb_done     ).swapaxes(1, 0)
        last_values  = self.getvalue(mb_new_stack[-1]).tolist()
        # Discount/bootstrap off value fn in all parallel environments
        for n, (rewards, dones, value) in enumerate(zip(mb_rew, mb_done, last_values)):
            rewards, dones = rewards.tolist(), dones.tolist()
            if dones[-1] == 0: rewards = self.__discount_with_dones(rewards+[value], dones+[0])[:-1]
            else:              rewards = self.__discount_with_dones(rewards, dones)
            mb_rew[n] = rewards
        # Instead of (num_envs, time_steps). Make them num_envs*time_steps.
        obsshape     = list(mb_obs_stack.shape)
        mb_obs_stack = mb_obs_stack.reshape([obsshape[0]*obsshape[1]]+obsshape[2:])
        mb_val       = mb_val.flatten()
        mb_act       = mb_act.flatten()
        mb_rew       = mb_rew.flatten()
        mb_adv       = mb_rew - mb_val
        feed_dict = {self.input: mb_obs_stack, self.actions: mb_act, self.rewards: mb_rew, self.advantages: mb_adv, self.learning_rate: lr, self.train: True}
        self.sess.run([self.optimize], feed_dict=feed_dict)
    def save(self,name):
        #print('saving')#,int(name.split('_')[1]))
        self.saver.save(self.sess, self.checkpoint_dir, global_step=1)#int(name.split('_')[1]))
    def load(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint: self.saver.restore(self.sess, latest_checkpoint)
