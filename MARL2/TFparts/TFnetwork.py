import tensorflow as tf
import numpy as np
from TFparts.TFlayer import conv2d, flatten, dense
def orthogonal_initializer(scale=1.0):# Orthogonal Initializer that uses SVD. The unused variables are just for passing in tensorflow
    def _ortho_init(shape, dtype, partition_info=None):
        shape = tuple(shape)
        if len(shape) == 2:   flat_shape = shape
        elif len(shape) == 4: flat_shape = (np.prod(shape[:-1]), shape[-1])# assumes NHWC
        else:                 raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init
def cnnmlp(shaped_input,output_num,phase,convs,denss):
    selfconv = shaped_input
    for i,conv in enumerate(convs):
        selfconv = conv2d('conv'+str(i), selfconv, num_filters=conv[4], kernel_size=(conv[0], conv[1]), padding='VALID', stride=(conv[2], conv[3]),
                initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu, is_training=phase)
    conv_flattened = flatten(selfconv)
    selfdens = conv_flattened
    for i,dens in enumerate(denss):
        selfdens = dense('dens'+str(i), selfdens, output_dim=dens,
                initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu, is_training=phase)
    output = dense('logits',selfdens, output_dim=output_num, initializer=orthogonal_initializer(np.sqrt(1.0)), is_training=phase)
    output1= dense('value', selfdens, output_dim=1         , initializer=orthogonal_initializer(np.sqrt(1.0)), is_training=phase)
    return output, output1

def resnet_layer(x, num_filters=16, kernel_size=(3,3), strides=(1,1), batch_normalization=False, activation=tf.nn.relu, conv_first=True, is_training=False):
    initializer = None#tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32) #he_init normal?     #regularizer = tf.nn.l2_loss()
    if conv_first:
        x = tf.layers.conv2d(inputs=x, filters=num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=initializer)
        if batch_normalization:    x = tf.layers.batch_normalization(inputs=x,scale=False,training=is_training)
        if activation is not None: x = activation(x)
    else:
        if batch_normalization:    x = tf.layers.batch_normalization(inputs=x,scale=False,training=is_training)
        if activation is not None: x = activation(x)
        x = tf.layers.conv2d(inputs=x, filters=num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=initializer)
    return x
class RESs:
    def __init__(self,env,args):
        self.batch_num, self.stack_num = None, int(args.hypers_model[0]) #args.batch_num, args.stack_num *env.env_num
        self.input_shape, self.output_num = [self.stack_num]+list(env.observation_space.shape), env.action_space.n
        self.input = tf.placeholder(tf.float32, [self.batch_num]+self.input_shape, name='input') #uint8
        self.actions    = tf.placeholder(tf.int32, [None])
        self.rewards    = tf.placeholder(tf.float32, [None])
        self.advantages = tf.placeholder(tf.float32, [None])
        self.learning_rate  = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, name='phase')
        self.shaped_input = tf.transpose(self.input[:,:,:,:,0],[0,2,3,1])
        self.shaped_input = tf.cast(self.shaped_input, tf.float32) / 255.0
        self.convinfos = [[int(num)for num in paras.split('-')] for paras in args.hypers_model[1].split('_')]
        self.densinfos = [int(num) for num in args.hypers_model[2].split('_')]
        x = self.shaped_input# conv layers
        pool_size = (self.input_shape[1], self.input_shape[2])
        for istack,stackinfos in enumerate(self.convinfos):
            kernel_size, num_filters, strides, num_res_blocks = (stackinfos[0], stackinfos[1]), stackinfos[2], (stackinfos[3], stackinfos[4]), stackinfos[5]
            for iblock in range(num_res_blocks):
                if iblock == 0:
                    y = resnet_layer(x=x, num_filters=num_filters, kernel_size=kernel_size, strides=strides)
                    y = resnet_layer(x=y, num_filters=num_filters, kernel_size=kernel_size, strides=(1,1),   activation=None)
                    x = resnet_layer(x=x, num_filters=num_filters, kernel_size=(1,1),       strides=strides, activation=None)
                else:
                    y = resnet_layer(x=x, num_filters=num_filters, kernel_size=kernel_size, strides=(1,1))
                    y = resnet_layer(x=y, num_filters=num_filters, kernel_size=kernel_size, strides=(1,1),   activation=None)
                x = tf.add(x,y)
                x = tf.nn.relu(x)
            pool_size = (math.ceil(pool_size[0]/strides[0]),math.ceil(pool_size[1]/strides[1]))
        print(pool_size)
        x = tf.layers.AveragePooling2D(pool_size=pool_size, strides=(1,1))(x)
        self.conv_flattened = tf.layers.flatten(x)
        h_value = self.conv_flattened # dense layers
        initializer = None#orthogonal_initializer(np.sqrt(2))#glorot_uniform_initializer(default)
        for i,units in enumerate(self.densinfos):
            if i==0:
                h_value = tf.layers.dense(inputs=h_value, units=units, activation=None, kernel_initializer=initializer)
                #h_value = tf.layers.batch_normalization(inputs=h_value,scale=False,training=True)
                h_value = tf.nn.relu(h_value)
            h_valueo= tf.identity(h_value)
            h_value = tf.layers.dense(inputs=h_value, units=units, activation=None, kernel_initializer=initializer)
            #h_value = tf.layers.batch_normalization(inputs=h_value,scale=False,training=True)
            h_value = tf.nn.relu(h_value)
            h_value = tf.layers.dense(inputs=h_value, units=units, activation=None, kernel_initializer=initializer)
            #h_value = tf.layers.batch_normalization(inputs=h_value,scale=False,training=True)
            h_value = h_value + h_valueo
            h_value = tf.nn.relu(h_value)
        self.y_value = h_value
        initializer = None#orthogonal_initializer(np.sqrt(1.0))#tf.truncated_normal_initializer # output layers
        self.ay_value = tf.layers.dense(inputs=self.y_value, units=self.output_num, activation=None, kernel_initializer=initializer)
        self.cy_value = tf.layers.dense(inputs=self.y_value, units=1              , activation=None, kernel_initializer=initializer)
        ###########################################################################
        #self.target_actor     = tf.placeholder(tf.float32, [self.batch_num]+list([self.output_num]), name='target_actor')
        #self.loss_actor       = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.ay_value, labels=self.target_actor))
        #self.target_critic    = tf.placeholder(tf.float32, [self.batch_num]+list([1]), name='target_critic')
        #self.loss_critic      = tf.reduce_mean(tf.square(self.target_critic - self.cy_value))
        #self.loss = self.loss_actor + self.loss_critic*self.ratio #- self.entropy*0.01 ######
        #self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
