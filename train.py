import numpy as np
import sugartensor as tf
from data_noise import SpeechCorpus, voca_size
from model import *


# set log level to debug
tf.sg_verbosity(10)


#
# hyper parameters
#

batch_size = 16    # total batch size

#
# inputs
#

# corpus input tensor
data = SpeechCorpus(batch_size=batch_size * tf.sg_gpus())

# mfcc feature of audio
inputs = tf.split(data.mfcc, tf.sg_gpus(), axis=0)
# mfcc_noise feature of audio
inputs_noise = tf.split(data.mfcc_noise, tf.sg_gpus(), axis=0)
# target sentence label
labels = tf.split(data.label, tf.sg_gpus(), axis=0)

# sequence length except zero-padding
seq_len = []
for input_ in inputs:
    seq_len.append(tf.not_equal(input_.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1))



def penalize_loss(gamma, lambd, tensor, tensor_n):

	#gamma * (vector-vector_d)**2 - lamada * (vector dot vector_d)/(nor(vector)*nor(vector_d))

	with tf.sg_context(name='penalize'):

		square = tf.reduce_sum(tf.reduce_sum(tf.square(tensor-tensor_n), 2), 1)

		cosine = tf.reduce_sum(tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(tensor, 2), tf.nn.l2_normalize(tensor_n, 2)), 2) ,1)

		return gamma * square - lambd * cosine


# parallel loss tower
@tf.sg_parallel
def get_loss(opt):

    # encode audio feature
    with tf.variable_scope("model"):
        logit_clean = get_logit(opt.input[opt.gpu_index], voca_size=voca_size)
        loss_clean = logit_clean.sg_ctc(target=opt.target[opt.gpu_index], seq_len=opt.seq_len[opt.gpu_index])
    with tf.variable_scope("model", reuse=True):    
        logit_noise = get_logit(opt.input_noise[opt.gpu_index], voca_size=voca_size)
        loss_noise = logit_noise.sg_ctc(target=opt.target[opt.gpu_index], seq_len=opt.seq_len[opt.gpu_index])
    # CTC loss
    loss_penalize = penalize_loss(opt.gamma, opt.lambd, logit_clean, logit_noise)
    
    return loss_clean + opt.alpha * loss_noise + loss_penalize
#
# train
#
tf.sg_train(lr=0.0001, loss=get_loss(input=inputs, input_noise=inputs_noise, target=labels, seq_len=seq_len, 
            alpha= 1, gamma= 0.01, lambd= 0.01),
            ep_size=data.num_batch, max_ep=50)

