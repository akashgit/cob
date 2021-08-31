import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import tensorflow.keras.initializers as initializers
from tensorflow.keras import regularizers
import numpy as np
import seaborn as sns; 
import tensorflow_probability as tfp
tfd = tfp.distributions
from scipy.stats import norm, uniform, cauchy
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
slim = tf.contrib.slim
from tqdm.notebook import tqdm
from time import sleep
from IPython.display import display, clear_output
import pickle
import os

tf.keras.backend.set_floatx('float32')


tol = 1e-35
bs = 1000
K = 3
do = 0.8
n_dims=320




def reset(seed=40):
    tf.reset_default_graph()
    tf.random.set_random_seed(seed)
    
    
    
def ratios_critic(h, prob = 1, K=3, deep=False):
    with tf.variable_scope('critic', reuse=tf.AUTO_REUSE) as scope:
        init = tf.keras.initializers.normal(stddev=2.)
        
        regularizer = tf.contrib.layers.l2_regularizer(1.)
        
        h = slim.fully_connected(h, 500, activation_fn=tf.nn.softplus, weights_regularizer=slim.l2_regularizer(.0)) #change back to 50
        h = slim.fully_connected(h, 100, activation_fn=tf.nn.softplus, weights_regularizer=slim.l2_regularizer(.0)) #change back to 10
        
        return slim.fully_connected(h, K, activation_fn=None, weights_regularizer=slim.l2_regularizer(.0))


def get_gt_ratio_kl(p,q,samples):
    ratio = p.log_prob(samples) - q.log_prob(samples)
    kl = tf.reduce_mean(ratio)
    return ratio, kl

def get_logits(samples,do=1., deep=False, training=True):
#     samples = tf.expand_dims(samples,1)
    return ratios_critic(samples,do,deep=deep)

def get_kl_from_cob(samples_p, samples_q):
    log_rat = get_logits(samples_p)
    V_p = log_rat[:,0]#-log_rat[:,1]
    return tf.reduce_mean(V_p)

# def get_kl_from_cob(samples_p, samples_q):
#     log_rat = get_logits(samples_p)
#     V_p = log_rat[:,0]-log_rat[:,1]
    
#     log_rat = get_logits(samples_q)
#     V_q = log_rat[:,0]-log_rat[:,1]
    
#     return 1 + tf.reduce_mean(V_p) - tf.reduce_mean(tf.exp(V_q))





def get_loss(p_samples,q_samples,m_samples,langrange,m_dist=None,do=0.8, deep=False):
    
    logP = get_logits(p_samples,do,deep=deep)
    logQ = get_logits(q_samples,do,deep=deep)
    logM = get_logits(m_samples,do,deep=deep)
    
    loss = -tf.reduce_mean(logP[:,0]) + (2+langrange)*tf.abs(tf.reduce_mean(tf.exp(logQ[:,0])-1))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    
    return loss #+ 0.001 * sum(reg_losses)

    
def get_optim(loss, lr=0.001, b1=0.001, b2=0.999):
    t_vars = tf.trainable_variables()
    print(t_vars)
    c_vars = [var for var in t_vars if 'critic' in var.name]
#     optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2=b2).minimize(loss, var_list=t_vars)
#     optim = tf.train.AdamOptimizer(lr).minimize(loss, var_list=t_vars)
    
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.compat.v1.train.cosine_decay(lr, global_step, 10000, alpha=5e-1, name=None)
    # Passing global_step to minimize() will increment it at each step.
    optim = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    return optim 

def train(sess, loss, optim, plotlosses, N=30000):

    pbar = range(0,N)
    for i in pbar:
    
        feed_dict = {}
        l,_ = sess.run([loss, optim],feed_dict=feed_dict)

        if i%1000==0:
            plotlosses.update({
                'Loss': l,
            })
            plotlosses.send()

    
    
def sample_and_plot(sess, kld, kl_from_pq, kl_from_cob, p_samples, q_samples, m_samples, log_ratio_p_q, log_ratio_p_m, mu_1, mu_2, scale_p, scale_q, mu_3, scale_m):
    kl_ratio_store=[]
    log_ratio_store=[]
    log_r_p_from_m_direct_store=[]


    feed_dict = {}
    kl_ratio, kl_true, kl_cob, p_s, q_s, m_s, lpq, lpq_from_cob_dre_direct = sess.run([kld, kl_from_pq, kl_from_cob, p_samples, q_samples, m_samples,
                                                                                log_ratio_p_q,  log_ratio_p_m],
                                                                              feed_dict=feed_dict)
    
    
    
    '''Save ratio estimates'''
    data_dir = "../data/sym/"+str(scale_p)+"-"+str(scale_q)+str(scale_m)+"/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    f = open(data_dir+"KLD"+".txt", "a")
    f.write("GT for mu_3 = "+str(mu_3)+": "+str(kl_ratio)+"\nGT-est: "+str(kl_true)+"\nCoB: "+str(kl_cob)+"\n----------\n")
    f.close()
    log_ratio_store.append(lpq)
    log_r_p_from_m_direct_store.append(lpq_from_cob_dre_direct)
    
    pickle.dump(log_r_p_from_m_direct_store, open(data_dir+"log_r_p_from_m_direct_store"+str(mu_3)+".p", "wb"))
    pickle.dump(m_s, open(data_dir+"xs"+str(mu_3)+".p", "wb"))
    pickle.dump(log_ratio_store, open(data_dir+"log_ratio_store"+str(mu_3)+".p", "wb"))
    
    xs = m_s

    fig, [ax1,ax2,ax3, ax4] = plt.subplots(1, 4,figsize=(13,4))
    ax1.hist(p_s, density=True, histtype='stepfilled', alpha=0.8, label='P')
    ax1.hist(q_s, density=True, histtype='stepfilled', alpha=0.8, label='Q')
    ax1.hist(m_s, density=True, histtype='stepfilled', alpha=0.8, label='M')
    ax1.legend(loc='best', frameon=False)
    ax1.set_xlim([-5,5])
    
    ax2.scatter(xs,log_ratio_store[0],label='True p/q',alpha=0.9,s=10.,c='b')
    ax2.scatter(xs,log_r_p_from_m_direct_store[-1][:,0]-log_r_p_from_m_direct_store[-1][:,1],label='CoB p/q',alpha=0.9,s=10.,c='r')
    ax2.scatter(xs,-log_ratio_store[0],label='True q/p',alpha=0.9,s=10.,c='b')
    ax2.scatter(xs,log_r_p_from_m_direct_store[-1][:,1]-log_r_p_from_m_direct_store[-1][:,0],label='CoB q/p',alpha=0.9,s=10.,c='r')

    ax2.set_xlabel("Samples")
    ax2.set_ylabel("Log Ratio")
    ax2.legend(loc='best')
    ax2.set_xlim([-6,10])
    ax2.set_ylim([-1000,1000])
    
    pm = [np.squeeze(norm.logpdf(x,mu_1,scale_p)-cauchy.logpdf(x,mu_3,scale_m)) for x in xs]
    qm = [np.squeeze(norm.logpdf(x,mu_2,scale_q)-cauchy.logpdf(x,mu_3,scale_m)) for x in xs]
    ax4.scatter(xs,pm,label='True p/m',alpha=0.9,s=10.,c='b')
    ax4.scatter(xs,log_r_p_from_m_direct_store[-1][:,0]-log_r_p_from_m_direct_store[-1][:,2],label='CoB p/m',alpha=0.9,s=10.,c='r')
    ax4.scatter(xs,qm,label='True q/m',alpha=0.9,s=10.,c='y')
    ax4.scatter(xs,log_r_p_from_m_direct_store[-1][:,1]-log_r_p_from_m_direct_store[-1][:,2],label='CoB q/m',alpha=0.9,s=10.,c='g')

    ax4.set_xlabel("Samples")
    ax4.set_ylabel("Log Ratio")
    ax4.legend(loc='best')
    ax4.set_xlim([-6,10])
    ax4.set_ylim([-1000,1000])
    
    
    rat = log_r_p_from_m_direct_store[-1][:,0]-log_r_p_from_m_direct_store[-1][:,1]
    d = [np.squeeze(norm.logpdf(x,mu_2,scale_q)) for x in xs]
    b = [np.squeeze(norm.logpdf(x,mu_1,scale_p)) for x in xs]
    ax3.scatter(xs,b,label='True P',alpha=0.9,s=5.)
    ax3.scatter(xs,rat+d,label='P',alpha=0.9,s=5.)

    ax3.set_xlabel("Samples")
    ax3.set_ylabel("Log P(x)")
    ax3.legend(loc='best')
    ax3.set_xlim([-6,10])
    ax3.set_ylim([-600,400])
    
    plt.savefig(data_dir+str(mu_3)+".jpg")
    
    
    