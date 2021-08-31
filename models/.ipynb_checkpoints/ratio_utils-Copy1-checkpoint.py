import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
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


tol = 1e-35
bs = 500
K = 3
do = 0.8


def reset(seed=40):
    tf.reset_default_graph()
    tf.random.set_random_seed(seed)

    
def ratios_critic(x, prob = 1, K=3, deep=False):
    with tf.variable_scope('critic', reuse=tf.AUTO_REUSE) as scope:
        x = tf.expand_dims(x,1)
        h=x
        h = slim.fully_connected(h, 100, activation_fn=tf.nn.softplus)
        h = tf.nn.dropout(h,prob)
        h = slim.fully_connected(h, 50, activation_fn=tf.nn.softplus)
        h = tf.nn.dropout(h,prob)
        h = slim.fully_connected(h, 50, activation_fn=tf.nn.softplus)
        h = tf.nn.dropout(h,prob)
        h = slim.fully_connected(h, 50, activation_fn=tf.nn.softplus)
        h = tf.nn.dropout(h,prob)
        
        if deep:
            l1 = slim.fully_connected(h, 50, activation_fn=None) #, biases_initializer=None
            l2 = slim.fully_connected(h, 50, activation_fn=None) 
            l3 = slim.fully_connected(h, 50, activation_fn=None) 
            
        log_d = slim.fully_connected(h, K, activation_fn=None)
#         log_d = tf.concat([tf.reduce_sum(l1*l1,1,keep_dims=True),tf.reduce_sum(l2*l2,1,keep_dims=True),tf.reduce_sum(l3*l3,1,keep_dims=True)],1) #slim.fully_connected(h*h, K, activation_fn=None) + l
        print(log_d)
    return tf.squeeze(log_d)

def get_data(mu_1=0.,mu_2=2.,mu_3=2.,scale_p=0.1,scale_q=0.1,scale_m=1.,mtype='cauchy'):
    p = tfd.Normal(loc=mu_1, scale=scale_p)
    q = tfp.distributions.StudentT(df=10., loc=mu_2, scale=scale_q)
#     q = tfd.Normal(loc=mu_2, scale=scale_q)
    if mtype=='cauchy':
        m = tfp.distributions.Cauchy(loc=mu_3, scale=scale_m)
    elif mtype=='student':
        m = tfp.distributions.StudentT(df=1., loc=mu_3, scale=scale_m)
    else:
        m = tfp.distributions.Normal(loc=mu_3, scale=scale_m)
        
    p_samples = p.sample([bs]) 
    q_samples = q.sample([bs])
    m_samples = m.sample([bs])
    
    return p, q, m, p_samples, q_samples, m_samples, m

def get_gt_ratio_kl(p,q,samples):
    ratio = p.log_prob(samples) - q.log_prob(samples)
    kl = tf.reduce_mean(ratio)
    return ratio, kl

def get_logits(samples, do=1., deep=False):
    return ratios_critic(samples,do,deep=deep)

def get_loss(p_samples,q_samples,m_samples,m_dist=None,do=0.8, deep=False):
    
    logP = get_logits(p_samples,do,deep=deep)
    logQ = get_logits(q_samples,do,deep=deep)
    logM = get_logits(m_samples,do,deep=deep)
    
    a = np.tile([1,0,0],bs)
    b = np.tile([0,1,0],bs)
    c = np.tile([0,0,1],bs)

    label_a = tf.reshape(a,[bs,K])
    label_b = tf.reshape(b,[bs,K])
    label_c = tf.reshape(c,[bs,K])

    disc_loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logP, labels=label_a))
    disc_loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logQ, labels=label_b))
    disc_loss_3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logM, labels=label_c))
    
    loss = disc_loss_1 + disc_loss_2 + disc_loss_3
    
    if m_dist != None:
        loss += 1e-4*tf.reduce_mean(m_dist.log_prob(m_samples) - logM[:,2])
    return loss

def get_optim(loss, lr=0.001, b1=0.001, b2=0.999):
    t_vars = tf.trainable_variables()
    c_vars = [var for var in t_vars if 'critic' in var.name]
    optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2=b2).minimize(loss, var_list=t_vars)
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

    
    
def sample_and_plot(sess, kld, p_samples, q_samples, m_samples, log_ratio_p_q, log_ratio_p_m, mu_1, mu_2, scale_p, scale_q, mu_3, scale_m):
    kl_ratio_store=[]
    log_ratio_store=[]
    log_r_p_from_m_direct_store=[]


    feed_dict = {}
    kl_ratio, p_s, q_s, m_s, lpq, lpq_from_cob_dre_direct = sess.run([kld, p_samples, q_samples, m_samples,
                                                                                log_ratio_p_q,  log_ratio_p_m],
                                                                              feed_dict=feed_dict)
    kl_ratio_store.append(kl_ratio)
    log_ratio_store.append(lpq)
    log_r_p_from_m_direct_store.append(lpq_from_cob_dre_direct)
    
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
    ax2.set_xlim([-4,6])
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
    ax4.set_xlim([-4,6])
    ax4.set_ylim([-1000,1000])
    
    
    rat = log_r_p_from_m_direct_store[-1][:,0]-log_r_p_from_m_direct_store[-1][:,1]
    d = [np.squeeze(norm.logpdf(x,mu_2,scale_q)) for x in xs]
    b = [np.squeeze(norm.logpdf(x,mu_1,scale_p)) for x in xs]
    ax3.scatter(xs,b,label='True P',alpha=0.9,s=5.)
    ax3.scatter(xs,rat+d,label='P',alpha=0.9,s=5.)

    ax3.set_xlabel("Samples")
    ax3.set_ylabel("Log P(x)")
    ax3.legend(loc='best')
    ax3.set_xlim([-4.,4])
    ax3.set_ylim([-600,400])
    
    