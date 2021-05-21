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
import pickle
import os
from scipy.linalg import cholesky
import math




tol = 1e-35
bs = 500
K = 3
do = 0.8


def reset(seed=40):
    tf.reset_default_graph()
    tf.random.set_random_seed(seed)
    

def ratios_critic(x, prob = 1, K=3, deep=False):
    with tf.variable_scope('critic', reuse=tf.AUTO_REUSE) as scope:
        
        q1 = tf.get_variable('q1',1.)
        q2 = tf.get_variable('q2',1.)
        q3 = tf.get_variable('q3',1.)
        
        q4 = tf.get_variable('q4',1.)
        q5 = tf.get_variable('q5',1.)
        q6 = tf.get_variable('q6',1.)
        
        b1 = tf.get_variable('b1',1.)
        b2 = tf.get_variable('b2',1.)
        b3 = tf.get_variable('b3',1.)
        
        s1 = tf.get_variable('s1',1.)
        s2 = tf.get_variable('s2',1.)
        s3 = tf.get_variable('s3',1.)
        
        t1 = tf.get_variable('t1',1.)
        t2 = tf.get_variable('t2',1.)
        t3 = tf.get_variable('t3',1.)

#         h1 = 1e12*(x-q1)*(x-q1)*s1 + (x-q4)*t1 + b1 
        h1 = (x-q1)*(x-q1)*s1 + (x-q4)*t1 + b1 
        h2 = (x-q2)*(x-q2)*s2 + (x-q5)*t2 + b2
        h3 = t3*(x-q6) + b3# t3*(x-q6) + b3 #(x-q3)*(x-q3)*s3 + t3*(x-q6) + b3
#         h3 = slim.fully_connected(tf.concat([h1,h2],1), 1, activation_fn=tf.nn.softplus)
#         h3 = slim.fully_connected(h3, 1, activation_fn=None)
        
        
#         h1 = tf.matmul(x,tf.matmul(q1,q1,transpose_b=True))
#         h1 = tf.reduce_sum(x*h1,-1, keep_dims=True) + b1
        
#         h2 = tf.matmul(x,tf.matmul(q2,q2,transpose_b=True))
#         h2 = tf.reduce_sum(x*h2,-1, keep_dims=True) + b2
        
#         h3 = tf.matmul(x,tf.matmul(q3,q3,transpose_b=True))
#         h3 = tf.reduce_sum(x*h3,-1, keep_dims=True) + b3
        
        logits = tf.concat([h1,h2,h3],1)
        
        return logits
    
# def ratios_critic(x, prob = 1, K=3, deep=False):
#     with tf.variable_scope('critic', reuse=tf.AUTO_REUSE) as scope:
        
#         h = slim.fully_connected(x, 100, activation_fn=tf.nn.softplus)
#         h = tf.nn.dropout(h,prob)
        
#         h = slim.fully_connected(h, 50, activation_fn=tf.nn.softplus)
#         h = tf.nn.dropout(h,prob)
        
#         return slim.fully_connected(h, K, activation_fn=None, biases_initializer = tf.constant_initializer(0))


# def get_data(mu_1=0.,mu_2=2.,mu_3=2.,scale_p=0.1,scale_q=0.1,scale_m=1.,mtype='cauchy'):
#     p = tfd.Normal(loc=mu_1, scale=scale_p)
#     q = tfd.Normal(loc=mu_2, scale=scale_q) #tfp.distributions.StudentT(df=1., loc=mu_2, scale=scale_q)#t
#     if mtype=='cauchy':
#         m = tfp.distributions.Cauchy(loc=mu_3, scale=scale_m)
#     if mtype=='cauchy_mix':
#         mix = 0.3
#         m = tfp.distributions.Mixture(
#           cat=tfp.distributions.Categorical(probs=[.6,.4]),
#           components=[
#             p,
#             q
#         ])
#     elif mtype=='student':
#         m = tfp.distributions.StudentT(df=1., loc=mu_3, scale=scale_m)
#     else:
#         m = tfp.distributions.Normal(loc=mu_3, scale=scale_m)
        
#     p_samples = p.sample([bs]) 
#     q_samples = q.sample([bs])
#     alpha = tfd.Uniform (0.,1.).sample([bs])
#     m_samples = m.sample([bs]) #tf.sqrt(1-alpha*alpha)*p_samples + alpha*q_samples + m.sample([bs])
    
#     return p, q, m, p_samples, q_samples, m_samples, m


def get_gt_ratio_kl(p,q,samples):
    ratio = p.log_prob(samples) - q.log_prob(samples)
    kl = tf.reduce_mean(ratio)
    try:
        true_kl = p.kl_divergence(q)
    except:
        true_kl = "-1"
    
    return ratio, kl, true_kl

def get_logits(samples, do=1., deep=False, training=True):
    samples = tf.expand_dims(samples,1)
    return ratios_critic(samples, do, deep=deep)

def get_kl_from_cob(samples):
    log_rat = get_logits(samples)
    return tf.reduce_mean(log_rat[:,0]-log_rat[:,1])

def get_loss(p_samples,q_samples,m_samples, m_dist=None,do=0.8, deep=False):
    
    logP = get_logits(p_samples,  do, deep=deep)
    logQ = get_logits(q_samples,  do, deep=deep)
    logM = get_logits(m_samples,  do, deep=deep)
    
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
        loss += 1e-5*tf.reduce_mean(m_dist.log_prob(m_samples) - logM[:,2])
    return loss

def get_optim(loss, lr=0.001, b1=0.001, b2=0.999):
    t_vars = tf.trainable_variables()
    c_vars = [var for var in t_vars if 'critic' in var.name]
#     optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2=b2).minimize(loss, var_list=t_vars)
    optim = tf.train.AdamOptimizer(lr).minimize(loss, var_list=t_vars)
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
    kl_true, kl_cob, p_s, q_s, m_s, lpq, lpq_from_cob_dre_direct = sess.run([kl_from_pq, kl_from_cob, p_samples, q_samples, m_samples,
                                                                                log_ratio_p_q,  log_ratio_p_m],
                                                                              feed_dict=feed_dict)
    
    
    
    '''Save ratio estimates'''
    data_dir = "../data/sym/"+str(scale_p)+"-"+str(scale_q)+str(scale_m)+"/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
#     f = open(data_dir+"KLD"+".txt", "a")
#     f.write("GT for mu_3 = "+str(mu_3)+": "+str(kl_ratio)+"\nGT-est: "+str(kl_true)+"\nCoB: "+str(kl_cob)+"\n----------\n")
#     f.close()
    log_ratio_store.append(lpq)
    log_r_p_from_m_direct_store.append(lpq_from_cob_dre_direct)
    
    pickle.dump(log_r_p_from_m_direct_store, open(data_dir+"log_r_p_from_m_direct_store"+str(mu_3)+".p", "wb"))
    pickle.dump(m_s, open(data_dir+"xs"+str(mu_3)+".p", "wb"))
#     pickle.dump(log_ratio_store, open(data_dir+"log_ratio_store"+str(mu_3)+".p", "wb"))
    
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
    
#     plt.savefig(data_dir+str(mu_3)+".jpg")
    
    
    