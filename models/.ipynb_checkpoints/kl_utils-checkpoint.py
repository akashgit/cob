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


tol = 1e+6
bs = 500
K = 3
do = 0.8
n_posterior_samples = bs 

prior = tfp.layers.default_mean_field_normal_fn(
    is_singular=False, loc_initializer=tf.initializers.random_normal(stddev=5.),
    untransformed_scale_initializer=tf.initializers.random_normal(mean=0.0,
    stddev=5.), loc_regularizer=None, untransformed_scale_regularizer=None,
    loc_constraint=None, untransformed_scale_constraint=None
)

layer = tfp.layers.DenseFlipout(K, activation=None) #, kernel_prior_fn=prior)

def reset(seed=40): 
    tf.reset_default_graph()
    tf.random.set_random_seed(seed)

    
# def ratios_critic(x, prob = 1):
#     with tf.variable_scope('critic', reuse=tf.AUTO_REUSE) as scope:
#         x = tf.expand_dims(x,1)
#         h = slim.fully_connected(x, 100, activation_fn=tf.nn.softplus)
#         h = tf.nn.dropout(h,prob)
#         h = slim.fully_connected(h, 50, activation_fn=tf.nn.softplus)
#         h = tf.nn.dropout(h,prob)
#     return h#tf.expand_dims(h,0)

def ratios_critic(x, prob = 1, K=3, deep=False):
    with tf.variable_scope('critic', reuse=tf.AUTO_REUSE) as scope:
        x = tf.expand_dims(x,1)
        q1 = tf.get_variable('q1',1.)
        q2 = tf.get_variable('q2',1.)
        q3 = tf.get_variable('q3',1.)
        
        q4 = tf.get_variable('q4',1.)
        q5 = tf.get_variable('q5',1.)
        
        b1 = tf.get_variable('b1',1.)
        b2 = tf.get_variable('b2',1.)
        b3 = tf.get_variable('b3',1.)
        
        s1 = tf.get_variable('s1',1.)
        s2 = tf.get_variable('s2',1.)
        s3 = tf.get_variable('s3',1.)
        
        t1 = tf.get_variable('t1',1.)
        t2 = tf.get_variable('t2',1.)
        t3 = tf.get_variable('t3',1.)

        h1 = (x-q1)*(x-q1)*s1 + (x-q2)*t1 + b1 
        h2 = (x-q3)*(x-q3)*s2 + (x-q4)*t2 + b2
        h3 = t3*(x-q5) + b3
        
        logits = tf.concat([h1,h2,h3],1)
        
        return logits#, [q1,q2,q4,q5,q6,s1,s2,t1,t2,t3,b1,b2,b3]



def get_data(mu_1=0.,mu_2=2.,mu_3=2.,scale_p=0.1,scale_q=0.1,scale_m=1.,mtype='cauchy'):
    p = tfd.Normal(loc=mu_1, scale=scale_p)
    q = tfd.Normal(loc=mu_2, scale=scale_q)
    if mtype=='cauchy':
        m = tfp.distributions.Cauchy(loc=mu_3, scale=scale_m)
    elif mtype=='student':
        m = tfp.distributions.StudentT(df=1., loc=mu_3, scale=scale_m)
    else:
        m = tfp.distributions.Normal(loc=mu_3, scale=scale_m)
        
    p_samples = p.sample([bs]) 
    q_samples = q.sample([bs])
    m_samples = m.sample([bs])
    
    return p, q, m, p_samples, q_samples, m_samples

def get_gt_ratio_kl(p,q,samples):
    ratio = p.log_prob(samples) - q.log_prob(samples)
    kl = tf.reduce_mean(ratio)
    return ratio, kl, p.kl_divergence(q)

def get_logits(samples, do=1.):
    return layer(ratios_critic(samples,do))

def get_kl_from_cob(samples):
    log_rat = get_logits(samples)
    return tf.reduce_mean(log_rat[:,0]-log_rat[:,1])

def get_loss(p_samples,q_samples,m_samples,do=0.8):
    
    logP = get_logits(p_samples,do)
    logQ = get_logits(q_samples,do)
    logM = get_logits(m_samples,do)
    
    a = np.tile([1,0,0],bs)
    b = np.tile([0,1,0],bs)
    c = np.tile([0,0,1],bs)

    label_a = tf.reshape(a,[bs,K])
    label_b = tf.reshape(b,[bs,K])
    label_c = tf.reshape(c,[bs,K])

    disc_loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logP, labels=label_a))
    disc_loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logQ, labels=label_b))
    disc_loss_3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logM, labels=label_c))
    
    dloss = disc_loss_1 + disc_loss_2 + disc_loss_3
    kl_loss = sum(layer.losses)/tol
    
    return  dloss+kl_loss

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


# helper function, mostly for plotting
def log_ratio_predictive(x_, w, b):
    pred = x_@w 
    pred = pred[0,:,:] - pred[1,:,:]
    return pred, pred.mean(1), pred.std(1)
    
def sample_and_plot(sess, kl_p_q, kl_cob, kld, p_samples, q_samples, m_samples, log_ratio_p_q, log_ratio_p_m, mu_1, mu_2, scale_p, scale_q, mu_3, scale_m, training=True):
    kl_ratio_store=[]
    log_ratio_store=[]
    log_r_p_from_m_direct_store=[]


    feed_dict = {}
    encoder_m = ratios_critic(m_samples)
    encoder_p = ratios_critic(p_samples)
    kl_ratio, kl_cob, kl_true, p_s, q_s, xs, enc_m, enc_p, lpq, lpq_from_cob_dre_direct= sess.run([kl_p_q, kl_cob, kld,
                                                                            p_samples, q_samples, m_samples, encoder_m, encoder_p,
                                                                            log_ratio_p_q,  log_ratio_p_m],
                                                                          feed_dict=feed_dict)
    
    
    log_ratio_store.append(lpq)
    log_r_p_from_m_direct_store.append(lpq_from_cob_dre_direct)
    
    fig, [ax1,ax2,ax3, ax4] = plt.subplots(1, 4,figsize=(13,4))
    ax1.hist(p_s, density=True, histtype='stepfilled', alpha=0.8, label='P')
    ax1.hist(q_s, density=True, histtype='stepfilled', alpha=0.8, label='Q')
    ax1.hist(xs, density=True, histtype='stepfilled', alpha=0.8, label='M')
    ax1.legend(loc='best', frameon=False)
    ax1.set_xlim([mu_1-1,mu_2+1])
    
    ax2.scatter(xs,log_ratio_store[0],label='True p/q',alpha=0.9,s=10.,c='b')
    ax2.scatter(xs,log_r_p_from_m_direct_store[-1][:,0]-log_r_p_from_m_direct_store[-1][:,1],label='CoB p/q',alpha=0.9,s=10.,c='r')
    ax2.scatter(xs,-log_ratio_store[0],label='True q/p',alpha=0.9,s=10.,c='b')
    ax2.scatter(xs,log_r_p_from_m_direct_store[-1][:,1]-log_r_p_from_m_direct_store[-1][:,0],label='CoB q/p',alpha=0.9,s=10.,c='r')

    ax2.set_xlabel("Samples")
    ax2.set_ylabel("Log Ratio")
    ax2.legend(loc='best')
    ax2.set_xlim([mu_1-1,mu_2+1])
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
    ax4.set_xlim([mu_1-1,mu_2+1])
    ax4.set_ylim([-1000,1000])
    
    
    rat = log_r_p_from_m_direct_store[-1][:,0]-log_r_p_from_m_direct_store[-1][:,1]
    d = [np.squeeze(norm.logpdf(x,mu_2,scale_q)) for x in xs]
    b = [np.squeeze(norm.logpdf(x,mu_1,scale_p)) for x in xs]
    ax3.scatter(xs,b,label='True P',alpha=0.9,s=5.)
    ax3.scatter(xs,rat+d,label='P',alpha=0.9,s=5.)

    ax3.set_xlabel("Samples")
    ax3.set_ylabel("Log P(x)")
    ax3.legend(loc='best')
    ax3.set_xlim([mu_1-1,mu_2+1])
    ax3.set_ylim([-600,400])

    plt.show()
    
    print('KL : ',kl_true)
    print('KL from samples : ',kl_ratio)
    print('KL from CoB: ', kl_cob) 
    
    
    n_posterior_samples
    
    candidate_ws = []
    candidate_bs = []
    
    if training:
        print(f"Taking {n_posterior_samples} samples from posterior distributions on weights\n")
        w_draw = layer.kernel_posterior.sample()
    else:
        print(f"Taking {n_posterior_samples} samples from prior distributions on weights\n")
        w_draw = layer.kernel_prior.sample()
    b_draw = layer.bias_posterior.sample()

    for mc in range(n_posterior_samples):
        w_, b_ = sess.run([w_draw, b_draw])
        candidate_ws.append(w_)
        candidate_bs.append(b_)


    candidate_ws = np.array(candidate_ws).astype(np.float32)
    candidate_bs = np.array(candidate_bs).astype(np.float32)

    post, post_pred,post_std = log_ratio_predictive(enc_m, candidate_ws.T, candidate_bs.T)
    kl_post,_,_ = log_ratio_predictive(enc_p, candidate_ws.T, candidate_bs.T)

    x_sorted = []
    m_sorted = []
    s_sorted = []
    p_sorted = []
    [(x_sorted.append(a),m_sorted.append(b), s_sorted.append(c), p_sorted.append(d)) for a,b,c,d in sorted(zip(xs,post_pred, post_std, post))]
    
    fig1, ax = plt.subplots()
    plt.plot(x_sorted, norm.logpdf(x_sorted,mu_1,scale_p)-norm.logpdf(x_sorted,mu_2,scale_q),label='True log_prob',c='r')
#     plt.plot(x_sorted, m_sorted,label='BC_Bayes_e',alpha=0.7)
    quan = [np.abs(np.quantile(p_, .05)-np.quantile(p_, .95)) for p_ in p_sorted]
    plt.errorbar(x_sorted, m_sorted, yerr = quan,label='BC_Bayes_e')
    
#     plt.scatter(x_sorted,norm.logpdf(x_sorted,mu_1,scale_p)-norm.logpdf(x_sorted,mu_2,scale_q),label='True log_prob',alpha=0.99,s=5.)
    
    plt.xlabel("Samples")
    plt.ylabel("Log Ratio")
    plt.legend(loc='upper right')
    plt.xlim(mu_1-1,mu_2+1)
    plt.ylim(-400,1000)
    plt.show()
    
    fig1, ax = plt.subplots(figsize=(30,4))
    plt.plot(norm.logpdf(x_sorted[100:400],mu_1,scale_p)-norm.logpdf(x_sorted[100:400],mu_2,scale_q),label='True log_prob',alpha=0.99)
    plt.plot(m_sorted[100:400],label='BC_Bayes_e',alpha=0.7)
    plt.boxplot(p_sorted[100:400],widths=0.05,notch=True,labels=x_sorted[100:400], showfliers=False, showbox=False, showcaps=False)
    plt.xlabel("Samples")
    plt.ylabel("Log Ratio")
    plt.legend(loc='upper right')
    plt.xticks(rotation = -65)
    plt.locator_params(axis='x', nbins=100)
    plt.ylim(-400,1000)
    
    plt.show()
    
    fig1, ax = plt.subplots()
    print(kl_post.shape)
    plt.boxplot(kl_post.mean(0),widths=0.5,notch=True, showfliers=False)
#     plt.boxplot(kl_post.mean(0),widths=0.5,notch=True, showfliers=False)
    plt.ylabel("KLD")

    