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

# def ratios_critic(x, prob = 1, K=3, deep=False):
#     with tf.variable_scope('critic', reuse=tf.AUTO_REUSE) as scope:
        
#         h = slim.fully_connected(x, 500, activation_fn=tf.nn.softplus) #change back to 50
#         h = tf.nn.dropout(h,prob)
        
#         h = slim.fully_connected(h, 100, activation_fn=tf.nn.softplus) #change back to 10
#         h = tf.nn.dropout(h,prob)
        
#         return slim.fully_connected(h, K, activation_fn=None, biases_initializer = tf.constant_initializer(0))
    

def ratios_critic(x, labels, prob = 1, K=3, deep=False):
    with tf.variable_scope('critic', reuse=tf.AUTO_REUSE) as scope:
        init = tf.keras.initializers.normal(stddev=2.)
        
        regularizer = tf.contrib.layers.l2_regularizer(1.)
        
        x_emb = slim.fully_connected(x, 100, activation_fn=None, biases_initializer = tf.constant_initializer(0))
        l_emb = slim.fully_connected(labels, 100, activation_fn=None, biases_initializer = tf.constant_initializer(0))
        
        h = tf.concat([x_emb,l_emb],1)
        
        h = slim.fully_connected(h, 100, activation_fn=tf.nn.softplus, weights_regularizer=slim.l2_regularizer(.0)) #change back to 50
        h = slim.fully_connected(h, 100, activation_fn=tf.nn.softplus, weights_regularizer=slim.l2_regularizer(.0)) #change back to 10
        
        h_hat = tf.reduce_mean(h[:bs],0,keep_dims=True) # 1x320
        
        theta = tf.get_variable('theta',(1,100),regularizer=regularizer)
        
        partial_ratio = tf.log(tf.maximum(1e-20,tf.matmul(h[:bs], theta, transpose_b=True)))
        
        
        return h, h_hat, theta, partial_ratio

    
def get_log_ratio(samples):
    a = tf.tile([1.,0.,0.],[bs])
    b = tf.tile([0.,1.,0.],[bs])

    label_a = tf.reshape(a,[bs,K])
    label_b = tf.reshape(b,[bs,K])
    
    _,_,_,log_p = ratios_critic(samples,label_a)
    _,_,_,log_q = ratios_critic(samples,label_b)
    return log_p - log_q


    
    
# def ratios_critic(x, prob = 1, K=3, deep=False, l1=K,l2=K,input_dim=320):
#     with tf.variable_scope('critic', reuse=tf.AUTO_REUSE) as scope:
        
#         init = tf.keras.initializers.normal(stddev=0.001)
        
#         def spectral_norm(w, u, iteration=1):
#             w_shape = w.shape.as_list()
#             w = tf.reshape(w, [-1, w_shape[-1]])

            

#             u_hat = u
#             v_hat = None
#             for i in range(iteration):
#                 """
#                 power iteration
#                 Usually iteration = 1 will be enough
#                 """
#                 v_ = tf.matmul(u_hat, tf.transpose(w))
#                 v_hat = tf.nn.l2_normalize(v_)

#                 u_ = tf.matmul(v_hat, w)
#                 u_hat = tf.nn.l2_normalize(u_)

#             u_hat = tf.stop_gradient(u_hat)
#             v_hat = tf.stop_gradient(v_hat)

#             sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

#             with tf.control_dependencies([u.assign(u_hat)]):
#                 w_norm = w / sigma
#                 w_norm = tf.reshape(w_norm, w_shape)


#             return w_norm
        
#         u1 = tf.get_variable("u1", [1, l1], initializer=tf.random_normal_initializer(), trainable=False)
#         u2 = tf.get_variable("u2", [1, l2], initializer=tf.random_normal_initializer(), trainable=False)
#         u4 = tf.get_variable("u4", [1, input_dim], initializer=tf.random_normal_initializer(), trainable=False)
#         u5 = tf.get_variable("u5", [1, input_dim], initializer=tf.random_normal_initializer(), trainable=False)
#         u6 = tf.get_variable("u6", [1, input_dim], initializer=tf.random_normal_initializer(), trainable=False)
        
#         W1 = spectral_norm(tf.get_variable('W1',(K-1,l1),initializer=init),u1)
#         W2 = spectral_norm(tf.get_variable('W2',(l1,l2),initializer=init),u2)
        
#         W4 = spectral_norm(tf.get_variable('W4',(input_dim,input_dim),initializer=init),u4)
#         W5 = spectral_norm(tf.get_variable('W5',(input_dim,input_dim),initializer=init),u5)
#         W6 = spectral_norm(tf.get_variable('W6',(input_dim,input_dim),initializer=init),u6)
        
#         b1 = tf.get_variable('b1',(l1))
#         b2 = tf.get_variable('b2',(l2))
#         b3 = tf.get_variable('b3',(K))
        
#         mu1 = tf.get_variable('mu1',(input_dim),initializer=init)
#         mu2 = tf.get_variable('mu2',(input_dim),initializer=init)
#         mu3 = tf.get_variable('mu3',(input_dim),initializer=init)
        
#         b4 = tf.get_variable('b4',(1))
#         b5 = tf.get_variable('b5',(1))
#         b6 = tf.get_variable('b6',(1))
        
#         x1 = x-mu1
#         W_psd = tf.matmul(W4,W4,transpose_b=True)
#         h = tf.matmul(x1,W_psd)
#         h = tf.matmul(h,x1, transpose_b=True)
#         h1 = tf.reduce_sum(h,-1,keep_dims=True) + b4
        
#         x2 = x-mu2
#         W_psd = tf.matmul(W5,W5,transpose_b=True)
#         h = tf.matmul(x2,W_psd)
#         h = tf.matmul(h,x2, transpose_b=True)
#         h2 = tf.reduce_sum(h,-1,keep_dims=True) + b5
        
# #         x3 = x-mu3
# #         W_psd = tf.matmul(W6,W6,transpose_b=True)
# #         h = tf.matmul(x3,W_psd)
# #         h = tf.matmul(h,x3, transpose_b=True)
# #         h3 = tf.reduce_sum(h,-1,keep_dims=True) + b6
        
#         h = tf.concat([h1,h2],1)
        
#         h = tf.nn.leaky_relu(tf.matmul(h,W1)+b1)
#         h = tf.matmul(h,W2)+b2
        

    
#         return h
    

# def ratios_critic(x, prob = 1, K=3, deep=False, l0=320):
#     with tf.variable_scope('critic', reuse=tf.AUTO_REUSE) as scope:
        
#         init = tf.keras.initializers.normal(stddev=0.001)
        
#         regularizer = tf.contrib.layers.l2_regularizer(0.01)
        
#         def tf_enforce_symmetric_and_pos_diag(A, shift=5.):

#             mask = tf.ones_like(A)
#             ldiag_mask = tf.matrix_band_part(mask, -1, 0)
#             diag_mask = tf.matrix_band_part(mask, 0, 0)
#             strict_ldiag_mask = ldiag_mask - diag_mask

#             B = strict_ldiag_mask * A
#             if len(A.get_shape().as_list()) == 3:
#                 B += tf.transpose(B, [0, 2, 1])
#             else:
#                 B += tf.transpose(B)

#             B += diag_mask * tf.exp(A - shift)

#             return B
        
#         u4 = tf.get_variable("u4", [1, l0], initializer=tf.random_normal_initializer(), trainable=False)
#         u5 = tf.get_variable("u5", [1, l0], initializer=tf.random_normal_initializer(), trainable=False)
#         u6 = tf.get_variable("u6", [1, l0], initializer=tf.random_normal_initializer(), trainable=False)
        
#         W4 = tf_enforce_symmetric_and_pos_diag(tf.get_variable('W4',(l0,l0),initializer=init,
#                                                     regularizer=regularizer))
#         W5 = tf_enforce_symmetric_and_pos_diag(tf.get_variable('W5',(l0,l0),initializer=init,
#                                                     regularizer=regularizer))
#         W6 = tf_enforce_symmetric_and_pos_diag(tf.get_variable('W6',(l0,l0),initializer=init,
#                                                     regularizer=regularizer))
        
        
#         mu1 = tf.get_variable('mu1',(l0),initializer=initializers.Zeros())
#         mu2 = tf.get_variable('mu2',(l0),initializer=initializers.Zeros())
#         mu3 = tf.get_variable('mu3',(l0),initializer=initializers.Zeros())
        
#         b4 = tf.get_variable('b4',(1))
#         b5 = tf.get_variable('b5',(1))
#         b6 = tf.get_variable('b6',(1))
        
#         x1 = x-mu1
#         h = tf.matmul(x1,W4)
#         h = tf.matmul(h,x1, transpose_b=True)
#         h1 = -(tf.reduce_sum(h,-1,keep_dims=True) + b4)
        
#         x2 = x-mu2
#         h = tf.matmul(x2,W5)
#         h = tf.matmul(h,x2, transpose_b=True)
#         h2 = -(tf.reduce_sum(h,-1,keep_dims=True) + b5)
        
# #         x3 = x-mu3
# #         h = tf.matmul(x3,W5)
# #         h = tf.matmul(h,x3, transpose_b=True)
# #         h3 = -(tf.reduce_sum(h,-1,keep_dims=True) + b6)
 
#         h3 = slim.fully_connected(x, K, activation_fn=tf.nn.softplus, weights_regularizer=slim.l2_regularizer(.01))
#         h3 = slim.fully_connected(h3, 1, activation_fn=None, weights_regularizer=slim.l2_regularizer(.01))
        
#         h = tf.concat([h1,h2,h3],1)
# #         h = slim.fully_connected(h, K, activation_fn=None, weights_regularizer=slim.l2_regularizer(.1))

#         return h





# def get_data(mu_1=0.,mu_2=2.,mu_3=2.,scale_p=0.1,scale_q=0.1,scale_m=1.):

#     p = tfd.MultivariateNormalFullCovariance(
#         loc=mu_1,
#         covariance_matrix=scale_p)
    
#     q = tfd.MultivariateNormalDiag(
#         loc=mu_2,
#         scale_diag=scale_q)
    
# #     m = tfd.MultivariateStudentTLinearOperator(
# #     df=1,
# #     loc=mu_3,
# #     scale=tf.linalg.LinearOperatorLowerTriangular(scale_m))
    
    
#     m = tfp.distributions.Mixture(
#           cat=tfp.distributions.Categorical(probs=[.5,.5]),
#           components=[
#             p,
#             q
#         ])
    
#     p_samples = p.sample([bs]) 
#     q_samples = q.sample([bs])
#     #heavy tailed waymark
#     m_samples = m.sample([bs])
#     #linear waymark
#     alpha = tf.expand_dims(tfd.Uniform (0.,1.).sample([bs]),1)
#     m_samples = ((1-alpha)*p_samples + alpha*q_samples )+ m.sample([bs])
# #     m_samples = tf.sqrt(1-alpha*alpha)*p_samples + alpha*q_samples
    
#     return p, q, m, p_samples, q_samples, m_samples, m


def get_gt_ratio_kl(p,q,samples):
    ratio = p.log_prob(samples) - q.log_prob(samples)
    kl = tf.reduce_mean(ratio)
    return ratio, kl

def get_logits(samples, labels, do=1., deep=False, training=True):
#     samples = tf.expand_dims(samples,1)
    return ratios_critic(samples,labels, do,deep=deep)

def get_kl_from_cob(samples_p, samples_q):
    return tf.reduce_mean(get_log_ratio(samples_p))

# def get_kl_from_cob(samples_p, samples_q):
#     log_rat = get_logits(samples_p)
#     V_p = log_rat[:,0]-log_rat[:,1]
    
#     log_rat = get_logits(samples_q)
#     V_q = log_rat[:,0]-log_rat[:,1]
    
#     return 1 + tf.reduce_mean(V_p) - tf.reduce_mean(tf.exp(V_q))

def get_loss(p_samples,q_samples,m_samples,m_dist=None,do=0.8, deep=False):
    
    a = tf.tile([1.,0.,0.],[K*bs])
    b = tf.tile([0.,1.,0.],[K*bs])
    c = tf.tile([0.,0.,1.],[K*bs])

    label_a = tf.reshape(a,[K*bs,K])
    label_b = tf.reshape(b,[K*bs,K])
    label_c = tf.reshape(c,[K*bs,K])
    
    p_samples_ = tf.concat([p_samples,q_samples,m_samples],0)  
    q_samples_ = tf.concat([q_samples,m_samples,p_samples],0)
    m_samples_ = tf.concat([m_samples,p_samples,q_samples],0)
    
    h_p, h_hat_p, theta, _ = get_logits(p_samples_,label_a,do,deep=deep)
    h_q, h_hat_q, _, _ = get_logits(q_samples_,label_b,do,deep=deep)
    h_m, h_hat_m, _, _ = get_logits(m_samples_,label_c,do,deep=deep)
    
    
    H_hat_p = tf.reduce_mean(tf.expand_dims(h_p,-1)@tf.expand_dims(h_p,1),0,keep_dims=True) # fxf
    H_hat_q = tf.reduce_mean(tf.expand_dims(h_q,-1)@tf.expand_dims(h_q,1),0,keep_dims=True) # 
    H_hat_m = tf.reduce_mean(tf.expand_dims(h_m,-1)@tf.expand_dims(h_m,1),0,keep_dims=True) # 
    
    def calc_loss(H_hat,h_hat):
        inner_prod = tf.matmul(theta, H_hat) # 1xf
        quad_term = tf.matmul(inner_prod, theta, transpose_b=True) #1x1
        lin_term = tf.matmul(h_hat, theta, transpose_b=True)
        return 0.5*quad_term - lin_term
        
    
    loss1 = calc_loss(H_hat_p,h_hat_p) + calc_loss(H_hat_q,h_hat_q) + calc_loss(H_hat_m,h_hat_m)
    
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    return loss1 + 0.01 * sum(reg_losses)



# def get_loss(p_samples,q_samples,m_samples,m_dist=None,do=0.8, deep=False):
    
#     logP = get_logits(p_samples,do,deep=deep)
#     logQ = get_logits(q_samples,do,deep=deep)
#     logM = get_logits(m_samples,do,deep=deep)
    
#     a = np.tile([1,0,0],bs)
#     b = np.tile([0,1,0],bs)
#     c = np.tile([0,0,1],bs)

#     label_a = tf.reshape(a,[bs,K])
#     label_b = tf.reshape(b,[bs,K])
#     label_c = tf.reshape(c,[bs,K])

#     disc_loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logP, labels=label_a))
#     disc_loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logQ, labels=label_b))
#     disc_loss_3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logM, labels=label_c))
    
#     loss1 = disc_loss_1 + disc_loss_2 + 1*disc_loss_3
    
#     loss2 = tf.abs(tf.reduce_mean(logP[:,0])) + tf.abs(tf.reduce_mean(logQ[:,1])) + tf.abs(tf.reduce_mean(logM[:,2]))
    
#     reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#     loss = loss1# + 0.01*loss2 #+ 0.001 * sum(reg_losses) + 0.001*loss2
    
#     return loss #+ 10000.*loss2 

# def get_loss(p_samples,q_samples,m_samples,m_dist=None,do=0.8, deep=False):
    
#     logP = get_logits(p_samples,do,deep=deep)
#     logQ = get_logits(q_samples,do,deep=deep)
#     logM = get_logits(m_samples,do,deep=deep)
    
#     ratio_p_m_on_p = logP[:,0]-logP[:,2]
#     ratio_p_m_on_m = logM[:,0]-logM[:,2]
    
#     ratio_q_m_on_q = logQ[:,1]-logQ[:,2]
#     ratio_q_m_on_m = logM[:,1]-logM[:,2]
    
#     loss = - ( tf.reduce_mean(- tf.nn.softplus(-(1+ratio_p_m_on_p))) - tf.reduce_mean(tf.nn.softplus(1+ratio_p_m_on_m)) ) - ( tf.reduce_mean(- tf.nn.softplus(-(1+ratio_q_m_on_q))) - tf.reduce_mean(tf.nn.softplus(1+ratio_q_m_on_m)) )
    
    
#     return loss

# def get_loss(p_samples,q_samples,m_samples,m_dist=None,do=0.8, deep=False):
    
#     logP = get_logits(p_samples,do,deep=deep)
#     logQ = get_logits(q_samples,do,deep=deep)
#     logM = get_logits(m_samples,do,deep=deep)
    
#     p_logit = logP[:,0] # log p/m on samples from p
#     m_logitp = logM[:,0] # log p/m on samples from m
#     q_logit = logQ[:,1] # log q/m on samples from q
#     m_logitq = logM[:,1] # log q/m on samples from m
    
    
#     # Define Single DRE
#     dloss_1 = tf.reduce_mean(
#         tf.nn.sigmoid_cross_entropy_with_logits(logits=p_logit, labels=tf.ones_like(p_logit)) +
#         tf.nn.sigmoid_cross_entropy_with_logits(logits=m_logitp, labels=tf.zeros_like(m_logitp)))
    
#     dloss_2 = tf.reduce_mean(
#         tf.nn.sigmoid_cross_entropy_with_logits(logits=q_logit, labels=tf.ones_like(q_logit)) +
#         tf.nn.sigmoid_cross_entropy_with_logits(logits=m_logitq, labels=tf.zeros_like(m_logitq)))
    
#     dloss_3 = tf.reduce_mean(
#         tf.nn.sigmoid_cross_entropy_with_logits(logits=logP[:,0], labels=tf.ones_like(p_logit)) +
#         tf.nn.sigmoid_cross_entropy_with_logits(logits=logQ[:,0], labels=tf.zeros_like(m_logitq)))
    
# #     dloss = tf.reduce_mean(tf.log_sigmoid(-logP[:,0])) + tf.reduce_mean(tf.log_sigmoid(logQ[:,0]))
    
    
#     return dloss_1 + dloss_2



def get_optim(loss, lr=0.001, b1=0.001, b2=0.999):
    t_vars = tf.trainable_variables()
    print(t_vars)
    c_vars = [var for var in t_vars if 'critic' in var.name]
#     optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2=b2).minimize(loss, var_list=t_vars)
    optim = tf.train.AdamOptimizer(lr).minimize(loss, var_list=t_vars)
    
#     global_step = tf.Variable(0, trainable=False)
#     learning_rate = tf.compat.v1.train.cosine_decay(lr, global_step, 10000, alpha=0.0, name=None)
#     # Passing global_step to minimize() will increment it at each step.
#     optim = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
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
    
    
    