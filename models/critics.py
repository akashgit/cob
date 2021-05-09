import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras

tf.random.set_random_seed(40)

class RatioCritic(tf.keras.Model):

    def __init__(self, K=3, do=0.2):
        super(RatioCritic, self).__init__()
        self.dense1 = tfk.layers.Dense(200, activation=tf.nn.softplus)
        self.dense2 = tfk.layers.Dense(100, activation=tf.nn.softplus)
        self.dense3 = tfk.layers.Dense(K)
        self.dropout1 = tfk.layers.Dropout(do)
        self.dropout2 = tfk.layers.Dropout(do)

    def call(self, inputs, training=False):
        x = tf.expand_dims(inputs,1)
        x = self.dense1(x)
        if training:
            x = self.dropout1(x, training=training)
        x = self.dense2(x)
        if training:
            x = self.dropout2(x, training=training)
            
        return self.dense3(x)
    
class BaeysianRatioCritic(tf.keras.Model):

    def __init__(self, K=3, do=0.2):
        super(BaeysianRatioCritic, self).__init__()
        self.dense1 = tfk.layers.Dense(20, activation=tf.nn.softplus)
        self.dense2 = tfk.layers.Dense(10, activation=tf.nn.softplus)
        self.dense3 = tfp.layers.DenseFlipout(K, 
                                activation=None,
                                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(), name="bayes")
        self.dropout1 = tfk.layers.Dropout(do)
        self.dropout2 = tfk.layers.Dropout(do)

    def call(self, inputs, training=False):
        x = tf.expand_dims(inputs,1)
        x = self.dense1(x)
        if training:
            x = self.dropout1(x, training=training)
        x = self.dense2(x)
        if training:
            x = self.dropout2(x, training=training)
            
        return x, self.dense3(x)