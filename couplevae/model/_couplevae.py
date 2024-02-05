import logging
import os

import numpy
import math
import tensorflow as tf
from scipy import sparse

from .util import balancer, extractor, shuffle_data

log = logging.getLogger(__file__)


class VAE:


    def __init__(self, x_dimension, z_dimension=100, **kwargs):
        tf.reset_default_graph()
        self.x_dim = x_dimension
        self.z_dim = z_dimension
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.dropout_rate = kwargs.get("dropout_rate", 0.2)
        self.model_to_use = kwargs.get("model_path", "./models/couplevae")
        self.alpha = kwargs.get("alpha", 0.001)
        self.beta = kwargs.get("beta", 0.0001)
        self.is_training = tf.placeholder(tf.bool, name='training_flag')
        self.global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
        self.x_0 = tf.placeholder(tf.float32, shape=[None, self.x_dim], name="data")
        self.x_1 = tf.placeholder(tf.float32, shape=[None, self.x_dim], name="data1")
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="latent")
        self.z_mean_c = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="latent_c")
        self.z_mean_p = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="latent_p")
        self.z_mean_0 = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="latent_0")
        self.z_mean_1 = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="latent_1")
        self.transform = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="transform")
        self.time_step = tf.placeholder(tf.int32)
        self.size = tf.placeholder(tf.int32)
        self.init_w = tf.contrib.layers.xavier_initializer()
        self._create_network()
        self._loss_function()
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=1)
        self.init = tf.global_variables_initializer().run(session=self.sess)

    def _encoder_c(self):

        with tf.variable_scope("encoder_c", reuse=tf.AUTO_REUSE):
            h = tf.layers.dense(inputs=self.x_0, units=800, kernel_initializer=self.init_w, use_bias=False)
            h = tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h, self.dropout_rate, training=self.is_training)
            h = tf.layers.dense(inputs=h, units=800, kernel_initializer=self.init_w, use_bias=False)
            h = tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h, self.dropout_rate, training=self.is_training)
            h = tf.layers.dense(inputs=h, units=800, kernel_initializer=self.init_w, use_bias=False)
            h = tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h, self.dropout_rate, training=self.is_training)
            mean_0 = tf.layers.dense(inputs=h, units=self.z_dim, kernel_initializer=self.init_w)
            log_var_0 = tf.layers.dense(inputs=h, units=self.z_dim, kernel_initializer=self.init_w)
            return mean_0, log_var_0

    def _decoder_c(self):


        with tf.variable_scope("decoder_c", reuse=tf.AUTO_REUSE):
            h = tf.layers.dense(inputs=self.z_mean_c, units=800, kernel_initializer=self.init_w, use_bias=False)
            h = tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h, self.dropout_rate, training=self.is_training)
            h = tf.layers.dense(inputs=h, units=800, kernel_initializer=self.init_w, use_bias=False)
            tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h, self.dropout_rate, training=self.is_training)
            h = tf.layers.dense(inputs=h, units=self.x_dim, kernel_initializer=self.init_w, use_bias=True)
            h = tf.nn.relu(h)
            return h
        
    def _encoder_p(self):

        with tf.variable_scope("encoder_p", reuse=tf.AUTO_REUSE):
            h = tf.layers.dense(inputs=self.x_1, units=800, kernel_initializer=self.init_w, use_bias=False)
            h = tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h, self.dropout_rate, training=self.is_training)
            h = tf.layers.dense(inputs=h, units=800, kernel_initializer=self.init_w, use_bias=False)
            h = tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h, self.dropout_rate, training=self.is_training)
            h = tf.layers.dense(inputs=h, units=800, kernel_initializer=self.init_w, use_bias=False)
            h = tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h, self.dropout_rate, training=self.is_training)
            mean_1 = tf.layers.dense(inputs=h, units=self.z_dim, kernel_initializer=self.init_w)
            log_var_1 = tf.layers.dense(inputs=h, units=self.z_dim, kernel_initializer=self.init_w)
            return mean_1, log_var_1

    def _decoder_p(self):

        with tf.variable_scope("decoder_p", reuse=tf.AUTO_REUSE):
            h = tf.layers.dense(inputs=self.z_mean_p, units=800, kernel_initializer=self.init_w, use_bias=False)
            h = tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h, self.dropout_rate, training=self.is_training)
            h = tf.layers.dense(inputs=h, units=800, kernel_initializer=self.init_w, use_bias=False)
            tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h, self.dropout_rate, training=self.is_training)
            h = tf.layers.dense(inputs=h, units=self.x_dim, kernel_initializer=self.init_w, use_bias=True)
            h = tf.nn.relu(h)
            return h

    def _decoder_cp(self):


        with tf.variable_scope("decoder_cp", reuse=tf.AUTO_REUSE):
            h = tf.layers.dense(inputs=self.z_mean_1, units=800, kernel_initializer=self.init_w, use_bias=False)
            h = tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h, self.dropout_rate, training=self.is_training)
            h = tf.layers.dense(inputs=h, units=800, kernel_initializer=self.init_w, use_bias=False)
            tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h, self.dropout_rate, training=self.is_training)
            h = tf.layers.dense(inputs=h, units=self.x_dim, kernel_initializer=self.init_w, use_bias=True)
            h = tf.nn.relu(h)
            return h
        
    def _decoder_pc(self):

        with tf.variable_scope("decoder_pc", reuse=tf.AUTO_REUSE):
            h = tf.layers.dense(inputs=self.z_mean_0, units=800, kernel_initializer=self.init_w, use_bias=False)
            h = tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h, self.dropout_rate, training=self.is_training)
            h = tf.layers.dense(inputs=h, units=800, kernel_initializer=self.init_w, use_bias=False)
            tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h, self.dropout_rate, training=self.is_training)
            h = tf.layers.dense(inputs=h, units=self.x_dim, kernel_initializer=self.init_w, use_bias=True)
            h = tf.nn.relu(h)
            return h

        
    def sample_z_c(self):

        eps = tf.random_normal(shape=[self.size, self.z_dim])
        return self.mu_0 + tf.exp(self.log_var_0 / 2) * eps
    
    def sample_z_p(self):

        eps = tf.random_normal(shape=[self.size, self.z_dim])
        return self.mu_1 + tf.exp(self.log_var_1 / 2) * eps
    
    def couple_c(self):
        with tf.variable_scope("trans_c", reuse=tf.AUTO_REUSE):
            h = tf.layers.dense(inputs=self.z_mean_c, units=800, kernel_initializer=self.init_w, use_bias=False)
            h = tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h, self.dropout_rate, training=self.is_training)
            mean_p = tf.layers.dense(inputs=h, units=self.z_dim, kernel_initializer=self.init_w)
            log_var_p = tf.layers.dense(inputs=h, units=self.z_dim, kernel_initializer=self.init_w)
            return mean_p, log_var_p
    
    def couple_p(self):
        with tf.variable_scope("trans_p", reuse=tf.AUTO_REUSE):
            h = tf.layers.dense(inputs=self.z_mean_p, units=800, kernel_initializer=self.init_w, use_bias=False)
            h = tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h, self.dropout_rate, training=self.is_training)
            mean_c = tf.layers.dense(inputs=h, units=self.z_dim, kernel_initializer=self.init_w)
            log_var_c = tf.layers.dense(inputs=h, units=self.z_dim, kernel_initializer=self.init_w)
            return mean_c, log_var_c
        
    def _sample_hat_z_c(self):

        eps = tf.random_normal(shape=[self.size, self.z_dim])
        return self.mu_p + tf.exp(self.log_var_p / 2) * eps

    def _sample_hat_z_p(self):

        eps = tf.random_normal(shape=[self.size, self.z_dim])
        return self.mu_c + tf.exp(self.log_var_c / 2) * eps
    
    def _create_network(self):

        self.mu_0, self.log_var_0 = self._encoder_c()
        self.mu_1, self.log_var_1 = self._encoder_p()
        self.z_mean_c = self.sample_z_c()
        self.z_mean_p = self.sample_z_p()
        self.mu_p, self.log_var_p = self.couple_c()
        self.mu_c, self.log_var_c = self.couple_p()
        self.z_mean_1 = self._sample_hat_z_c()
        self.z_mean_0 = self._sample_hat_z_p()        
        self.x_hat_0 = self._decoder_c()
        self.x_hat_1 = self._decoder_p()
        self.x_hat_cp = self._decoder_cp()
        self.x_hat_pc = self._decoder_pc()

    def _loss_function(self):

        kl_loss0 = 0.25 * tf.reduce_sum(
            tf.exp(self.log_var_0) + tf.square(self.mu_0) - 1. - self.log_var_0, 1)
        recon_loss0 = 0.25 * tf.reduce_sum(tf.square((self.x_0 - self.x_hat_0)), 1)
        trans_loss0 = 0.25 * tf.reduce_sum(tf.square((self.x_0 - self.x_hat_pc)), 1)
        coupl_loss0 = 0.25 * tf.reduce_sum(tf.square((self.z_mean_c - self.z_mean_1)), 1)
        kl_loss1 = 0.25 * tf.reduce_sum(
            tf.exp(self.log_var_1) + tf.square(self.mu_1) - 1. - self.log_var_1, 1)
        recon_loss1 = 0.25 * tf.reduce_sum(tf.square((self.x_1 - self.x_hat_1)), 1)
        trans_loss1 = 0.25 * tf.reduce_sum(tf.square((self.x_1 - self.x_hat_cp)), 1)
        coupl_loss1 = 0.25 * tf.reduce_sum(tf.square((self.z_mean_p - self.z_mean_0)), 1)
        kl_loss = kl_loss0 + kl_loss1
        recon_loss = recon_loss0 + recon_loss1
        trans_loss = trans_loss0 + trans_loss1
        coupl_loss = coupl_loss0 + coupl_loss1
        self.vae_loss = tf.reduce_mean(recon_loss + coupl_loss + self.alpha * kl_loss + self.alpha * coupl_loss)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.solver = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.vae_loss)
       

    def to_latent(self, data):

        latent = self.sess.run(self.z_mean_c, feed_dict={self.x_0: data, self.size: data.shape[0], self.is_training: False})
        return latent
    
    def to_latent0(self, data):

        latent = self.sess.run(self.z_mean_0, feed_dict={self.x_1: data, self.size: data.shape[0], self.is_training: False})
        return latent

    def to_latent1(self, data1):

        latent = self.sess.run(self.z_mean_1, feed_dict={self.x_0: data1, self.size: data1.shape[0], self.is_training: False})
        return latent
    


    def reconstruct(self, data, use_data=False):

        if use_data:
            latent = data
        else:
            latent = self.to_latent1(data)
        rec_data = self.sess.run(self.x_hat_1, feed_dict={self.z_mean_p: latent, self.is_training: False})
        return rec_data



    def predict(self, adata, conditions, cell_type_key, condition_key, adata_to_predict=None, celltype_to_predict=None,
                 biased=True):


        if celltype_to_predict is not None:
            ctrl_pred = extractor(adata, celltype_to_predict, conditions, cell_type_key, condition_key)[1]
        else:
            ctrl_pred = adata_to_predict

        if sparse.issparse(ctrl_pred.X):
            latent_cd = self.to_latent1(ctrl_pred.X.A)
        else:
            latent_cd = self.to_latent1(ctrl_pred.X)
        
        predicted_cells = self.reconstruct(latent_cd, use_data=True)
        return predicted_cells
    
    def predict_species(self, adata, conditions, cell_type_key, condition_key, adata_to_predict=None, celltype_to_predict=None,
                 biased=True):


        if celltype_to_predict is not None:
            ctrl_pred = extractor(adata, celltype_to_predict, conditions, cell_type_key, condition_key)[1]
        else:
            ctrl_pred = adata_to_predict

        if sparse.issparse(ctrl_pred.X):
            latent_cd = self.to_latent1(ctrl_pred.X.A)
        else:
            latent_cd = self.to_latent1(ctrl_pred.X)
        
        return latent_cd


    

    def restore_model(self):

        self.saver.restore(self.sess, self.model_to_use)

    def train(self, train_data, train_data1, use_validation=False, valid_data=None, valid_data1=None, n_epochs=25, n_epoch=250, batch_size=32, early_stop_limit=20,
              threshold=0.0025, initial_run=True, shuffle=True, save=True):
  
        if initial_run:
            log.info("----Training----")
            assign_step_zero = tf.assign(self.global_step, 0)
            _init_step = self.sess.run(assign_step_zero)
        if not initial_run:
            self.saver.restore(self.sess, self.model_to_use)
        if use_validation and valid_data is None:
            raise Exception("valid_data is None but use_validation is True.")
        if shuffle:
            train_data = shuffle_data(train_data)
            train_data1 = shuffle_data(train_data1)
        loss_hist = []
        patience = early_stop_limit
        min_delta = threshold
        patience_cnt = 0
        for it in range(n_epochs):
            increment_global_step_op = tf.assign(self.global_step, self.global_step + 1)
            _step = self.sess.run(increment_global_step_op)
            current_step = self.sess.run(self.global_step)
            train_loss = 0.0
            for lower in range(0, train_data.shape[0], batch_size):
                upper = min(lower + batch_size, train_data.shape[0])
                if sparse.issparse(train_data.X):
                    x_mb = train_data[lower:upper, :].X.A
                else:
                    x_mb = train_data[lower:upper, :].X
                if sparse.issparse(train_data1.X):
                    x_mb1 = train_data1[lower:upper, :].X.A
                else:
                    x_mb1 = train_data1[lower:upper, :].X
                if upper - lower > 1:
                    _, current_loss_train = self.sess.run([self.solver, self.vae_loss],
                                                          feed_dict={self.x_0: x_mb, self.x_1:x_mb1, self.time_step: current_step,
                                                                      self.size: len(x_mb), self.is_training: True})
                    train_loss += current_loss_train
            if use_validation:
                valid_loss = 0
                for lower in range(0, valid_data.shape[0], batch_size):
                    upper = min(lower + batch_size, valid_data.shape[0])
                    if sparse.issparse(valid_data.X):
                        x_mb = valid_data[lower:upper, :].X.A
                    else:
                        x_mb = valid_data[lower:upper, :].X
                    if sparse.issparse(valid_data1.X):
                        x_mb1 = valid_data1[lower:upper, :].X.A
                    else:
                        x_mb1 = valid_data1[lower:upper, :].X
                    current_loss_valid = self.sess.run(self.vae_loss,
                                                        feed_dict={self.x_0: x_mb, self.x_1: x_mb1, self.time_step: current_step,
                                                                  self.size: len(x_mb), self.is_training: False})
                    valid_loss += current_loss_valid
                loss_hist.append(valid_loss / valid_data.shape[0])
                if it > 0 and loss_hist[it - 1] - loss_hist[it] > min_delta:
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                if patience_cnt > patience:
                    save_path = self.saver.save(self.sess, self.model_to_use)
                    break
            print(f"Epoch {it}: Train VAE Loss: {train_loss / (train_data.shape[0] // batch_size)}")

            
        if save:
            os.makedirs(self.model_to_use, exist_ok=True)
            save_path = self.saver.save(self.sess, self.model_to_use)
            log.info(f"Model saved in file: {save_path}. Training finished")
