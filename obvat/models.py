from layers import *
from metrics import *
from utils import kl_divergence_with_logit, entropy_y_x, get_normalized_matrix, get_normalized_vector

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, is_sparse, adv_shape, input_dim, multitask, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.is_sparse = is_sparse
        self.multitask = multitask

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        with tf.variable_scope(self.name):
            self._build()
        self.outputs = self.get_output(self.inputs)

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
        training_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        self.r_vadv = tf.get_variable(initializer=tf.random_uniform_initializer(), name="r_vadv", shape=adv_shape)
        self.vloss = self.vat_loss()
        self.rnorm = tf.nn.l2_loss(self.r_vadv, "vat_l2")
        self.adv_reset = tf.assign(self.r_vadv, tf.truncated_normal(tf.shape(self.r_vadv), stddev=0.01))
        self.adv_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.rnorm-self.vloss, var_list=[self.r_vadv])
        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss, var_list=training_variables)

    def vat_loss(self, logit=None):
        if self.is_sparse:
            inp = tf.sparse_add(self.inputs, tf.SparseTensor(self.inputs.indices, self.r_vadv, self.inputs.dense_shape))
        else:
            inp = tf.add(self.inputs, self.r_vadv)
        if logit is None:
            logit = self.get_output(self.inputs)
        logit_p = tf.stop_gradient(logit)
        logit_m = self.get_output(inp)
        loss = kl_divergence_with_logit(logit_p, logit_m)
        return loss
    # def vat_loss_edge(self, x, logit):
    #     a_hat = self.placeholders["support"]
    #     d = tf.random_normal(shape=a_hat.dense_shape)
    #     d = get_normalized_vector(d) * FLAGS.xi
    #     a_hat_ = tf.sparse_add(a_hat, d)
    #     print(a_hat_)
    #     logit_p = logit
    #     logit_m = self.get_output(x, a_hat_, False)
    #     dist = kl_divergence_with_logit(logit_p, logit_m)
    #     grad = tf.gradients(dist, [d], aggregation_method=2)[0]
    #     d = tf.stop_gradient(grad)
    #     r_vadv = get_normalized_vector(d) * FLAGS.epsilon1
    #     a_hat_ = tf.sparse_add(a_hat, r_vadv)
    #     logit_p = tf.stop_gradient(logit)
    #     logit_m = self.get_output(x, a_hat_, False)
    #     loss = kl_divergence_with_logit(logit_p, logit_m)
    #     return tf.identity(loss, name="vat_loss")

    def get_output(self, inp, a_hat=None, a_sparse=True):
        if a_hat is None:
            a_hat = self.placeholders["support"]
        activations = []
        activations.append(inp)
        for layer in self.layers:
            hidden = layer(activations[-1], a_hat, a_sparse, self.placeholders['dropout'], self.placeholders['num_features_nonzero'])
            activations.append(hidden)
        return activations[-1]

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'])

        x = self.inputs
        logit = self.get_output(x)
        self.loss += FLAGS.p1*self.vat_loss(logit) + FLAGS.p2*entropy_y_x(logit)#FLAGS.p3*self.vat_loss_edge(self.inputs, logit)

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=self.is_sparse,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)
