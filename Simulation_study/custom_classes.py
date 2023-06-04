import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tensorflow.python.ops import array_ops
from tensorflow import nn
from tensorflow.python.framework import smart_cond as smart_module
from tensorflow.python.ops import control_flow_ops, variables

# --- Defining custom classes --- #


def smart_cond(pred, true_fn=None, false_fn=None, name=None):

    if isinstance(pred, variables.Variable):
        return control_flow_ops.cond(
            pred, true_fn=true_fn, false_fn=false_fn, name=name)
    return smart_module.smart_cond(
        pred, true_fn=true_fn, false_fn=false_fn, name=name)


class CVE(Layer):
    def __init__(self, hid_units, output_dim):
        self.hid_units = hid_units
        self.output_dim = output_dim
        super(CVE, self).__init__()

    def build(self, input_shape):
        self.W1 = self.add_weight(name='CVE_W1',
                                  shape=(1, self.hid_units),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.b1 = self.add_weight(name='CVE_b1',
                                  shape=(self.hid_units,),
                                  initializer='zeros',
                                  trainable=True)
        self.W2 = self.add_weight(name='CVE_W2',
                                  shape=(self.hid_units, self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(CVE, self).build(input_shape)

    def call(self, x):
        x = K.expand_dims(x, axis=-1)
        x = K.dot(K.tanh(K.bias_add(K.dot(x, self.W1), self.b1)), self.W2)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)


class Attention(Layer):

    def __init__(self, hid_dim):
        self.hid_dim = hid_dim
        super(Attention, self).__init__()

    def build(self, input_shape):
        d = input_shape.as_list()[-1]
        self.W = self.add_weight(shape=(d, self.hid_dim), name='Att_W',
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.hid_dim,), name='Att_b',
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(shape=(self.hid_dim, 1), name='Att_u',
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask, mask_value=-1e30):
        attn_weights = K.dot(
            K.tanh(K.bias_add(K.dot(x, self.W), self.b)), self.u)
        mask = K.expand_dims(mask, axis=-1)
        attn_weights = mask*attn_weights + (1-mask)*mask_value
        attn_weights = K.softmax(attn_weights, axis=-2)
        return attn_weights

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)


class Transformer(Layer):

    def __init__(self, N=2, h=8, dk=None, dv=None, dff=None, dropout=0):
        self.N, self.h, self.dk, self.dv, self.dff, self.dropout = N, h, dk, dv, dff, dropout
        self.epsilon = K.epsilon() * K.epsilon()
        super(Transformer, self).__init__()

    def build(self, input_shape):
        d = input_shape.as_list()[-1]
        if self.dk is None:
            self.dk = d//self.h
        if self.dv is None:
            self.dv = d//self.h
        if self.dff is None:
            self.dff = 2*d
        self.Wq = self.add_weight(shape=(self.N, self.h, d, self.dk), name='Wq',
                                  initializer='glorot_uniform', trainable=True)
        self.Wk = self.add_weight(shape=(self.N, self.h, d, self.dk), name='Wk',
                                  initializer='glorot_uniform', trainable=True)
        self.Wv = self.add_weight(shape=(self.N, self.h, d, self.dv), name='Wv',
                                  initializer='glorot_uniform', trainable=True)
        self.Wo = self.add_weight(shape=(self.N, self.dv*self.h, d), name='Wo',
                                  initializer='glorot_uniform', trainable=True)
        self.W1 = self.add_weight(shape=(self.N, d, self.dff), name='W1',
                                  initializer='glorot_uniform', trainable=True)
        self.b1 = self.add_weight(shape=(self.N, self.dff), name='b1',
                                  initializer='zeros', trainable=True)
        self.W2 = self.add_weight(shape=(self.N, self.dff, d), name='W2',
                                  initializer='glorot_uniform', trainable=True)
        self.b2 = self.add_weight(shape=(self.N, d), name='b2',
                                  initializer='zeros', trainable=True)
        self.gamma = self.add_weight(shape=(2*self.N,), name='gamma',
                                     initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=(2*self.N,), name='beta',
                                    initializer='zeros', trainable=True)
        super(Transformer, self).build(input_shape)

    def call(self, x, mask, mask_value=-1e-30):
        mask = K.expand_dims(mask, axis=-2)
        for i in range(self.N):
            # MHA
            mha_ops = []
            for j in range(self.h):
                q = K.dot(x, self.Wq[i, j, :, :])
                k = K.permute_dimensions(
                    K.dot(x, self.Wk[i, j, :, :]), (0, 2, 1))
                v = K.dot(x, self.Wv[i, j, :, :])
                A = K.batch_dot(q, k)
                # Mask unobserved steps.
                A = mask*A + (1-mask)*mask_value
                # Mask for attention dropout.

                def dropped_A():
                    dp_mask = K.cast(
                        (K.random_uniform(shape=array_ops.shape(A)) >= self.dropout), K.floatx())
                    return A*dp_mask + (1-dp_mask)*mask_value
                A = smart_cond(K.learning_phase(), dropped_A,
                               lambda: array_ops.identity(A))
                A = K.softmax(A, axis=-1)
                mha_ops.append(K.batch_dot(A, v))
            conc = K.concatenate(mha_ops, axis=-1)
            proj = K.dot(conc, self.Wo[i, :, :])
            # Dropout.
            proj = smart_cond(K.learning_phase(), lambda: array_ops.identity(nn.dropout(proj, rate=self.dropout)),
                              lambda: array_ops.identity(proj))
            # Add & LN
            x = x+proj
            mean = K.mean(x, axis=-1, keepdims=True)
            variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            x = (x - mean) / std
            x = x*self.gamma[2*i] + self.beta[2*i]
            # FFN
            ffn_op = K.bias_add(K.dot(K.relu(K.bias_add(K.dot(x, self.W1[i, :, :]), self.b1[i, :])),
                                      self.W2[i, :, :]), self.b2[i, :,])
            # Dropout.
            ffn_op = smart_cond(K.learning_phase(), lambda: array_ops.identity(nn.dropout(ffn_op, rate=self.dropout)),
                                lambda: array_ops.identity(ffn_op))
            # Add & LN
            x = x+ffn_op
            mean = K.mean(x, axis=-1, keepdims=True)
            variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            x = (x - mean) / std
            x = x*self.gamma[2*i+1] + self.beta[2*i+1]
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
