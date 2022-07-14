import tensorflow as tf

class CrossCompressUnit(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(CrossCompressUnit, self).__init__()
        self.dim = dim
        self.weight_vv_layer = tf.keras.layers.Dense(1)
        self.weight_ev_layer = tf.keras.layers.Dense(1)
        self.bias_v = tf.Variable([ 0.1 for _ in range(dim)])

    def call(self, inputs):
        # [batch_size, dim]
        v, e = inputs

        # [batch_size, dim, 1], [batch_size, 1, dim]
        v = tf.expand_dims(v, axis=2)
        e = tf.expand_dims(e, axis=1)

        # [batch_size, dim, dim]
        c_matrix = tf.matmul(v, e)
        c_matrix_transpose = tf.transpose(c_matrix, perm=[0, 2, 1])

        # [batch_size * dim, dim]
        c_matrix = tf.reshape(c_matrix, [-1, self.dim])
        c_matrix_transpose = tf.reshape(c_matrix_transpose, [-1, self.dim])

        # [batch_size, dim]
        v_output = tf.reshape(self.weight_vv_layer(c_matrix) + self.weight_ev_layer(c_matrix_transpose),
                              [-1, self.dim]) + self.bias_v

        return v_output