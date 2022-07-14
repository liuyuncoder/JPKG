import tensorflow as tf

class Attention(tf.keras.Model):
    def __init__(self, A_in, units,
                attention_units=None,
                activation=None,
                use_bias=True,
                num_heads=1,
                split_value_heads=True,
                query_activation=tf.nn.relu,
                key_activation=tf.nn.relu,
                drop_rate=0.0,
                kernel_regularizer=None,
                bias_regularizer=None,
                 *args, **kwargs):
        """
        units=self.output_dim, num_heads=self.num_heads, split_value_heads = False, activation=tf.nn.relu

        :param units: Positive integer, dimensionality of the output space.
        :param attention_units: Positive integer, dimensionality of the output space for Q and K in attention.
        :param activation: Activation function to use.
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param num_heads: Number of attention heads.
        :param split_value_heads: Boolean. If true, split V as value attention heads, and then concatenate them as output.
            Else, num_heads replicas of V are used as value attention heads, and the mean of them are used as output.
        :param query_activation: Activation function for Q in attention.
        :param key_activation: Activation function for K in attention.
        :param drop_rate: Dropout rate.
        :param kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        :param bias_regularizer: Regularizer function applied to the bias vector.
        """
        super().__init__(*args, **kwargs)
        self.units = units
        self.attention_units = units if attention_units is None else attention_units
        self.drop_rate = drop_rate

        self.query_kernel = None
        self.query_bias = None
        self.query_activation = query_activation

        self.key_kernel = None
        self.key_bias = None
        self.key_activation = key_activation

        self.kernel = None
        self.bias = None

        self.activation = activation
        self.use_bias = use_bias
        self.num_heads = num_heads
        self.split_value_heads = split_value_heads

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        
        self.A_in = A_in
    
    # inputs: u_h_e, u_t_pos_e, u_t_neg_e
    def build(self, input_shapes):
        """
        :param x: Tensor, shape: [num_nodes, num_features], node features
        :param edge_index: Tensor, shape: [2, num_edges], edge information
        :param query_kernel: Tensor, shape: [num_features, num_query_features], weight for Q in attention
        :param query_bias: Tensor, shape: [num_query_features], bias for Q in attention
        :param query_activation: Activation function for Q in attention.
        :param key_kernel: Tensor, shape: [num_features, num_key_features], weight for K in attention
        :param key_bias: Tensor, shape: [num_key_features], bias for K in attention
        :param key_activation: Activation function for K in attention.
        :param kernel: Tensor, shape: [num_features, num_output_features], weight
        :param bias: Tensor, shape: [num_output_features], bias
        :param activation: Activation function to use.
        :param num_heads: Number of attention heads.
        :param split_value_heads: Boolean. If true, split V as value attention heads, and then concatenate them as output.
            Else, num_heads replicas of V are used as value attention heads, and the mean of them are used as output.
        :param drop_rate: Dropout rate.
        :param training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (doing nothing).
        :return: Updated node features (x), shape: [num_nodes, num_output_features]
        """
        x_shape = input_shapes[0]
        num_features = x_shape[-1]
        
        self.query_kernel = self.add_weight("query_kernel", shape=[num_features, self.attention_units],
                                            initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        self.query_bias = self.add_weight("query_bias", shape=[self.attention_units],
                                            initializer="zeros", regularizer=self.bias_regularizer)

        self.key_kernel = self.add_weight("key_kernel", shape=[num_features, self.attention_units],
                                            initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        self.key_bias = self.add_weight("key_bias", shape=[self.attention_units],
                                        initializer="zeros", regularizer=self.bias_regularizer)

        self.kernel = self.add_weight("kernel", shape=[num_features, self.units],
                                        initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units],
                                        initializer="zeros", regularizer=self.bias_regularizer)
        # self.edge_dropout = tf.keras.layers.Dropout(0.1)

    def predict_edge(self, h_e, t_e):
        # dot product
        logits = tf.reduce_sum(h_e * t_e, axis=-1)
        return logits
    
    def call(self, inputs, update, training=False):
        
        if update is False:
            h_e, t_pos_e, t_neg_e, h_indices, pos_t_indices = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            h = self._batch_encode(h_e, t_pos_e, h_indices, pos_t_indices, training=True)
            pos_logits = self.predict_edge(h, t_pos_e)
            neg_logits = self.predict_edge(h, t_neg_e)
            return h, pos_logits, neg_logits
        else:
            h_e, t_pos_e = inputs[0], inputs[1]
            update_attn = self._update(h_e, t_pos_e)
            return update_attn

    def _update(self, h_e, t_e):
        Q = h_e @ self.query_kernel
        Q += self.query_bias
        if self.query_activation is not None:
            Q = self.query_activation(Q)
        
        K = t_e @ self.key_kernel
        K += self.key_bias
        if self.key_activation is not None:
            K = self.key_activation(K)
        if self.split_value_heads:
            Q_ = tf.concat(tf.split(Q, self.num_heads, axis=-1), axis=0)
            K_ = tf.concat(tf.split(K, self.num_heads, axis=-1), axis=0)
        else:
            Q_ = Q
            K_ = K
        scale = tf.math.sqrt(tf.cast(tf.shape(Q_)[-1], tf.float32))
        att_score_ = tf.reduce_sum(tf.matmul(Q_, K_, transpose_b=True) / scale, 1)
        
        return att_score_
    
    def _batch_encode(self, h_e, t_e, h_indices, t_indices, training=False):
        t_V = t_e @ self.kernel
        att_score = self.A_in[h_indices, t_indices]
        if self.split_value_heads:
            t_V_ = tf.concat(tf.split(t_V, self.num_heads, axis=-1), axis=0)
            att_score_ = tf.transpose(tf.tile(att_score, [1, self.num_heads]))
        else:
            t_V_ = t_V
            att_score_ = tf.transpose(att_score)
        # using A_in to compute head embeddings by its neighbors.
        t_h_ = tf.math.multiply(att_score_, t_V_)

        if self.split_value_heads:
            t_h = tf.concat(tf.split(t_h_, self.num_heads, axis=0), axis=-1)
        else:
            t_h = t_h_

        if self.bias is not None:
            t_h += self.bias

        if self.activation is not None:
            t_h = self.activation(t_h)
        return t_h