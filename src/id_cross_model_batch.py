import numpy as np
from pandas import concat
import tensorflow as tf
from id_cross_layers import CrossCompressUnit
from gat_layers_batch import Attention
import scipy.sparse as sp


class MKR(object):
    def __init__(self, args, n_users, n_items, n_entities, A_in, all_h_list, all_t_list):
        super(MKR, self).__init__()
        self.user_bias = tf.Variable([ 0.1 for _ in range(n_users)])
        self.item_bias = tf.Variable([ 0.1 for _ in range(n_items)])
        self.global_bias = tf.Variable(0.1)

        self.user_emb_layer = tf.keras.layers.Embedding(n_users, args.dim)
        self.item_emb_layer = tf.keras.layers.Embedding(n_items, args.dim)
        self.entity_emb_layer = tf.keras.layers.Embedding(n_entities, args.dim)
        
        self.user_cc_unit = CrossCompressUnit(args.dim)
        self.item_cc_unit = CrossCompressUnit(args.dim)

        self.rs_mlp = tf.keras.layers.Dense(args.dim)
        self.rs_pred_mlp = tf.keras.layers.Dense(1)
        
        # for GAT layers
        # units: attention_units.
        self.A_in = A_in
        self.gat_layer = Attention(A_in=self.A_in, units=args.dim, num_heads=args.num_heads, split_value_heads = False, activation=tf.nn.relu)
        self.n_entities = n_entities
        self.n_fold = 100
        self.all_h_list = all_h_list
        self.all_t_list = all_t_list

        self.concat_emb_layer = tf.keras.layers.Dense(args.dim, activation='LeakyReLU')
        
        self.u_JPKG_layer = tf.keras.layers.Dense(args.dim)
        self.i_JPKG_layer = tf.keras.layers.Dense(args.dim)
        
    def _propagation_ml(self, args, u_e, u_h_e, i_e, i_h_e):
        
        for l in range(0, 1):
            u_e = self.user_cc_unit([u_e, u_h_e])
            i_e = self.item_cc_unit([i_e, i_h_e])
        return u_e, i_e
    
    def train_rs(self, training, args, user_indices, item_indices, labels, user_head_indices, item_head_indices, user_start_index):
        u_e = self.user_emb_layer(user_indices-user_start_index)
        i_e = self.item_emb_layer(item_indices)
        user_bias_num = tf.nn.embedding_lookup(self.user_bias, user_indices-user_start_index)
        item_bias_num = tf.nn.embedding_lookup(self.item_bias, item_indices)
        # get the representations of heads and tails based on A_in obtained from last epoch.
        entity_embeddings = self._create_r_embedding(args, training = training)

        u_JPKG_e = []
        i_JPKG_e = []
        
        for l in range(0, args.L+1):
            u_h_e = tf.nn.embedding_lookup(entity_embeddings[l], user_head_indices)
            i_h_e = tf.nn.embedding_lookup(entity_embeddings[l], item_head_indices)
            u_cc_e, i_cc_e = self._propagation_ml(args, u_e, u_h_e, i_e, i_h_e)
            u_JPKG_e += [u_cc_e]
            i_JPKG_e += [i_cc_e]
            
        u_JPKG_e = tf.concat(u_JPKG_e, 1)
        u_e = self.u_JPKG_layer(u_JPKG_e)
        i_JPKG_e = tf.concat(i_JPKG_e, 1)
        i_e = self.i_JPKG_layer(i_JPKG_e)

        # high layer
        use_inner_product = True
        if use_inner_product:
            # [batch_size]
            scores = tf.reduce_sum(u_e * i_e, axis=1) 
            scores = scores+ user_bias_num + item_bias_num + self.global_bias
        else:
            # [batch_size, dim * 2]
            user_item_concat = tf.concat([u_e, i_e], axis=1)
            for _ in range(args.H):
                # [batch_size, dim * 2]
                user_item_concat = self.rs_mlp(user_item_concat)
            # [batch_size]
            scores = tf.squeeze(self.rs_pred_mlp(user_item_concat))
            scores = scores+ user_bias_num + item_bias_num + self.global_bias
        
        # RS
        base_loss_rs = tf.reduce_mean(
            tf.square(tf.subtract(scores, labels)))
        l2_loss_rs = tf.nn.l2_loss(u_e) + tf.nn.l2_loss(i_e)
        loss_rs = base_loss_rs + l2_loss_rs * args.l2_weight

        return scores, loss_rs

    def compute_kge_loss(self, pos_edge_logits, neg_edge_logits):
        pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=pos_edge_logits,
        labels=tf.ones_like(pos_edge_logits)
        )

        neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=neg_edge_logits,
            labels=tf.zeros_like(neg_edge_logits)
        )

        return tf.reduce_mean(pos_losses)+tf.reduce_mean(neg_losses)
    
    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = self.n_entities // self.n_fold

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_entities
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X.tocsr()[start:end]))
        return A_fold_hat
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)
    
    def _create_r_embedding(self, args, training):
        entity_num = np.arange(self.n_entities)
        ego_embeddings = self.entity_emb_layer(entity_num)
        all_embeddings = [ego_embeddings]
        A_fold_hat = self._split_A_hat(self.A_in)
        for k in range(0, args.L):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse.sparse_dense_matmul(A_fold_hat[f], ego_embeddings)) # function 3 in the paper.

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            
            concat_embeddings = tf.concat([ego_embeddings, side_embeddings], 1)
            ego_embeddings = self.concat_emb_layer(concat_embeddings)

            all_embeddings += [ego_embeddings]

        return all_embeddings
    
    def _create_Bi_embedding(self, args, training):
        entity_num = np.arange(self.n_entities)
        ego_embeddings = self.entity_emb_layer(entity_num)
        all_embeddings = [ego_embeddings]
        A_fold_hat = self._split_A_hat(self.A_in)
        for k in range(0, args.L):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse.sparse_dense_matmul(A_fold_hat[f], ego_embeddings)) # function 3 in the paper.

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)

            add_embeddings = ego_embeddings + side_embeddings
            sum_embeddings = self.add_emb_layer(add_embeddings)
            
            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = self.bi_emb_layer(bi_embeddings)

            ego_embeddings = bi_embeddings + sum_embeddings

            all_embeddings += [ego_embeddings]
            
        return all_embeddings
    
    def train_kge(self, training, args, h_indices, pos_t_indices, neg_t_indices, u_start_index):
        h_e = self.entity_emb_layer(h_indices)
        pos_t_e = self.entity_emb_layer(pos_t_indices)
        neg_t_e = self.entity_emb_layer(neg_t_indices)
        
        for _ in range(args.H):
            inputs = [h_e, pos_t_e, neg_t_e, h_indices, pos_t_indices]
            # h, pos_logits, neg_logits
            h_e, pos_edge_logits, neg_edge_logits = self.gat_layer(inputs, update=False, training=True)
            
        score_kge = self.compute_kge_loss(pos_edge_logits, neg_edge_logits)
        base_loss_kge = -score_kge
        l2_loss_kge = tf.nn.l2_loss(h_e) + tf.nn.l2_loss(pos_t_e) + tf.nn.l2_loss(neg_t_e)

        loss_kge = base_loss_kge + l2_loss_kge * args.l2_weight

        return loss_kge
    
    def _create_attentive_A_out(self): # function 5 in the paper.
        indices = np.mat([self.all_h_list, self.all_t_list]).transpose()
        A = tf.sparse.softmax(tf.sparse.SparseTensor(indices, self.A_values, self.A_in.shape))
        return A
    
    def update_attn(self):
        fold_len = len(self.all_h_list) // self.n_fold
        kg_score = []

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = len(self.all_h_list)
            else:
                end = (i_fold + 1) * fold_len

            update_h_indices = np.array(self.all_h_list[start:end])
            update_pos_t_indices = np.array(self.all_t_list[start:end])
            update_h_e = self.entity_emb_layer(update_h_indices)
            update_pos_t_e = self.entity_emb_layer(update_pos_t_indices)
            attn = self.gat_layer([update_h_e, update_pos_t_e], update=True, training=True)
            kg_score = tf.concat([kg_score, attn], axis=0)

        # π(h, r , t)
        self.A_values = kg_score
        new_A = self._create_attentive_A_out()
        new_A_values = new_A.values
        new_A_indices = new_A.indices

        rows = new_A_indices[:, 0]
        cols = new_A_indices[:, 1]
        self.A_in = sp.coo_matrix((new_A_values, (rows, cols)), shape=(self.n_entities, self.n_entities))
