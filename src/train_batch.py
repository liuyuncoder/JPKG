import tensorflow as tf
import numpy as np
from id_cross_model_batch import MKR
import time

physical_devices = tf.config.experimental.list_physical_devices('GPU') 
# setting visuable GPU.
tf.config.experimental.set_visible_devices(physical_devices[0:], 'GPU')
# tf.config.experimental.set_visible_devices([], 'GPU')

def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity = data[0], data[1], data[2]
    train_data, eval_data, test_data = data[3], data[4], data[5]
    pos_neg_kg_edges = data[6]
    user_start_index = data[7]
    A_in = data[8]
    all_h_list = data[9]
    all_t_list = data[10]

    model = MKR(args, n_user, n_item, n_entity, A_in, all_h_list, all_t_list)
    optimizer_rs = tf.keras.optimizers.Adam(
    args.lr_rs, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    optimizer_kge = tf.keras.optimizers.Adam(
    args.lr_kge, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    @tf.function  # optimize eager code
    def rs_train_step(user_indices, item_indices, labels, user_head_indices, item_head_indices):
        with tf.GradientTape() as tape:
            _, loss_rs = model.train_rs(True, args, user_indices, item_indices, labels, user_head_indices, item_head_indices, user_start_index)
        vars = tape.watched_variables()
        rs_gradients = tape.gradient(loss_rs, vars)
        optimizer_rs.apply_gradients(zip(rs_gradients, vars))
    
    # @tf.function
    def kge_train_step(h_ind, pos_t_ind, neg_t_ind):
        with tf.GradientTape() as kge_tape:
            # loss_kge, layer_attn
            loss_kge = model.train_kge(True, args, h_ind, pos_t_ind, neg_t_ind, user_start_index)
        kge_vars = kge_tape.watched_variables()
        kge_gradients = kge_tape.gradient(loss_kge, kge_vars)
        optimizer_kge.apply_gradients(zip(kge_gradients, kge_vars))
    
    def rmse_eval(user_indices, item_indices, labels, user_head_indices, item_head_indices):
        scores, loss_re = model.train_rs(False, args, user_indices, item_indices, labels, user_head_indices, item_head_indices, user_start_index)
        rating_rmse = tf.sqrt(tf.reduce_mean(
            tf.square(tf.subtract(scores, labels))))
        return labels, scores, rating_rmse.numpy(), loss_re
    for step in range(args.n_epochs):
        # RS training
        start_time = time.time()
        np.random.shuffle(train_data)
        rs_start = 0
        # print("---------start train rs")
        while rs_start < train_data.shape[0]:
            user_indices, item_indices, labels, user_head_indices, item_head_indices = get_feed_dict_for_rs(train_data,
            rs_start, rs_start+args.batch_size)
            rs_start += args.batch_size
            # print("start is: ", start)
            rs_train_step(user_indices, item_indices, labels, user_head_indices, item_head_indices)
            
        # KGE training
        if step % args.kge_interval == 0:
            np.random.shuffle(pos_neg_kg_edges)
            # np.random.shuffle(item_kg_edges)
            kg_len = len(pos_neg_kg_edges)
            kge_start = 0
            # print("------- start train kge")
            while kge_start < kg_len:
                h_indices, pos_t_kg_indices, neg_t_kg_indices = get_feed_dict_for_kge(
                    pos_neg_kg_edges, kge_start, kge_start+args.kge_batch_size)
                kge_start += args.kge_batch_size
                # train kge and obtain explainable results
                kge_train_step(h_indices, pos_t_kg_indices, neg_t_kg_indices)
        
        if args.update_attn is True:
            model.update_attn()
        # rating prediction
        train_user_indices, train_item_indices, train_labels, train_user_head_indices, train_item_head_indices = get_feed_dict_for_rs(train_data, 0, train_data.shape[0])
        eval_user_indices, eval_item_indices, eval_labels, eval_user_head_indices, eval_item_head_indices = get_feed_dict_for_rs(eval_data, 0, eval_data.shape[0])
        test_user_indices, test_item_indices, test_labels, test_user_head_indices, test_item_head_indices = get_feed_dict_for_rs(test_data, 0, test_data.shape[0])
        _, _, train_rmse, loss_train = rmse_eval(train_user_indices, train_item_indices, train_labels, train_user_head_indices, train_item_head_indices)
        _, _, eval_rmse, loss_eval = rmse_eval(eval_user_indices, eval_item_indices, eval_labels, eval_user_head_indices, eval_item_head_indices)
        _, _, test_rmse, loss_test = rmse_eval(test_user_indices, test_item_indices, test_labels, test_user_head_indices, test_item_head_indices)
        # loss_file.write('%.4f' % (eval_rmse))
    
        print('epoch %d    train rmse: %.4f    eval rmse: %.4f    eval loss: %.4f    test rmse: %.4f'
                % (step, train_rmse, eval_rmse, loss_eval, test_rmse))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start_time))
        # end
    # loss_file.close()

def get_feed_dict_for_rs(data, start, end):
    user_indices = data[start:end, 0]
    item_indices = data[start:end, 1]
    labels =  tf.cast(data[start:end, 2], tf.float32)
    user_head_indices = data[start:end, 0]
    item_head_indices = data[start:end, 1]

    return user_indices, item_indices, labels, user_head_indices, item_head_indices

def get_feed_dict_for_kge(pos_neg_kg_edges, start, end):
    h_indices = pos_neg_kg_edges[start:end, 0]
    pos_t_kg_indices = pos_neg_kg_edges[start:end, 1]
    neg_t_kg_indices = pos_neg_kg_edges[start:end, 2]

    return h_indices, pos_t_kg_indices, neg_t_kg_indices
