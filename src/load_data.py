import numpy as np
import os
import scipy.sparse as sp
import collections


def load_data(args):
    n_user, n_item, train_data, eval_data, test_data = load_rating(args)
    user_review_dict, item_review_dict, user_start_index = load_review2entity(args)
    n_entity, pos_neg_kg_edges = load_kg(args, user_review_dict, item_review_dict)
    adj_list, _ = _get_relational_adj_list(n_entity, pos_neg_kg_edges.astype(int)) # adj_list only includes two coo_matrices.
    lap_list = _get_relational_lap_list(args, adj_list)
    all_h_list, all_t_list, _ = _get_all_kg_data(lap_list) # matrices-->list-->reorder list
    A_in = sum(lap_list).astype('float32')
    
    print('data loaded.')
    return n_user, n_item, n_entity, train_data, eval_data, test_data, pos_neg_kg_edges, user_start_index, A_in, all_h_list, all_t_list

# define the adj matrix of kg.
def _get_relational_adj_list(n_entity, pos_neg_kg_edges):
    adj_mat_list = []
    adj_r_list = []
    relation = 0
    relation_inv = 1
    def _np_mat2sp_adj(np_mat):
        n_all = n_entity
        # single-direction
        a_rows = np_mat[:, 0]
        a_cols = np_mat[:, 1]
        a_vals = [1.] * len(a_rows)

        b_rows = a_cols
        b_cols = a_rows
        b_vals = [1.] * len(b_rows)

        a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
        b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))

        return a_adj, b_adj

    R, R_inv = _np_mat2sp_adj(pos_neg_kg_edges)
    adj_mat_list.append(R)
    adj_r_list.append(relation)

    adj_mat_list.append(R_inv)
    adj_r_list.append(relation_inv)
    print('\tconvert kg into adj mat done.')

    return adj_mat_list, adj_r_list
    
def _get_relational_lap_list(args, adj_list):
    def _bi_norm_lap(adj):
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()
    def _si_norm_lap(adj):
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()
    if args.adj_type == 'bi':
        lap_list = [_bi_norm_lap(adj) for adj in adj_list]
        print('\tgenerate bi-normalized adjacency matrix.')
    else:
        lap_list = [_si_norm_lap(adj) for adj in adj_list]
        print('\tgenerate si-normalized adjacency matrix.')

    return lap_list

def _get_all_kg_data(lap_list):
    def _reorder_list(org_list, order):
        new_list = np.array(org_list)
        new_list = new_list[order]
        return new_list

    all_h_list, all_t_list = [], []
    all_v_list = []

    for l_id, lap in enumerate(lap_list):
        all_h_list += list(lap.row)
        all_t_list += list(lap.col)
        all_v_list += list(lap.data)

    assert len(all_h_list) == sum([len(lap.data) for lap in lap_list])

    # resort the all_h/t/r/v_list,
    # ... since tensorflow.sparse.softmax requires indices sorted in the canonical lexicographic order
    print('\treordering indices...')
    org_h_dict = dict()

    for idx, h in enumerate(all_h_list):
        if h not in org_h_dict.keys():
            org_h_dict[h] = [[],[]]

        org_h_dict[h][0].append(all_t_list[idx])
        # org_h_dict[h][1].append(all_r_list[idx])
        org_h_dict[h][1].append(all_v_list[idx])
    print('\treorganize all kg data done.')

    sorted_h_dict = dict()
    for h in org_h_dict.keys():
        org_t_list, org_v_list = org_h_dict[h]
        sort_t_list = np.array(org_t_list)
        sort_order = np.argsort(sort_t_list)

        sort_t_list = _reorder_list(org_t_list, sort_order)
        # sort_r_list = _reorder_list(org_r_list, sort_order)
        sort_v_list = _reorder_list(org_v_list, sort_order)

        sorted_h_dict[h] = [sort_t_list, sort_v_list]
    print('\tsort meta-data done.')

    od = collections.OrderedDict(sorted(sorted_h_dict.items()))
    new_h_list, new_t_list, new_v_list = [], [], []

    for h, vals in od.items():
        new_h_list += [h] * len(vals[0])
        new_t_list += list(vals[0])
        # new_r_list += list(vals[1])
        new_v_list += list(vals[1])


    assert sum(new_h_list) == sum(all_h_list)
    assert sum(new_t_list) == sum(all_t_list)

    print('\tsort all data done.')


    return new_h_list, new_t_list, new_v_list

def load_review2entity(args):
    user_review_dict = dict()
    item_review_dict = dict()
    user_id_dict = {}
    item_id_dict = {}
    print('reading review2entity file ...')
    user_file = args.TPS_DIR + 'review_user2entity_final.txt'
    item_file = args.TPS_DIR + 'review_item2entity_final.txt'
    user_start_index_file = args.TPS_DIR + 'user_start_index.txt'
    for user_line in open(user_file, encoding='utf-8').readlines():
        user_index = int(user_line.strip().split(',')[0])
        if user_line.strip().split(',')[1] != 'PAD':
            entity_id = int(user_line.strip().split(',')[1])
        if user_index not in user_review_dict:
            user_review_dict[user_index] = []
        if entity_id not in user_review_dict[user_index]:
            user_review_dict[user_index].append(entity_id)

        if entity_id not in user_id_dict:
            user_id_dict[entity_id] = []
        if user_index not in user_id_dict[entity_id]:
            user_id_dict[entity_id].append(user_index)

    for item_line in open(item_file, encoding='utf-8').readlines():
        item_index = int(item_line.strip().split(',')[0])
        if item_line.strip().split(',')[1] != 'PAD':
            item_entity_id = int(item_line.strip().split(',')[1])
        if item_index not in item_review_dict:
            item_review_dict[item_index] = []
        if item_entity_id not in item_review_dict[item_index]:
            item_review_dict[item_index].append(item_entity_id)
        
        if item_entity_id not in item_id_dict:
            item_id_dict[item_entity_id] = []
        if item_index not in item_id_dict[item_entity_id]:
            item_id_dict[item_entity_id].append(item_index)
    user_start_index = open(user_start_index_file, encoding='utf-8').readlines()[0]
    return user_review_dict, item_review_dict, int(user_start_index)

def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_train = args.TPS_DIR + 'amazon_train_final'
    rating_valid = args.TPS_DIR + 'amazon_valid_final'
    rating_test = args.TPS_DIR + 'amazon_test_final'

    if os.path.exists(rating_train + '.npy'):
        rating_np_train = np.load(rating_train + '.npy')
    else:
        rating_np_train = np.loadtxt(rating_train + '.txt', dtype=np.int32)
        np.save(rating_train + '.npy', rating_np_train)
    
    if os.path.exists(rating_valid + '.npy'):
        rating_np_valid = np.load(rating_valid + '.npy')
    else:
        rating_np_valid = np.loadtxt(rating_valid + '.txt', dtype=np.int32)
        np.save(rating_valid + '.npy', rating_np_valid)

    if os.path.exists(rating_test + '.npy'):
        rating_np_test = np.load(rating_test + '.npy')
    else:
        rating_np_test = np.loadtxt(rating_test + '.txt', dtype=np.int32)
        np.save(rating_test + '.npy', rating_np_test)

    n_user = len(set.union(set(rating_np_train[:, 0]), set(rating_np_valid[:, 0]), set(rating_np_test[:, 0])))
    n_item = len(set.union(set(rating_np_train[:, 1]), set(rating_np_valid[:, 1]),set(rating_np_test[:, 1])))

    return n_user, n_item, rating_np_train, rating_np_valid, rating_np_test

def load_kg(args, user_review_dict, item_review_dict):
    print('reading kg ...')
    pos_neg_kg_file = args.TPS_DIR + '/pos_neg_kg_edges.txt'
    if os.path.exists(pos_neg_kg_file):
        pos_neg_kg_edges = np.array(np.loadtxt(pos_neg_kg_file))
    else:
        pos_neg_kg_edges = []
        
        review_entity_list = np.array(np.load(os.path.join(args.TPS_DIR, 'review_entity_list.npy')))
        kg_file = args.TPS_DIR + 'kg_final'
        writer_pos_neg_kg = open(pos_neg_kg_file, 'w', encoding='utf-8')
        if os.path.exists(kg_file + '.npy'):
            kg = np.load(kg_file + '.npy')
        else:
            kg = np.loadtxt(kg_file + '.txt', dtype=np.int32)
            np.save(kg_file + '.npy', kg)
        
        def sample_neg_triples_for_h(h, review_dict, num):
            neg_ts = []
            while True:
                if len(neg_ts) == num: break
                neg_t_list = np.setdiff1d(review_entity_list, h)
                neg_t= np.random.choice(neg_t_list, 1)[0]
                if neg_t not in review_dict[h] and neg_t not in neg_ts:
                    neg_ts.append(neg_t)
            return neg_ts
        
        for kg_line in kg:
            if kg_line[0] in user_review_dict.keys():
                neg_u_edge = sample_neg_triples_for_h(kg_line[0], user_review_dict, 1)
                pos_neg_kg_edges.append([kg_line[0], kg_line[1], neg_u_edge[0]])
                writer_pos_neg_kg.write('%d\t%d\t%d\n' % (kg_line[0], kg_line[1], neg_u_edge[0]))
                
            if kg_line[0] in item_review_dict.keys():
                neg_i_edge = sample_neg_triples_for_h(kg_line[0], item_review_dict, 1)
                pos_neg_kg_edges.append([kg_line[0], kg_line[1], neg_i_edge[0]])
                writer_pos_neg_kg.write('%d\t%d\t%d\n' % (kg_line[0], kg_line[1], neg_i_edge[0]))

        pos_neg_kg_edges = np.array(pos_neg_kg_edges)
        writer_pos_neg_kg.close()

    # obtained from preprocess: the number of nodes of the graph.
    n_entity = args.n_entity
    return n_entity, pos_neg_kg_edges
