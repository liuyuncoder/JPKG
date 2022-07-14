import os
import pandas as pd
import numpy as np
TPS_DIR = '../data/amazon/10-core/'
np.random.seed(2022)

def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=True)
    count = playcount_groupbyid.size()
    return count


def numerize(tp, user2id, item2id):
    uid = list(map(lambda x: user2id[x], tp['user_id']))
    sid = list(map(lambda x: item2id[x], tp['item_id']))
    tp['user_id'] = uid
    tp['item_id'] = sid
    return tp

if __name__ == '__main__':
    data = pd.read_csv(os.path.join(TPS_DIR, 'amazon.csv'))
    usercount, itemcount = get_count(
        data, 'user_id'), get_count(data, 'item_id')
    unique_uid = usercount.index
    unique_sid = itemcount.index
    item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
    data = numerize(data, user2id, item2id)
    tp_rating = data[['user_id', 'item_id', 'ratings']]

    n_ratings = tp_rating.shape[0]
    test = np.random.choice(n_ratings, size=int(
        0.20 * n_ratings), replace=False)
    test_idx = np.zeros(n_ratings, dtype=bool)
    test_idx[test] = True

    tp_1 = tp_rating[test_idx]
    tp_train = tp_rating[~test_idx]

    data_test = data[test_idx]
    data_train_valid = data[~test_idx]

    n_ratings = tp_1.shape[0]
    test = np.random.choice(n_ratings, size=int(
        0.50 * n_ratings), replace=False)

    test_idx = np.zeros(n_ratings, dtype=bool)
    test_idx[test] = True

    tp_test = tp_1[test_idx]
    tp_valid = tp_1[~test_idx]

    tp_train.to_csv(os.path.join(TPS_DIR, 'amazon_train.dat'),
                    index=False, header=None)
    tp_valid.to_csv(os.path.join(TPS_DIR, 'amazon_valid.dat'),
                    index=False, header=None)
    tp_test.to_csv(os.path.join(TPS_DIR, 'amazon_test.dat'),
                    index=False, header=None)


    item2entity_id = data[['item_id']].drop_duplicates()
    user2entity_id = data[['user_id']].drop_duplicates()

    review2entity = data_train_valid[['user_id', 'item_id', 'review_identifiers']].drop_duplicates()
    review_user2entity = data_train_valid[['user_id', 'review_identifiers']].drop_duplicates()
    review_item2entity = data_train_valid[['item_id', 'review_identifiers']].drop_duplicates()
    review_users = list(set(review_user2entity['user_id'].values))
    review_items = list(set(review_item2entity['item_id'].values))
    for i in data_test.values:
        if i[0] not in review_users:
            s_u = pd.Series([i[0], ''])
            s_u.index = ['user_id','review_identifiers']
            review_user2entity = review_user2entity.append(s_u,ignore_index=True)
        if i[1] not in review_items:
            s_i = pd.Series([i[1], ''])
            s_i.index = ['item_id','review_identifiers']
            review_item2entity = review_item2entity.append(s_i,ignore_index=True)

    item2entity_id.to_csv(os.path.join(TPS_DIR, 'item_index2entity_id.txt'), index=False, header=None)
    user2entity_id.to_csv(os.path.join(TPS_DIR, 'user_index2entity_id.txt'), index=False, header=None)
    review_user2entity.to_csv(os.path.join(TPS_DIR, 'review_user2entity.txt'), index=False, header=None)
    review_item2entity.to_csv(os.path.join(TPS_DIR, 'review_item2entity.txt'), index=False, header=None)

    usercount, itemcount = get_count(
        data, 'user_id'), get_count(data, 'item_id')

    print(np.sort(np.array(usercount.values)))

    print(np.sort(np.array(itemcount.values)))
