import os
import pandas as pd
import dill as pickle
import numpy as np
import pymysql
import itertools
from collections import Counter
TPS_DIR = 'F:\\Study\\doctor_work\\Git\\copy_JPKG\\data\\amazon\\10-core\\'
np.random.seed(2022) # for now 5-core, 4-core, 3-core, 2-core are 2021random.

def read_movieid_identifier():
    movie_id_identifier = dict()
    movie_id_number = dict()

    db = pymysql.connect(host='localhost',
                             port=3306,
                             user='root',
                             password='302485',
                             db='imdb',
                             charset='utf8')
    cursor = db.cursor()
    search_sql = """select distinct d.movie_id, d.identifier, c.number from entity_emb c,
    (SELECT 
        a.movie_id, a.identifier
    FROM
        amazon_id2identifier a,
        (SELECT DISTINCT
            (movie_id) AS movie_id
        FROM
            amazon_dataset_id) b
    WHERE
        a.movie_id = b.movie_id and a.identifier != '') d where c.entity_id = d.identifier"""
    try:
        cursor.execute(search_sql)
    except Exception as e:
        db.rollback()
        print(str(e))
    finally:
        cursor.close()
        db.close()
    data_result = cursor.fetchall()
    for data in data_result:
        if data[0] not in movie_id_identifier.keys():
            movie_id_identifier[data[0]] = data[1]
        if data[0] not in movie_id_number.keys():
            movie_id_number[data[0]] = data[2]

    return movie_id_identifier, movie_id_number

def read_dataset():
    __, movie_id_number = read_movieid_identifier()
    users_id = []
    items_id = []
    ratings = []
    review_identifiers = []
    amazon_data=pd.DataFrame(columns = ['user_id', 'item_id', 'ratings', 'review_identifiers'])

    db = pymysql.connect(host='localhost',
                             port=3306,
                             user='root',
                             password='302485',
                             db='imdb',
                             charset='utf8')
    cursor = db.cursor()
    search_sql = """SELECT 
    d.user_id, d.movie_id, d.rating, d.identifiers
FROM
    (SELECT 
        b.user_id AS user_id,
            c.movie_id AS movie_id,
            c.rating AS rating,
            c.identifier_inx as identifiers
    FROM
        (SELECT 
        COUNT(*) AS repetitions, user_id
    FROM
        amazon4mkr
    GROUP BY user_id
    HAVING repetitions > 9) b, amazon4mkr c
    WHERE
        c.user_id = b.user_id) d"""

    try:
        cursor.execute(search_sql)
        data_result = cursor.fetchall()
        # 
        for data in data_result:
            users_id.append(data[0])
            items_id.append(data[1])
            nums = data[3].split(",")
            if data[1] in movie_id_number.keys() and movie_id_number[data[1]] not in nums:
                review_identifiers.append(str(movie_id_number[data[1]])+','+data[3])
            else:
                review_identifiers.append(data[3])
            ratings.append(data[2])

        amazon_data = pd.DataFrame({'user_id': pd.Series(users_id),
                                    'item_id': pd.Series(items_id),
                                    'ratings': pd.Series(ratings),
                                    'review_identifiers': pd.Series(review_identifiers)})
    except Exception as e:
        db.rollback()
        print(str(e))
    finally:
        cursor.close()
        db.close()
    amazon_data.to_csv(os.path.join(TPS_DIR, 'amazon.csv'),
                    index=False)
    return amazon_data

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
    read_dataset()
    # data = pd.read_csv(os.path.join(TPS_DIR, 'amazon.csv'))
    # usercount, itemcount = get_count(
    #     data, 'user_id'), get_count(data, 'item_id')
    # unique_uid = usercount.index
    # unique_sid = itemcount.index
    # item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    # user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
    # data = numerize(data, user2id, item2id)
    # # review_id2entity = numerize(review_id2entity, user2id, item2id)
    # tp_rating = data[['user_id', 'item_id', 'ratings']]

    # n_ratings = tp_rating.shape[0]
    # test = np.random.choice(n_ratings, size=int(
    #     0.20 * n_ratings), replace=False)
    # test_idx = np.zeros(n_ratings, dtype=bool)
    # test_idx[test] = True

    # tp_1 = tp_rating[test_idx]
    # tp_train = tp_rating[~test_idx]

    # data_test = data[test_idx]
    # data_train_valid = data[~test_idx]

    # n_ratings = tp_1.shape[0]
    # test = np.random.choice(n_ratings, size=int(
    #     0.50 * n_ratings), replace=False)

    # test_idx = np.zeros(n_ratings, dtype=bool)
    # test_idx[test] = True

    # tp_test = tp_1[test_idx]
    # tp_valid = tp_1[~test_idx]

    # tp_train.to_csv(os.path.join(TPS_DIR, 'amazon_train.dat'),
    #                 index=False, header=None)
    # tp_valid.to_csv(os.path.join(TPS_DIR, 'amazon_valid.dat'),
    #                 index=False, header=None)
    # tp_test.to_csv(os.path.join(TPS_DIR, 'amazon_test.dat'),
    #                 index=False, header=None)


    # item2entity_id = data[['item_id']].drop_duplicates()
    # user2entity_id = data[['user_id']].drop_duplicates()

    # review2entity = data_train_valid[['user_id', 'item_id', 'review_identifiers']].drop_duplicates()
    # review_user2entity = data_train_valid[['user_id', 'review_identifiers']].drop_duplicates()
    # review_item2entity = data_train_valid[['item_id', 'review_identifiers']].drop_duplicates()
    # review_users = list(set(review_user2entity['user_id'].values))
    # review_items = list(set(review_item2entity['item_id'].values))
    # for i in data_test.values:
    #     if i[0] not in review_users:
    #         # df.append({'lib': 2, 'qty1': 3, 'qty2': 4}, ignore_index=True)
    #         s_u = pd.Series([i[0], ''])
    #         s_u.index = ['user_id','review_identifiers']
    #         review_user2entity = review_user2entity.append(s_u,ignore_index=True)
    #         # review_user2entity = review_user2entity.append({'user_id':i[0], 'review_identifiers':''}, ignore_index=True)
    #     if i[1] not in review_items:
    #         s_i = pd.Series([i[1], ''])
    #         s_i.index = ['item_id','review_identifiers']
    #         review_item2entity = review_item2entity.append(s_i,ignore_index=True)
    #         # review_user2entity = review_item2entity.append({'item_id':i[1], 'review_identifiers':''}, ignore_index=True)

    # # tp_rating.to_csv(os.path.join(TPS_DIR, 'ratings.dat'), index=False, header=None)
    # item2entity_id.to_csv(os.path.join(TPS_DIR, 'item_index2entity_id.txt'), index=False, header=None)
    # user2entity_id.to_csv(os.path.join(TPS_DIR, 'user_index2entity_id.txt'), index=False, header=None)
    # # review2entity.to_csv(os.path.join(TPS_DIR, 'review2entity.txt'), index=False, header=None)
    # review_user2entity.to_csv(os.path.join(TPS_DIR, 'review_user2entity.txt'), index=False, header=None)
    # review_item2entity.to_csv(os.path.join(TPS_DIR, 'review_item2entity.txt'), index=False, header=None)

    # usercount, itemcount = get_count(
    #     data, 'user_id'), get_count(data, 'item_id')

    # print(np.sort(np.array(usercount.values)))

    # print(np.sort(np.array(itemcount.values)))
