import argparse
import numpy as np
from numpy.lib.shape_base import split
import pandas as pd
import os

RATING_FILE_NAME = dict({'imdb':'ratings.dat',
                         'amazon':'ratings.dat'})
SEP = dict({'imdb': ',', 'amazon': ','})
THRESHOLD = dict({'imdb':8, 'amazon':4})
# the direction of dataset.
TPS_DIR = '../data/amazon/10-core/'

# fusing users/items and their corresponding review entities into a heterogeneous graph.
def convert_review_to_entity_id_file():
    file_user = TPS_DIR + 'review_user2entity.txt'
    print('reading user index to entity id file: ' + file_user + ' ...')
    file_item = TPS_DIR + 'review_item2entity.txt'
    print('reading item index to entity id file: ' + file_item+ ' ...')
    file_item_id = TPS_DIR + 'item_index2entity_id.txt'
    print('reading item id index to entity id file: ' + file_item_id+ ' ...')
    file_user_id = TPS_DIR + 'user_index2entity_id.txt'
    print('reading user id index to entity id file: ' + file_user_id+ ' ...')
    i = 0
    user_list = []
    item_list = []
    user_entity_list = []
    item_entity_list = []
    user4kg_ids = []
    item4kg_ids = []
    user_review_entity_list = []
    item_review_entity_list = []
    for item_id_line in open(file_item_id, encoding='utf-8').readlines():
        item_id = item_id_line.strip().split(',')[0]
        if 'i' + item_id not in entity_id2index:
            item_index_old2new[item_id] = i
            entity_id2index['i' + item_id] = i
            # entity_index2id[i] = item_id
            i += 1
        item_list.append(item_id)
        item_entity_list.append(entity_id2index['i'+item_id])
    items_num = i
    print("the number of item is: ", items_num)
    for item_line in open(file_item, encoding='utf-8').readlines():
        item_index = item_line.strip().split(',')[0]
        item_satori_id = item_line.strip().split(',', 1)[1]
        if item_satori_id != '':
            item_satori_id = item_satori_id.strip('\"').split(',')
            for entity in item_satori_id:
                if entity not in entity_id2index:
                    entity_id2index[entity] = i
                    # entity_index2id[i] = entity
                    i+=1
                item4kg_ids.append(item_index_old2new[item_index])
                item_review_entity_list.append(entity_id2index[entity])

    item_review_convert = pd.DataFrame({'item_entity_inx': pd.Series(item4kg_ids),
                                'review_entity_index': pd.Series(item_review_entity_list)})[['item_entity_inx', 'review_entity_index']]
    item_id_convert = pd.DataFrame({'item_id': pd.Series(item_list),
                                'entity_index': pd.Series(item_entity_list)})[['item_id', 'entity_index']]
    item_review_convert.to_csv(os.path.join(TPS_DIR, 'review_item2entity_final.txt'), index=False, header=None)
    item_id_convert.to_csv(os.path.join(TPS_DIR, 'item2entity_final.txt'), index=False, header=None)
    user_start_index = i
    user_start_index_doc = open(TPS_DIR + '/user_start_index.txt', 'w', encoding='utf-8')
    user_start_index_doc.write('%d' % (user_start_index))
    user_start_index_doc.close()

    start_user_inx = i
    for user_id_line in open(file_user_id, encoding='utf-8').readlines():
        user_id = user_id_line.strip().split(',')[0]
        if 'u' + user_id not in entity_id2index:
            user_index_old2new[user_id] = i
            entity_id2index['u' + user_id] = i
            i += 1
        user_list.append(user_id)
        user_entity_list.append(entity_id2index['u'+user_id])
    users_num = i-start_user_inx
    print("the number of users is: ", users_num)
    for user_line in open(file_user, encoding='utf-8').readlines():
        user_index = user_line.strip().split(',')[0]
        satori_id = user_line.strip().split(',', 1)[1]
        if satori_id !='':
            satori_id = satori_id.strip('\"').split(',')
            for user_satori_id in satori_id:
                if user_satori_id not in entity_id2index:
                    entity_id2index[user_satori_id] = i
                    i += 1
                user4kg_ids.append(user_index_old2new[user_index])
                user_review_entity_list.append(entity_id2index[user_satori_id])

    user_review_convert = pd.DataFrame({'user_entity_inx': pd.Series(user4kg_ids),
                                'review_entity_index': pd.Series(user_review_entity_list)})[['user_entity_inx', 'review_entity_index']]
    user_id_convert = pd.DataFrame({'user_id': pd.Series(user_list),
                                'entity_index': pd.Series(user_entity_list)})[['user_id', 'entity_index']]
    user_review_convert.to_csv(os.path.join(TPS_DIR, 'review_user2entity_final.txt'), index=False, header=None)
    user_id_convert.to_csv(os.path.join(TPS_DIR, 'user2entity_final.txt'), index=False, header=None)
    np.save(os.path.join(TPS_DIR, 'entity_id2index.npy'), entity_id2index)

    # combine all of the user review entity idexes and item review entity indexes into an array for kg edges negative sampling.
    user_review_entity_list.extend(item_review_entity_list)
    review_entity_list = list(set(user_review_entity_list))
    if 'PAD' in review_entity_list:
        review_entity_list.remove('PAD')
    np.save(os.path.join(TPS_DIR, 'review_entity_list.npy'), review_entity_list)
    
    return users_num, items_num

# add by liu at 20200908
def convert_rating():
    file_train = TPS_DIR + '/amazon_train.dat'
    file_valid = TPS_DIR + '/amazon_valid.dat'
    file_test = TPS_DIR + '/amazon_test.dat'
    print('reading rating file...')
    writer_train = open(TPS_DIR + '/amazon_train_final.txt', 'w', encoding='utf-8')
    writer_valid = open(TPS_DIR + '/amazon_valid_final.txt', 'w', encoding='utf-8')
    writer_test = open(TPS_DIR + '/amazon_test_final.txt', 'w', encoding='utf-8')
    sample_num = 0
    for line in open(file_train, encoding='utf-8').readlines():
        array = line.strip().split(SEP[DATASET])
        user_index_old = array[0]
        item_index_old = array[1]
        # if item_index_old not in item_index_old2new:  # the item is not in the final item set
        #     continue
        item_index = item_index_old2new[item_index_old]
        user_index = user_index_old2new[user_index_old]
        rating = float(array[2])
        sample_num = sample_num + 1
        writer_train.write('%d\t%d\t%d\n' % (user_index, item_index, rating))
    writer_train.close()
    for line in open(file_test, encoding='utf-8').readlines():
        array = line.strip().split(SEP[DATASET])
        user_index_old = array[0]
        item_index_old = array[1]
        # if item_index_old not in item_index_old2new:  # the item is not in the final item set
        #     continue
        item_index = item_index_old2new[item_index_old]
        user_index = user_index_old2new[user_index_old]
        rating = float(array[2])
        sample_num = sample_num + 1
        writer_test.write('%d\t%d\t%d\n' % (user_index, item_index, rating))
    writer_test.close()
    for line in open(file_valid, encoding='utf-8').readlines():
        array = line.strip().split(SEP[DATASET])
        user_index_old = array[0]
        item_index_old = array[1]
        # if item_index_old not in item_index_old2new:  # the item is not in the final item set
        #     continue
        item_index = item_index_old2new[item_index_old]
        user_index = user_index_old2new[user_index_old]
        rating = float(array[2])
        sample_num = sample_num + 1
        writer_valid.write('%d\t%d\t%d\n' % (user_index, item_index, rating))
    writer_valid.close()
    print("sample number is: ", sample_num)

def convert_kg():
    print('converting kg.txt file ...')
    entity_cnt = len(entity_id2index)
    edge_num = 0
    kg_dict = dict()

    writer = open(TPS_DIR + '/kg_final.txt', 'w', encoding='utf-8')
    user_review_triple = open(os.path.join(TPS_DIR, 'review_user2entity_final.txt'), encoding='utf-8')
    item_review_triple = open(os.path.join(TPS_DIR, 'review_item2entity_final.txt'), encoding='utf-8')

    # add the triples of movie-review, user-reviews.
    for user_triple in user_review_triple:
        user_head = int(user_triple.strip().split(',')[0])
        user_review_tail = int(user_triple.strip().split(',')[1])
        if user_head not in kg_dict:
            kg_dict[user_head]=[]
        if user_review_tail not in kg_dict[user_head]:
            edge_num = edge_num + 1
            kg_dict[user_head].append(user_review_tail)
            writer.write('%d\t%d\n' % (user_head, user_review_tail))
    
    for item_triple in item_review_triple:
        item_head = int(item_triple.strip().split(',')[0])
        item_review_tail = int(item_triple.strip().split(',')[1])
        if item_head not in kg_dict:
            kg_dict[item_head] = []
        if item_review_tail not in kg_dict[item_head]:
            edge_num = edge_num + 1
            kg_dict[item_head].append(item_review_tail)

            writer.write('%d\t%d\n' % (item_head, item_review_tail))

    writer.close()
    print('number of graph nodes (containing users and items): %d' % entity_cnt)
    print("number of review entities: ", entity_cnt-users_num-items_num)
    print('number of edges: %d' % edge_num)


if __name__ == '__main__':
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='amazon', help='which dataset to preprocess')
    args = parser.parse_args()
    DATASET = args.d

    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()
    user_index_old2new = dict()
    tail_nodes = []

    users_num, items_num = convert_review_to_entity_id_file()
    convert_rating()
    convert_kg()

    print('done')
