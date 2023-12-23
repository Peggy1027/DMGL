# import utils as metrics
from .utils import auc, precision_at_k, recall_at_k, ndcg_at_k, hit_at_k
from functools import partial
import multiprocessing
import heapq
import torch
import random
import numpy as np
import sys

cores = multiprocessing.cpu_count() // 5

class Test(object):
    def __init__(self, config, args):
        self.Ks = eval(args.Ks)
        self.batch_size = args.batch_size
        self.batch_test_flag = args.batch_test_flag      
        self.test_flag = args.test_flag
        self.n_candidate = args.n_candidate

        self.n_users = config['n_users']
        self.n_items = config['n_items']
        self.all_set = config['all_set']
        self.train_set = config['train_set']
        self.val_set = config['val_set']
        self.test_set = config['test_set']
        self.val_test_set = config['val_test_set']
    
    def test_predict(self, ua_embeddings, ia_embeddings, is_val):
        result = {'precision': np.zeros(len(self.Ks)), 'recall': np.zeros(len(self.Ks)), 'ndcg': np.zeros(len(self.Ks)),
                    'hit_ratio': np.zeros(len(self.Ks)), 'auc': 0.}
        pool = multiprocessing.Pool(cores)    
        
        # u_batch_size = self.batch_size * 2
        # i_batch_size = self.batch_size

        # if self.with_test == False:
        #     test_users = list(self.val_test_set.keys())
        if is_val:
            test_users = list(self.val_set.keys())
        else:
            test_users = list(self.test_set.keys())
        n_test_users = len(test_users)

        ia_embeddings = ia_embeddings.detach().cpu().numpy()
        ua_embeddings = ua_embeddings.detach().cpu().numpy()
        p_func = partial(self.test_one_user, ia_embeddings, ua_embeddings)

        test_users_uid = zip(test_users, [is_val]* len(test_users))
        batch_result = pool.map(p_func, test_users_uid)

        for re in batch_result:
            result['precision'] += re['precision'] / n_test_users
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            result['auc'] += re['auc'] / n_test_users
        pool.close()
        return result
    
    def test_one_user(self, ia_embeddings, ua_embeddings, x):
        # user u's ratings for user u
        is_val = x[-1]
        #uid
        u = x[0]
        
        #user u's items in the test set
        if is_val:
            user_pos_test = self.val_set[u]
        else:
            user_pos_test = self.test_set[u]

        test_items = self.sample_test_items(u, user_pos_test)

        user_emb = ua_embeddings[u]
        item_emb = ia_embeddings[test_items]
        # print('user_emb', user_emb.shape)
        # print('item_emb', item_emb.shape)

        rating = user_emb * item_emb
        rating = np.mean(rating, axis=-1)

        if self.test_flag == 'part':
            r, auc = self.ranklist_by_heapq(user_pos_test, test_items, rating)
        else:
            r, auc = self.ranklist_by_sorted(user_pos_test, test_items, rating)

        return self.get_performance(user_pos_test, r, auc)
    
    def sample_test_items(self, u, user_pos_test):
        n_items = self.n_items
        n_candidate = self.n_candidate

        #user u's items in the training set
        try:
            all_items = self.all_set[u]
        except Exception:
            print("test user doesn't have items!")
            sys.exit()

        all_neg_items = list(set(range(n_items)) - set(all_items))
        if (len(all_neg_items)) <= n_candidate:
            test_items = user_pos_test + all_neg_items
        else:
            neg_items = random.sample(all_neg_items, n_candidate)
            # print("pos:",user_pos_test)
            # print("neg:",neg_items)
            test_items = user_pos_test + neg_items
        return test_items

    def ranklist_by_heapq(self, user_pos_test, test_items, rating):
        item_score = {}
        for i in range(len(test_items)):
            item_score[test_items[i]] = rating[i]
        # print(len(item_score))

        K_max = max(self.Ks)
        K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

        r = []
        for i in K_max_item_score:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = 0.
        return r, auc
    
    def ranklist_by_sorted(self, user_pos_test, test_items, rating):
        item_score = {}
        for i in range(len(test_items)):
            item_score[test_items[i]] = rating[i]

        K_max = max(self.Ks)
        K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

        r = []
        for i in K_max_item_score:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = self.get_auc(item_score, user_pos_test)
        return r, auc
    
    def get_auc(item_score, user_pos_test):
        item_score = sorted(item_score.items(), key=lambda kv: kv[1])
        item_score.reverse()
        item_sort = [x[0] for x in item_score]
        posterior = [x[1] for x in item_score]

        r = []
        for i in item_sort:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = auc(ground_truth=r, prediction=posterior)
        return auc
    
    def get_performance(self, user_pos_test, r, auc):
        precision, recall, ndcg, hit_ratio = [], [], [], []

        for K in self.Ks:
            precision.append(precision_at_k(r, K))
            recall.append(recall_at_k(r, K, len(user_pos_test)))
            ndcg.append(ndcg_at_k(r, K))
            hit_ratio.append(hit_at_k(r, K))

        return {'recall': np.array(recall), 'precision': np.array(precision),
                'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}