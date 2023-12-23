
import numpy as np
import random as rd
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
from time import time
import json

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        # user:[i1,i2,i3...]
        train_file = path + '/train.json'
        val_file = path + '/val.json'
        test_file = path + '/test.json'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test, self.n_val = 0, 0, 0
        self.neg_pools = {}

        self.exist_users = []

        train = json.load(open(train_file))
        test = json.load(open(test_file))
        val = json.load(open(val_file))
        for uid, items in train.items():
            if len(items) == 0:
                continue
            uid = int(uid)
            self.exist_users.append(uid)
            self.n_items = max(self.n_items, max(items))
            self.n_users = max(self.n_users, uid)
            self.n_train += len(items)

        for uid, items in test.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_test += len(items)
            except:
                continue

        for uid, items in val.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_val += len(items)
            except:
                continue

        self.n_items += 1
        self.n_users += 1

        # self.print_statistics()
        # end of get statistics

        self.R_tr = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_val = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        # self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        # self.R_Item_Interacts = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)

        self.all_set, self.train_set, self.test_set, self.val_set, self.val_test_set = {}, {}, {}, {}, {}
        for uid, train_items in train.items():
            if len(train_items) == 0:
                continue
            uid = int(uid)
            for idx, i in enumerate(train_items):
                self.R_tr[uid, i] = 1.
                self.R_val[uid, i] = 1.

            self.train_set[uid] = train_items
            self.all_set[uid] = train_items

        for uid, test_items in test.items():
            uid = int(uid)
            if len(test_items) == 0 or uid not in self.all_set.keys():
                continue
            try:
                self.test_set[uid] = test_items
                self.val_test_set[uid] = test_items
                self.all_set[uid] = self.all_set[uid] + test_items
            except:
                print('wrong1!')
                continue

        for uid, val_items in val.items():
            uid = int(uid)
            if len(val_items) == 0 or uid not in self.all_set.keys():
                continue
            try:
                self.val_set[uid] = val_items
                self.all_set[uid] = self.all_set[uid] + val_items
                if uid in self.val_test_set.keys():
                    self.val_test_set[uid] = self.val_test_set[uid] + val_items
                else:
                    self.val_test_set[uid] = val_items
                for idx, i in enumerate(val_items):
                    self.R_val[uid, i] = 1.
            except:
                print("wrong2!")
                continue              
    
    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat_tr = sp.load_npz(self.path + '/adj_mat_tr.npz')
            adj_mat_val = sp.load_npz(self.path + '/adj_mat_val.npz')
            A_tr = sp.load_npz(self.path + '/ui_iu_adj_mat_tr.npz')
            A_val = sp.load_npz(self.path + '/ui_iu_adj_mat_val.npz')
            # norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            # mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat_tr.shape, A_tr.shape)

        except Exception:
            adj_mat_tr, adj_mat_val, A_tr, A_val = self.create_adj_mat()
            sp.save_npz(self.path + '/adj_mat_tr.npz', adj_mat_tr)
            sp.save_npz(self.path + '/adj_mat_val.npz', adj_mat_val)
            sp.save_npz(self.path + '/ui_iu_adj_mat_tr.npz', A_tr)
            sp.save_npz(self.path + '/ui_iu_adj_mat_val.npz', A_val)

            # print('already create adjacency matrix', adj_mat.shape)
        return adj_mat_tr, adj_mat_val, A_tr, A_val

    def create_adj_mat(self):
        t1 = time()
        adj_mat_tr = self.R_tr.tocsr()
        adj_mat_val = self.R_val.tocsr()

        A_tr = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        A_val = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        A_tr = A_tr.tolil()
        A_val = A_val.tolil()
        R_tr = self.R_tr.tolil()
        R_val = self.R_val.tolil()

        A_tr[:self.n_users, self.n_users:] = R_tr
        A_tr[self.n_users:, :self.n_users] = R_tr.T
        A_val[:self.n_users, self.n_users:] = R_val
        A_val[self.n_users:, :self.n_users] = R_val.T
        
        A_tr = A_tr.todok()
        A_val = A_val.todok()
        print('already create adjacency matrix', adj_mat_tr.shape, A_tr.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def get_D_inv(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            return d_mat_inv

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        mean_A_tr = normalized_adj_single(A_tr+ sp.eye(A_tr.shape[0]))
        mean_A_val = normalized_adj_single(A_val+ sp.eye(A_val.shape[0]))

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat_tr.tocsr(), adj_mat_val.tocsr(), mean_A_tr.tocsr(), mean_A_val.tocsr()

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        # users = self.exist_users[:]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_set[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_set[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u(u, 3)
        return users, pos_items, neg_items
        

    # def print_statistics(self):
    #     print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
    #     print('n_interactions=%d' % (self.n_train + self.n_test))
    #     print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))

class myTrainset(Dataset):
    def __init__(self, config, neg):
        self.n_items = config['n_items']
        self.n_users = config['n_users']
        self.all_set = config['all_set']
        self.neg = neg
        train_data = config['train_set']
        self.data_trans = self.trans(train_data)

    def trans(self, train_data):
        data_trans = []
        for uid in train_data:
            for item in train_data[uid]:
                data_trans.append([uid, item])
        data_trans = np.array(data_trans)
        return data_trans
    
    def __getitem__(self, index):
        user, pos_item = self.data_trans[index][0], self.data_trans[index][1]
        neg_item = np.empty(self.neg, dtype=np.int32)
        for idx in range(self.neg):
            t = np.random.randint(0, self.n_items)
            while t in self.all_set[user]:
                t = np.random.randint(0, self.n_items)
            neg_item[idx] = t
        return user, pos_item, neg_item
    
    def __len__(self):
        return len(self.data_trans)
    
def get_train_loader(config, args):
    dataset = myTrainset(config, 1)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataloader