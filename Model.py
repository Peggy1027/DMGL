import numpy as np
from time import time
import pickle
import sys
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class MyModel(nn.Module):
    def __init__(self, config, args, device):
        super(MyModel, self).__init__()
        self.device = device
        self.n_users = config['n_users']
        self.n_items = config['n_items']
        self.image_feats = torch.tensor(config['image_feats']).float().to(device)
        self.text_feats = torch.tensor(config['text_feats']).float().to(device)

        self.emb_dim = args.embed_size      
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)       
        self.weight_size = [self.emb_dim] + self.weight_size
        
        self.feat_conv_layers = args.feat_conv_layers       
        self.id_conv_layers = args.id_conv_layers           
        self.id_reg_decay = args.id_reg_decay               
        self.feat_reg_decay = args.feat_reg_decay           
        self.cl_decay = args.cl_decay                       

        self.feat_aug_rate = args.feat_aug_rate            
        # transformer
        self.head_num = args.head_num
        self.model_cat_rate = args.model_cat_rate           

        self.id_cat_rate = args.id_cat_rate                 
        self.topk_rate = args.topk_rate
        self.recon_rate = args.recon_rate                   

        self.user_id_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.emb_dim)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.ui_graph = self.ui_graph_raw = config['ui_graph_tr']
        self.iu_graph = self.iu_graph_raw = self.ui_graph.T

        self.ui_graph_raw = self.matrix_to_tensor(self.ui_graph_raw)
        self.iu_graph_raw = self.matrix_to_tensor(self.iu_graph_raw)
        self.ui_graph = self.matrix_to_tensor(self.csr_norm(self.ui_graph, mean_flag=True))
        self.iu_graph = self.matrix_to_tensor(self.csr_norm(self.iu_graph, mean_flag=True))
        self.image_ui_graph = self.text_ui_graph = self.ui_graph.coalesce()
        self.image_iu_graph = self.text_iu_graph = self.iu_graph.coalesce()

        hidden_dim = args.hidden_dim
        enc_hidden_dim = hidden_dim
        self.image_trans1 = nn.Linear(config['image_feats'].shape[1], enc_hidden_dim)
        self.text_trans1 = nn.Linear(config['text_feats'].shape[1], enc_hidden_dim)
        self.image_trans2 = nn.Linear(enc_hidden_dim, self.emb_dim)
        self.text_trans2 = nn.Linear(enc_hidden_dim, self.emb_dim)
        
        nn.init.xavier_uniform_(self.image_trans1.weight)
        nn.init.xavier_uniform_(self.text_trans1.weight) 
        nn.init.xavier_uniform_(self.image_trans2.weight)
        nn.init.xavier_uniform_(self.text_trans2.weight) 

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=args.drop_rate)
        
        self.sparse = args.sparse
        self.tau = args.tau

        # initializer = nn.init.xavier_uniform_
        # self.weight_dict = nn.ParameterDict({
        #     'w_q': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
        #     'w_k': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
        #     'w_v': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
        #     'w_self_attention_item': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
        #     'w_self_attention_user': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
        #     'w_self_attention_cat': nn.Parameter(initializer(torch.empty([args.head_num*args.embed_size, args.embed_size]))),
        # })
        self.embedding_dict = {'user':{}, 'item':{}}
        
    def para_dict_to_tenser(self, para_dict):  
        """
        :param para_dict: nn.ParameterDict()
        :return: tensor
        """
        tensors = []

        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = torch.stack(tensors, dim=0)     

        return tensors
    
    def multi_head_self_attention(self, trans_w, embedding_t_1, embedding_t):  
       
        q = self.para_dict_to_tenser(embedding_t)
        v = k = self.para_dict_to_tenser(embedding_t_1)
        beh, N, d_h = q.shape[0], q.shape[1], self.emb_dim/self.head_num

        Q = torch.matmul(q, trans_w['w_q'])  
        K = torch.matmul(k, trans_w['w_k'])
        V = v

        Q = Q.reshape(beh, N, self.head_num, int(d_h)).permute(2, 0, 1, 3)  
        K = Q.reshape(beh, N, self.head_num, int(d_h)).permute(2, 0, 1, 3)

        Q = torch.unsqueeze(Q, 2) 
        K = torch.unsqueeze(K, 1)  
        V = torch.unsqueeze(V, 1)  

        att = torch.mul(Q, K) / torch.sqrt(torch.tensor(d_h))  
        att = torch.sum(att, dim=-1) 
        att = torch.unsqueeze(att, dim=-1)  
        att = F.softmax(att, dim=2)  

        Z = torch.mul(att, V)  
        Z = torch.sum(Z, dim=2)  

        Z_list = [value for value in Z]
        Z = torch.cat(Z_list, -1)
        Z = torch.matmul(Z, self.weight_dict['w_self_attention_cat'])

        return Z, att.detach()


    def forward(self, new_image_ui_graph, new_image_iu_graph, new_text_ui_graph, new_text_iu_graph, epoch):
        
        image_user_id = text_user_id = self.user_id_embedding.weight
        image_item_id = text_item_id = self.item_id_embedding.weight

        for i in range(self.id_conv_layers):
            # image id_emb
            image_user_id = self.mm(new_image_ui_graph, image_item_id)
            image_item_id = self.mm(new_image_iu_graph, image_user_id)

            # text id_emb
            text_user_id = self.mm(new_text_ui_graph, text_item_id)
            text_item_id = self.mm(new_text_iu_graph, text_user_id)

        self.embedding_dict['user']['image'] = image_user_id
        self.embedding_dict['user']['text'] = text_user_id
        self.embedding_dict['item']['image'] = image_item_id
        self.embedding_dict['item']['text'] = text_item_id
        # multi-head
        # user_z, _ = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['user'], self.embedding_dict['user'])
        # item_z, _ = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['item'], self.embedding_dict['item'])
        # user_emb
        # user_emb = user_z.mean(0)
        # item_emb = item_z.mean(0)
        user_emb = 0.5 * image_user_id + 0.5 * text_user_id
        item_emb = 0.5 * image_item_id + 0.5 * text_item_id

        u_g_embeddings = self.user_id_embedding.weight + self.id_cat_rate*F.normalize(user_emb, p=2, dim=1)
        i_g_embeddings = self.item_id_embedding.weight + self.id_cat_rate*F.normalize(item_emb, p=2, dim=1)

        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]

        for i in range(self.n_layers):    
            if i == (self.n_layers-1):
                u_g_embeddings = self.softmax( torch.mm(self.ui_graph, i_g_embeddings) ) 
                i_g_embeddings = self.softmax( torch.mm(self.iu_graph, u_g_embeddings) )

            else:
                u_g_embeddings = torch.mm(self.ui_graph, i_g_embeddings) 
                i_g_embeddings = torch.mm(self.iu_graph, u_g_embeddings) 

            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        u_g_embeddings = torch.mean(torch.stack(user_emb_list), dim=0)
        i_g_embeddings = torch.mean(torch.stack(item_emb_list), dim=0)

        return u_g_embeddings, i_g_embeddings, image_user_id, text_user_id, image_item_id, text_item_id
    
    def bpr_loss(self, users, pos_items, neg_items, batch_size):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()        
        regularizer = regularizer / batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        reg_loss = self.id_reg_decay * regularizer
        
        return mf_loss, reg_loss
    
    def feat_reg_loss_calculation(self, item_image, item_text, user_image, user_text):
        item_feat_reg = 1./2*(item_image**2).sum() + 1./2*(item_text**2).sum()
        user_feat_reg = 1./2*(user_image**2).sum() + 1./2*(user_text**2).sum()        
        feat_reg = item_feat_reg / self.n_items + user_feat_reg / self.n_users
        feat_reg_loss = self.feat_reg_decay * feat_reg
        return feat_reg_loss
    
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    
    def batched_contrastive_loss(self, z1, z2, batch_size=1024):

        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)   #       

        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            tmp_i = indices[i * batch_size:(i + 1) * batch_size]

            tmp_refl_sim_list = []
            tmp_between_sim_list = []
            for j in range(num_batches):
                tmp_j = indices[j * batch_size:(j + 1) * batch_size]
                tmp_refl_sim = f(self.sim(z1[tmp_i], z1[tmp_j]))  
                tmp_between_sim = f(self.sim(z1[tmp_i], z2[tmp_j]))  

                tmp_refl_sim_list.append(tmp_refl_sim)
                tmp_between_sim_list.append(tmp_between_sim)

            refl_sim = torch.cat(tmp_refl_sim_list, dim=-1)
            between_sim = torch.cat(tmp_between_sim_list, dim=-1)

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()/ (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())+1e-8))

            del refl_sim, between_sim, tmp_refl_sim_list, tmp_between_sim_list
                   
        loss_vec = torch.cat(losses)
        return loss_vec.mean()
    
    def graph_learner(self, users, image_user_feats, image_item_feats, text_user_feats, text_item_feats, batch_size):
        uni_users = torch.unique(users)
        with torch.no_grad():
            batch_image_sim = self.sim_calculation(uni_users, image_user_feats, image_item_feats, batch_size)
            batch_text_sim = self.sim_calculation(uni_users, text_user_feats, text_item_feats, batch_size)
        
        old_ui_graph = self.ui_graph_raw
        new_image_ui_graph = self.build_graph(uni_users, batch_image_sim, old_ui_graph)
        new_image_iu_graph = self.sp_T(new_image_ui_graph)

        new_text_ui_graph = self.build_graph(uni_users, batch_text_sim, old_ui_graph)
        new_text_iu_graph = self.sp_T(new_text_ui_graph)
        
        return new_image_ui_graph, new_image_iu_graph, new_text_ui_graph, new_text_iu_graph
    
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).to(self.device)
    
    def sim_calculation(self, users, user_feats, item_feats, batch_size):
        batch_u = user_feats[users]

        num_batches = (self.n_items - 1) // batch_size + 1     
        indices = torch.arange(0, self.n_items).to(self.device)         
        sim_list = []

        for i_b in range(num_batches):
            index = indices[i_b * batch_size:(i_b + 1) * batch_size]      
            sim = torch.mm(batch_u, item_feats[index].T)
                      
            sim_list.append(sim)
                
        batch_ui_sim = F.normalize(torch.cat(sim_list, dim=-1), p=2, dim=1)   
        return batch_ui_sim
    
    def calculate_loss(self, users, pos_items, neg_items, image_user_feats, image_item_feats, text_user_feats, text_item_feats, batch_size, epoch):
        image_item_f = self.image_trans1(self.image_feats)
        text_item_f = self.text_trans1(self.text_feats)

        for i in range(self.feat_conv_layers):
        
            image_user_f = self.mm(self.ui_graph, image_item_f)
            image_item_f = self.mm(self.iu_graph, image_user_f)
            # text
            text_user_f = self.mm(self.ui_graph, text_item_f)
            text_item_f = self.mm(self.iu_graph, text_user_f)
        
        image_user_feats = self.feat_aug_rate * image_user_f + (1-self.feat_aug_rate) * image_user_feats
        image_item_feats = self.feat_aug_rate * image_item_f + (1-self.feat_aug_rate) * image_item_feats
        text_user_feats = self.feat_aug_rate * text_user_f + (1-self.feat_aug_rate) * text_user_feats
        text_item_feats = self.feat_aug_rate * text_item_f + (1-self.feat_aug_rate) * text_item_feats
        
        # encoder
        image_user_feats = self.dropout(self.image_trans2(image_user_feats))
        image_item_feats = self.dropout(self.image_trans2(image_item_feats))
        text_user_feats = self.dropout(self.text_trans2(text_user_feats))
        text_item_feats = self.dropout(self.text_trans2(text_item_feats))

        new_image_ui_graph, new_image_iu_graph, new_text_ui_graph, new_text_iu_graph = self.graph_learner(users, image_user_feats, image_item_feats, text_user_feats, text_item_feats, batch_size)
        
        # new_image_ui_graph, new_image_iu_graph, new_text_ui_graph, new_text_iu_graph = self.ui_graph, self.iu_graph, self.ui_graph, self.iu_graph
        (ua_embeddings, ia_embeddings, image_user_id, text_user_id, image_item_id, text_item_id) = self.forward(new_image_ui_graph, new_image_iu_graph, new_text_ui_graph, new_text_iu_graph, epoch)
        
        uaf_embeddings = ua_embeddings + self.model_cat_rate*F.normalize(image_user_feats, p=2, dim=1) + self.model_cat_rate*F.normalize(text_user_feats, p=2, dim=1)
        iaf_embeddings = ia_embeddings + self.model_cat_rate*F.normalize(image_item_feats, p=2, dim=1) + self.model_cat_rate*F.normalize(text_item_feats, p=2, dim=1)

        u_g_embeddings = uaf_embeddings[users]
        pos_i_g_embeddings = iaf_embeddings[pos_items]
        neg_i_g_embeddings = iaf_embeddings[neg_items]

        # bpr loss, id reg
        batch_mf_loss, batch_id_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, batch_size)

        # feat reg
        batch_feat_reg_loss = self.feat_reg_loss_calculation(image_item_feats, text_item_feats, image_user_feats, text_user_feats)

        # cl
        uni_u = torch.unique(users)
        batch_contrastive_loss = 0
        batch_contrastive_loss1 = self.batched_contrastive_loss(image_user_id[uni_u], ua_embeddings[uni_u])
        batch_contrastive_loss2 = self.batched_contrastive_loss(text_user_id[uni_u], ua_embeddings[uni_u])
        batch_contrastive_loss = batch_contrastive_loss1 + batch_contrastive_loss2

        batch_loss = batch_mf_loss + batch_id_reg_loss + batch_feat_reg_loss + self.cl_decay * batch_contrastive_loss
        
        return batch_loss, batch_mf_loss, batch_id_reg_loss, batch_feat_reg_loss, self.cl_decay * batch_contrastive_loss

    def mm(self, x, y):
        if self.sparse:
            return torch.sparse.mm(x, y)
        else:
            return torch.mm(x, y)

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum+1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)

        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum+1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)

        if mean_flag == False:
            return rowsum_diag*csr_mat*colsum_diag
        else:
            return rowsum_diag*csr_mat

    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  
        values = torch.from_numpy(cur_matrix.data)
        shape = torch.Size(cur_matrix.shape)

        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).to(self.device)

    def build_graph(self, users, sim_adj, old_graph):
        ui_val, ui_ind = torch.topk(sim_adj, int(self.n_items*self.topk_rate), dim=-1)
        tuple_list = [[row, int(col)] for row in range(len(ui_ind)) for col in ui_ind[row]]
        row = [users[i[0]] for i in tuple_list]
        col = [i[1] for i in tuple_list]
        indices = torch.LongTensor([row, col])
        values = torch.ones(len(row))
        # print(row)
        # print(col)
        
        shape = self.ui_graph.size()
        new_graph_tmp = torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).to(self.device)
        new_graph = self.sp_add(old_graph, new_graph_tmp)
        return new_graph

    def sp_T(self, sparse_graph):
        sparse_graph = sparse_graph.coalesce()
        indices = sparse_graph.indices()
        values = sparse_graph.values()
        shape = sparse_graph.size()

        trans_indices = torch.flip(indices, [0])
        trans_shape = torch.Size([shape[1], shape[0]])

        trans_graph = torch.sparse.FloatTensor(trans_indices, values, trans_shape).to(torch.float32).to(self.device)

        return trans_graph
    
    def sp_norm(self, sparse_graph):
        sparse_graph = sparse_graph.coalesce()
        dense_graph = sparse_graph.to_dense()
        normalized_graph = F.normalize(dense_graph, p=2, dim=1)

        indices = normalized_graph.nonzero()
        values = normalized_graph[indices[:,0], indices[:,1]]
        size = normalized_graph.size()

        sparse_graph = torch.sparse.FloatTensor(indices.t(), values, size).to(torch.float32).to(self.device)
        return sparse_graph
        
    def sp_add(self, old_graph, new_graph):
        old_graph = old_graph.coalesce()
        new_graph = new_graph.coalesce()
        
        old_values = old_graph.values()
        old_indices = old_graph.indices()

        new_values = new_graph.values()
        new_indices = new_graph.indices()

        shape = old_graph.size()

        values = torch.cat((self.recon_rate * old_values, (1-self.recon_rate) * new_values), dim = -1)
        indices = torch.cat((old_indices, new_indices), dim = -1)

        graph = torch.sparse.FloatTensor(indices, values, shape).coalesce().to(torch.float32).to(self.device)
        
        return self.sp_norm(graph)

    def full_predict(self, image_user_feats, image_item_feats, text_user_feats, text_item_feats, epoch):
        image_item_f = self.image_trans1(self.image_feats)
        text_item_f = self.text_trans1(self.text_feats)

        for i in range(self.feat_conv_layers):
            image_user_f = self.mm(self.ui_graph, image_item_f)
            image_item_f = self.mm(self.iu_graph, image_user_f)
            # text
            text_user_f = self.mm(self.ui_graph, text_item_f)
            text_item_f = self.mm(self.iu_graph, text_user_f)
        
        image_user_feats = self.feat_aug_rate * image_user_f + (1-self.feat_aug_rate) * image_user_feats
        image_item_feats = self.feat_aug_rate * image_item_f + (1-self.feat_aug_rate) * image_item_feats
        text_user_feats = self.feat_aug_rate * text_user_f + (1-self.feat_aug_rate) * text_user_feats
        text_item_feats = self.feat_aug_rate * text_item_f + (1-self.feat_aug_rate) * text_item_feats
        
        # encoder
        image_user_feats = self.image_trans2(image_user_feats)
        image_item_feats = self.image_trans2(image_item_feats)
        text_user_feats = self.text_trans2(text_user_feats)
        text_item_feats = self.text_trans2(text_item_feats)
        
        users = torch.LongTensor(range(self.n_users))
        new_image_ui_graph, new_image_iu_graph, new_text_ui_graph, new_text_iu_graph = self.graph_learner(users, image_user_feats, image_item_feats, text_user_feats, text_item_feats, batch_size = 1024)
        # new_image_ui_graph, new_image_iu_graph, new_text_ui_graph, new_text_iu_graph = self.ui_graph, self.iu_graph, self.ui_graph, self.iu_graph

        (ua_embeddings, ia_embeddings, image_user_id, text_user_id, image_item_id, text_item_id) = self.forward(new_image_ui_graph, new_image_iu_graph, new_text_ui_graph, new_text_iu_graph, epoch)
        
        ua_embeddings = ua_embeddings + self.model_cat_rate*F.normalize(image_user_feats, p=2, dim=1) + self.model_cat_rate*F.normalize(text_user_feats, p=2, dim=1)
        ia_embeddings = ia_embeddings + self.model_cat_rate*F.normalize(image_item_feats, p=2, dim=1) + self.model_cat_rate*F.normalize(text_item_feats, p=2, dim=1)

        return ua_embeddings, ia_embeddings


