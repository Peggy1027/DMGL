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

from functools import partial

from .gcn import GCN
from .gat import GAT
from .utils import create_norm

def setup_module(m_type, enc_dec, in_dim, hidden_dim, out_dim, num_layers, dropout, n_head, activation, residual, norm):
    if m_type == 'gcn':
        modal = GCN(in_dim=in_dim, 
            hidden_dim=hidden_dim, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"))
    elif m_type == 'gat':
        modal = GAT(in_dim=in_dim, 
            hidden_dim=hidden_dim, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            n_head=n_head,
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"))
    else:
        raise NotImplementedError
    return modal

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

class Augmentation(nn.Module):
    def __init__(self, feat_type, config, args, device):
        super(Augmentation, self).__init__()
        self.device = device
        self.n_users = config['n_users']
        self.n_items = config['n_items']
        if feat_type == 'image':
            self.feats = torch.tensor(config['image_feats']).float().to(device)
        elif feat_type == 'text':
            self.feats = torch.tensor(config['text_feats']).float().to(device)

        self.A = config['A_tr']
        self.A = self.matrix_to_tensor(self.A)
        self.ui_graph = config['ui_graph_tr']
        self.iu_graph = self.ui_graph.T
        self.ui_graph = self.matrix_to_tensor(self.csr_norm(self.ui_graph, mean_flag=True))
        self.iu_graph = self.matrix_to_tensor(self.csr_norm(self.iu_graph, mean_flag=True))

        self.feat_conv_layers = args.feat_conv_layers

        self.encoder_type = args.encoder
        self.decoder_type = args.decoder
        self.loss_fn = args.loss_fn         # mse/sce
        # self.drop_edge_rate = 0.0
        self.mask_rate = args.mask_rate     # 0.3
        self.replace_rate = args.replace_rate       # 0.1

        self.hidden_dim = args.hidden_dim       # 1024
        
        self.enc_in_dim = self.feats.shape[1]
        self.enc_hidden_dim = self.hidden_dim
        
        self.n_head = args.n_head       # 4
        self.enc_num_layers = args.pre_layers       # 2
        self.feat_drop = args.in_drop       # 0.2
        self.activation = args.activation       # prelu
        self.residual = args.residual       # True
        self.norm = args.norm       # None

        self.dec_in_dim = self.hidden_dim
        self.dec_hidden_dim = self.hidden_dim
        self.dec_num_layers = args.dec_layers

        self.concate_hidden = args.concate_hidden       # True

        # build encoder
        self.encoder = setup_module(
            m_type=self.encoder_type,
            enc_dec="encoding",
            in_dim=self.enc_in_dim,
            hidden_dim=self.enc_hidden_dim,
            out_dim=self.enc_hidden_dim,
            num_layers=self.enc_num_layers,
            activation=self.activation,
            dropout=self.feat_drop,
            n_head = self.n_head, 
            residual=self.residual,
            norm=self.norm)
        
        self.decoder = setup_module(
            m_type=self.decoder_type,
            enc_dec="decoding",
            in_dim=self.dec_in_dim,
            hidden_dim=self.dec_hidden_dim,
            out_dim=self.enc_in_dim,
            num_layers=self.dec_num_layers,
            activation=self.activation,
            dropout=self.feat_drop,
            n_head = self.n_head, 
            residual=self.residual,
            norm=self.norm)
        
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.enc_in_dim))
        
        if self.concate_hidden:
            self.encoder_to_decoder = nn.Linear(self.dec_in_dim * self.enc_num_layers, self.dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(self.dec_in_dim, self.dec_in_dim, bias=False)
        
        if args.loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        elif args.loss_fn == 'sce':
            self.criterion = partial(sce_loss, alpha=args.sce_alpha)     
        else:
            print('error')
            sys.exit()

    def forward(self):
        item_feats = self.feats

        for i in range(self.feat_conv_layers):
            user_feats = torch.sparse.mm(self.ui_graph, item_feats)
            item_feats = torch.sparse.mm(self.iu_graph, user_feats)
        
        mask_user_feats, (mask_user_nodes, keep_user_nodes) = self.encoding_mask_noise(user_feats, self.mask_rate, self.replace_rate)
        mask_item_feats, (mask_item_nodes, keep_item_nodes) = self.encoding_mask_noise(item_feats, self.mask_rate, self.replace_rate)

        enc_user_rep, enc_item_rep, enc_user_hidden, enc_item_hidden = self.encoder(self.A, mask_user_feats, mask_item_feats, return_hidden=True)
        if self.concate_hidden:
            enc_user_rep = torch.cat(enc_user_hidden, dim = 1)
            enc_item_rep = torch.cat(enc_item_hidden, dim = 1)

        user_rep = self.encoder_to_decoder(enc_user_rep)
        item_rep = self.encoder_to_decoder(enc_item_rep)

        # -------recon dmask--------
        user_rep[mask_user_nodes] = 0.0
        item_rep[mask_item_nodes] = 0.0

        dec_user_rep, dec_item_rep = self.decoder(self.A, user_rep, item_rep)

        user_init = user_feats[mask_user_nodes]
        user_rec = dec_user_rep[mask_user_nodes]
        item_init = item_feats[mask_item_nodes]
        item_rec = dec_item_rep[mask_item_nodes]

        loss = self.criterion(user_rec, user_init) + self.criterion(item_rec, item_init)
        # enc_user_rep, enc_item_rep = self.encoder(self.A, user_feats, item_feats)

        return loss
    
    def embed(self):
        item_feats = self.feats

        for i in range(self.feat_conv_layers):
            user_feats = torch.sparse.mm(self.ui_graph, item_feats)
            item_feats = torch.sparse.mm(self.iu_graph, user_feats)
        
        # enc_user_rep, enc_item_rep = user_feats, item_feats
        enc_user_rep, enc_item_rep = self.encoder(self.A, user_feats, item_feats)

        return enc_user_rep, enc_item_rep

    def encoding_mask_noise(self, feats, mask_rate, replace_rate):
        n_nodes = feats.shape[0]
        perm = torch.randperm(n_nodes, device=self.device)

        # random masking
        n_mask_nodes = int(n_nodes * mask_rate)
        mask_nodes = perm[ :n_mask_nodes]
        keep_nodes = perm[n_mask_nodes: ]

        if replace_rate > 0:
            n_noise_nodes = int(replace_rate * n_mask_nodes)
            perm_mask = torch.randperm(n_mask_nodes, device=self.device)
            noise_nodes = mask_nodes[perm_mask[ :n_noise_nodes]]
            token_nodes = mask_nodes[perm_mask[n_noise_nodes: ]]
            noise_to_be_chosen = torch.randperm(n_nodes, device=self.device)[ :n_noise_nodes]

            out_feats = feats.clone()
            out_feats[token_nodes] = 0.0
            out_feats[noise_nodes] = feats[noise_to_be_chosen]

        else:
            out_feats = feats.clone()
            token_nodes = mask_nodes
            out_feats[mask_nodes] = 0.0

        out_feats[token_nodes] += self.enc_mask_token

        return out_feats, (mask_nodes, keep_nodes)
    
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