import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import create_activation

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, activation, residual, norm, encoding=False):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.dropout = dropout
        self.activation = activation

        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual           # True or False
        last_norm = norm if encoding else None
        
        if num_layers == 1:
            self.gcn_layers.append(GraphConv(
                in_dim, out_dim, residual=last_residual, norm=last_norm, activation=last_activation))
        else:
            # input projection (no residual)
            self.gcn_layers.append(GraphConv(
                in_dim, hidden_dim, residual=residual, norm=norm, activation=create_activation(activation)))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gcn_layers.append(GraphConv(
                    hidden_dim, hidden_dim, residual=residual, norm=norm, activation=create_activation(activation)))
            # output projection
            self.gcn_layers.append(GraphConv(
                hidden_dim, out_dim, residual=last_residual, activation=last_activation, norm=last_norm))

        self.norms = None
        self.head = nn.Identity()

    def forward(self, A_graph, user_feat, item_feat, return_hidden = False):
        h_u, h_i = user_feat, item_feat
        user_hidden_list, item_hidden_list = [], []

        for l in range(self.num_layers):
            h_u = F.dropout(h_u, p = self.dropout, training=self.training)
            h_i = F.dropout(h_i, p = self.dropout, training=self.training)
            h_u, h_i = self.gcn_layers[l](A_graph, h_u, h_i)
            
            user_hidden_list.append(h_u)
            item_hidden_list.append(h_i)
        
        if return_hidden:
            return self.head(h_u), self.head(h_i), user_hidden_list, item_hidden_list
        else:
            return self.head(h_u), self.head(h_i)
    
    


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, norm=None, activation=None, residual=True):
        super().__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim

        self.fc = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.fc.weight)

        if residual:
            if self._in_feats != self._out_feats:
                self.res_fc = nn.Linear(
                    self._in_feats, self._out_feats, bias=False)
                print("! Linear Residual !")
            else:
                print("Identity Residual ")
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        
        self.norm = norm
        if norm is not None:
            self.norm = norm(out_dim)
        self._activation = activation

        # self.reset_parameters()

    # def reset_parameters(self):
    #     self.fc.reset_parameters()

    def forward(self, A_graph, user_feat, item_feat):
        trans_user_feats = self.fc(user_feat)
        trans_item_feats = self.fc(item_feat)

        n_users = trans_user_feats.size()[0]
        n_items = trans_item_feats.size()[0]

        all_embs = torch.cat([trans_user_feats, trans_item_feats])

        conv_embs = torch.sparse.mm(A_graph, all_embs)
        conv_user_feats, conv_item_feats = torch.split(conv_embs, [n_users, n_items])
        
        # conv_user_feats = torch.sparse.mm(ui_graph, trans_item_feats)
        # conv_item_feats = torch.sparse.mm(iu_graph, trans_user_feats)

        if self.res_fc is not None:
            conv_user_feats = conv_user_feats + self.res_fc(user_feat)
            conv_item_feats = conv_item_feats + self.res_fc(item_feat)

        if self.norm is not None:
            conv_user_feats = self.norm(conv_user_feats)
            conv_item_feats = self.norm(conv_item_feats)

        if self._activation is not None:
            conv_user_feats = self._activation(conv_user_feats)
            conv_item_feats = self._activation(conv_item_feats)

        return conv_user_feats, conv_item_feats




