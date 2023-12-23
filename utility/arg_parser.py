import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--data_path', nargs='?', default='../MM_dataset/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='allrecipes', help='Choose a dataset from {baby, tiktok, allrecipes}')

    parser.add_argument('--gpu_id', type=int, default=3, help='GPU id')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    parser.add_argument('--num_epoch', type=int, default=500, help='Number of epoch.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')    # 1024
    parser.add_argument('--lr', type=float, default=0.00055, help='Learning rate.')

    parser.add_argument('--embed_size', type=int, default=64,help='Embedding size.')
    parser.add_argument('--weight_size', nargs='?', default='[64, 64]', help='Output sizes of every layer')
    parser.add_argument('--id_reg_decay', nargs='?', default=1e-7, help='id_reg_decay')
    parser.add_argument('--feat_reg_decay', default=1e-7, help='feat_reg_decay')
    parser.add_argument('--cl_decay', type=float, default=0.01, help='Control the effect of the contrastive auxiliary task')
    parser.add_argument('--drop_rate', type=float, default=0, help='dropout rate')
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--sparse', type=int, default=1, help='Sparse or dense adjacency matrix')

    parser.add_argument('--feat_conv_layers', type=int, default=1, help='Number of feature graph conv layers')
    parser.add_argument('--id_conv_layers', type=int, default=1, help='Number of id graph conv layers')
    parser.add_argument('--feat_aug_rate', default=0.64)
    parser.add_argument('--head_num', default=4, type=int, help='head_num_of_multihead_attention. For multi-model relation.')
    parser.add_argument('--model_cat_rate', type=float, default=0.55, help='model_cat_rate')
    parser.add_argument('--id_cat_rate', type=float, default=0.36, help='before GNNs')
    parser.add_argument('--topk_rate', default=0.001, type=float, help='for reconstruct')       # [0.005, 0.001]
    parser.add_argument('--recon_rate', default=0.9, type=float)

    parser.add_argument('--Ks', nargs='?', default='[5, 10, 20, 50]', help='K value of ndcg/recall @ k')
    parser.add_argument('--batch_test_flag', default=False)
    parser.add_argument('--test_flag', nargs='?', default='part', help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='')
    parser.add_argument('--with_test', default=False)
    parser.add_argument('--n_candidate', default=999)

    # pretrain args
    parser.add_argument("--encoder", type=str, default="gcn")
    parser.add_argument("--decoder", type=str, default="gcn")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument('--sce_alpha', default=1)           # [1, 3]
    parser.add_argument("--mask_rate", type=float, default=0.3)
    parser.add_argument('--replace_rate', type=float, default=0.1)
    parser.add_argument('--hidden_dim', type=float, default=1024)
    parser.add_argument('--pre_layers', type=int, default=2)    # [2,3,4]
    parser.add_argument('--dec_layers', type=int, default=1)    # [1,2]
    parser.add_argument('--in_drop', type=float, default=0.2)
    parser.add_argument('--n_head', default=4)
    parser.add_argument('--activation', default='prelu')
    parser.add_argument('--residual', default=True)
    parser.add_argument('--norm', type=str, default=None)
    parser.add_argument('--concate_hidden', default=True)
    
    parser.add_argument('--global_early_stopping_patience', type=int, default=6, help='')

    return parser.parse_args()