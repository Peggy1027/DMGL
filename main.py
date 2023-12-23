import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import math
import numpy as np
import os
import random
from torch import optim as optim
import time

from utility.arg_parser import parse_args
from utility.load_data import Data, get_train_loader
from utility.utils import get_parameter_number, get_local_time, mkdir_ifnotexist, create_optimizer
from utility.test import Test
from utility.augmentation import Augmentation

from Model import MyModel


import torch.utils.tensorboard as tb


args = parse_args()

def pretrain(feat_net, feat_optimizer, feat_scheduler):
    feat_net.train()

    loss = feat_net.forward()

    feat_optimizer.zero_grad()
    loss.backward()
    feat_optimizer.step()

    feat_scheduler.step()

    return loss

def train(net, image_user_feats, image_item_feats, text_user_feats, text_item_feats, train_loader, optimizer, scheduler, epoch, device):
    train_loss, mf_loss, id_reg_loss, feat_reg_loss = 0., 0., 0., 0.
    contrastive_loss = 0.
    
    image_user_feats = image_user_feats.to(device)
    image_item_feats = image_item_feats.to(device)
    text_user_feats = text_user_feats.to(device)
    text_item_feats = text_item_feats.to(device)

    for batch_idx, (user, pos, neg) in enumerate(tqdm(train_loader, file=sys.stdout)):
        net.train()

        t3 = time.time()
        users = user.to(device)  # [B]
        pos_items = pos.to(device)  # [B]
        neg_items = neg.squeeze(1).type(torch.long)
        neg_items = neg_items.to(device)

        epoch = torch.LongTensor([epoch])
        epoch = epoch.to(device)

        batch_loss, batch_mf_loss, batch_id_reg_loss, batch_feat_reg_loss, batch_contrastive_loss =\
            net.calculate_loss(users, pos_items, neg_items, image_user_feats, image_item_feats, text_user_feats, text_item_feats, args.batch_size, epoch)
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        scheduler.step()
        
        train_loss += batch_loss.item()
        mf_loss += batch_mf_loss.item()
        id_reg_loss += batch_id_reg_loss.item()
        feat_reg_loss += batch_feat_reg_loss.item()
        contrastive_loss += batch_contrastive_loss.item()
        t4 = time.time()
        print(t4-t3)
        print(f'Training on Epoch {epoch + 1}  [batch_loss {float(batch_loss):f}] [train_loss {float(train_loss):f}]')

    return train_loss, mf_loss, id_reg_loss, feat_reg_loss, contrastive_loss

def validate(net, image_user_feats, image_item_feats, text_user_feats, text_item_feats, config, epoch, is_val=True):
    net.eval()
    test = Test(config, args)
    with torch.no_grad():
        ua_embeddings, ia_embeddings = net.full_predict(image_user_feats, image_item_feats, text_user_feats, text_item_feats, epoch)
        result = test.test_predict(ua_embeddings, ia_embeddings, is_val)
    return result

def set_lr_scheduler(optimizer):
    fac = lambda epoch: 0.96 ** (epoch / 50)
    scheduler_D = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fac)
    return scheduler_D

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

if __name__ == '__main__':
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(device)
    set_seed(args.seed)

    data_generator = Data(path = args.data_path + args.dataset, batch_size=args.batch_size)
    
    print("----------------------------")
    print(args.dataset, "statistical information")
    print('n_users=%d, n_items=%d' % (data_generator.n_users, data_generator.n_items))
    print('n_interactions=%d' % (data_generator.n_train + data_generator.n_val + data_generator.n_test))
    print('n_train=%d, n_val=%d, n_test=%d' % (data_generator.n_train, data_generator.n_val, data_generator.n_test))
    print('sparsity=%.5f' % ((data_generator.n_train + data_generator.n_val + data_generator.n_test)/(data_generator.n_users * data_generator.n_items)))
    print("----------------------------")  
    
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_train'] = data_generator.n_train
    config['n_test'] = data_generator.n_test
    config['n_val'] = data_generator.n_val
    
    config['all_set'] = data_generator.all_set
    config['train_set'] = data_generator.train_set
    config['test_set'] = data_generator.test_set
    config['val_set'] = data_generator.val_set
    config['val_test_set'] = data_generator.val_test_set
    config['ui_graph_tr'], config['ui_graph_val'], config['A_tr'], config['A_val'] = data_generator.get_adj_mat()

    image_feats = np.load(args.data_path + '{}/image_feat.npy'.format(args.dataset))
    text_feats = np.load(args.data_path + '{}/text_feat.npy'.format(args.dataset))
    
    config['image_feats'] = image_feats
    config['text_feats'] = text_feats
    config['image_in_dim'] = image_feats.shape[1]       # 4096
    config['text_in_dim'] = text_feats.shape[1]         # 1024

    train_loader = get_train_loader(config=config, args=args)

    ################## hyperparameter ###############
    id_reg_decay = [0]
    feat_reg_decay = [0]
    cl_decay = [0.01]      # 0.001

    ################## hyperparameter ###############

    for i in id_reg_decay:
        for j in feat_reg_decay:
            for k in cl_decay:                                    
                args.id_reg_decay = i
                args.feat_reg_decay = j
                args.cl_decay = k
                t = get_local_time()
                ########## TODO tensorboard ##########
                # dir = f'./runs/{args.dataset}_{args.encoder}/lr{args.lr}_id_reg_decay{args.id_reg_decay}_feat_reg_decay{args.feat_reg_decay}_cl_decay{args.cl_decay}_drop_rate{args.drop_rate}_topk_rate{args.topk_rate}_recon_rate{args.recon_rate}_sce_alpha{args.sce_alpha}_pre_layers{args.pre_layers}_dec_layers{args.dec_layers}_feat_aug_rate{args.feat_aug_rate}_{t}'
                writer = tb.SummaryWriter(log_dir=dir)

                image_feat_net = Augmentation(feat_type = 'image', config = config, args = args, device = device)
                image_feat_net = image_feat_net.to(device)
                text_feat_net = Augmentation(feat_type = 'text', config = config, args = args, device = device)
                text_feat_net = text_feat_net.to(device)

                image_feat_optimizer = optim.AdamW(
                    [
                        {'params':image_feat_net.parameters()},      
                    ]
                        , lr=args.lr)
                image_feat_scheduler = set_lr_scheduler(image_feat_optimizer)

                text_feat_optimizer = optim.AdamW(
                    [
                        {'params':text_feat_net.parameters()},      
                    ]
                        , lr=args.lr)
                text_feat_scheduler = set_lr_scheduler(text_feat_optimizer)
                
                # TODO
                output_dir =  f"./log/{args.dataset}_{args.encoder}/{t}" 
                mkdir_ifnotexist(output_dir)
                f = open(os.path.join(output_dir, 'logs.txt'), 'w')
                f.write(f'{get_parameter_number(image_feat_net)} \n')
                f.write(f'{get_parameter_number(text_feat_net)} \n')
                f.write('-------------------\n')
                f.write('hyperparameter: \n')
                f.write('-------------------\n')
                f.write('\n'.join([str(k) + ': ' + str(v) for k, v in vars(args).items()]))
                f.write('\n-----------------')
                f.write('\nresults:')
                f.write('\n-----------------')

                global_best_recall = 0.0
                global_best_epoch = 0
                global_stopping_step = 0
                for feat_epoch in range(args.num_epoch):
                    image_loss = pretrain(image_feat_net, image_feat_optimizer, image_feat_scheduler)
                    text_loss = pretrain(text_feat_net, text_feat_optimizer, text_feat_scheduler)

                    a = f'feat_epoch {feat_epoch+1}: image_loss == {image_loss:.5f}, text_loss == {text_loss:.5f}'
                    print(a)
                    f.write('\n' + a)

                    writer.add_scalar("image_training Loss", image_loss, feat_epoch+1)
                    writer.add_scalar("text_training Loss", text_loss, feat_epoch+1)
                    
                    if (feat_epoch+1) % 10 == 0:
                        a = f'-------------------start validating --------------------'
                        f.write('\n'+a)
                        print(a)
                        
                        t1 = time.time()
                        image_feat_net.eval()
                        text_feat_net.eval()

                        with torch.no_grad():
                            image_user_feats, image_item_feats = image_feat_net.embed()
                            text_user_feats, text_item_feats = text_feat_net.embed()
                        t2 = time.time()
                        # print("encoder: ", t2-t1)

                        net = MyModel(config = config, args = args, device = device)
                        net = net.to(device)    
                        optimizer = optim.AdamW(
                            [
                                {'params':net.parameters()},      
                            ]
                                , lr=args.lr)
                        scheduler = set_lr_scheduler(optimizer)
                        
                        best_recall = 0.0
                        best_epoch = 0
                        stopping_step = 0
                        for epoch in range(args.num_epoch):

                            whole_epoch = whole_epoch + 1
                            train_loss, mf_loss, id_reg_loss, feat_reg_loss, contrastive_loss =\
                                train(net, image_user_feats, image_item_feats, text_user_feats, text_item_feats, train_loader, optimizer, scheduler, epoch, device)
                        
                            if math.isnan(train_loss) == True:
                                a = f'{epoch + 1} ERROR: loss is nan.'
                                print(a)
                                f.write('\n' + a)
                                sys.exit()

                    
                            a = f'----Epoch {epoch+1}:  train_loss=={train_loss:.5f}={mf_loss:.5f}+{id_reg_loss:.5f}+{feat_reg_loss:.5f}+{contrastive_loss:.5f}'
                            print(a)
                            f.write('\n' + a)

                            result = validate(net, image_user_feats, image_item_feats, text_user_feats, text_item_feats, config, epoch, is_val=True)
                    
                            writer.add_scalar("recall@20:", result['recall'][2], whole_epoch)   
                    
                            a = '----Epoch %d : recall=[%.5f, %.5f, %.5f, %.5f], precision=[%.5f, %.5f, %.5f, %.5f], ' \
                                                'hit=[%.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f]' % \
                                                (epoch+1, result['recall'][0], result['recall'][1], result['recall'][2], result['recall'][3],
                                                result['precision'][0], result['precision'][1], result['precision'][2], result['precision'][3],
                                                result['hit_ratio'][0], result['hit_ratio'][1], result['hit_ratio'][2], result['hit_ratio'][3],
                                                result['ndcg'][0], result['ndcg'][1], result['ndcg'][2], result['ndcg'][3])
                            print(a)
                            f.write('\n'+a)

                            if result['recall'][2] > best_recall:
                                best_recall = result['recall'][2]
                                best_epoch = epoch + 1
                                # if args.with_test:
                                #     test_result = validate(net, config, epoch, is_val = False)
                                #     a = "Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[1], test_result['recall'][1], test_result['precision'][1], test_result['ndcg'][1])
                                #     print(a)
                                #     f.write("\n" + a)
                                    # writer.add_scalar("Test acc:", test_result['recall'][1], epoch+1)
                                stopping_step = 0

                            elif stopping_step < args.early_stopping_patience:
                                stopping_step += 1
                                a = f'#####Early stopping steps: {stopping_step} #####'
                                f.write("\n" + a)
                                print(a)

                            else:
                                a = f'#####Early stop! #####'
                                f.write("\n" + a)
                                print(a)
                                break
                        
                        a = f'best epoch : {best_epoch}, best recall : {best_recall}'
                        f.write('\n'+a)
                        print('\n'+a)
                        a = f'-------------------finishing validating --------------------'
                        f.write('\n'+a)
                        print('\n'+a)
                        
                        # first-stage
                        if best_recall > global_best_recall:
                            global_best_recall = best_recall
                            global_best_epoch = feat_epoch
                            global_stopping_step = 0
                        elif global_stopping_step < args.global_early_stopping_patience:
                            global_stopping_step += 1
                            a = f'**** Global Early stopping steps: {global_stopping_step} ********'
                            f.write('\n' + a)
                            print(a)
                        else:
                            a = f'**** Global Early stop! *****'
                            f.write('\n' + a)
                            print(a)
                            break
                a = f'global best epoch : {global_best_epoch}, best recall : {global_best_recall}'
                f.write('\n'+a)
                print('\n'+a)
                f.close()
                writer.close()









