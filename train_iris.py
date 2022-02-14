#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pdb
import time
import math
import torch
import shutil
import random
import logging
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc, average_precision_score

import model as mlib
from args_config import arcface_UE
from data_config import MobIn_config, LmpIn_config, ThsIn_config,DisIn_config
from load_norm_imglist import ImageList

torch.backends.cudnn.bencmark = True
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # TODO

def get_tpr_at_fpr(tpr, fpr, thr):
    idx = np.argwhere(fpr > thr)
    return tpr[idx[0]][0]

def OneHot(x):
    # get one hot vectors
    n_class = int(x.max() + 1)
    onehot = torch.eye(n_class)[x.long()]
    return onehot # N X D

def get_eer(tpr, fpr):
    # pdb.set_trace()
    for i, fpr_point in enumerate(fpr):
        # print(i)
        # print(fpr_point)
        if (tpr[i] >= 1 - fpr_point):
            idx = i
            break
    if (tpr[idx] == tpr[idx+1]):
        return 1 - tpr[idx]
    else:
        return fpr[idx]

class UETrainer(object):

    def __init__(self, args, config):
        self.args    = args
        self.model   = dict()
        self.data    = dict()
        self.result  = dict()
        self.softmax = torch.nn.Softmax(dim=1)
        self.use_gpu = args.use_gpu and torch.cuda.is_available()
        self.config = config


        logger = logging.getLogger()
        logger.setLevel(logging.INFO) 
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_path = os.getcwd() + '/log_save/'  
        log_name = log_path + self.config.data_name + '_'+self.config.test_type+'_UE_' + rq + '.log'
        logfile = log_name
        logger.handlers.clear()
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        self.logger = logger

        self.args.classnum = sum(config.num_classGet())
        save_to = './checkpoint/'+self.config.data_name+'_'+self.config.test_type+'_UE'
        self.args.save_to = save_to.replace(' ','')

    def _print_log_print(self, print_words):        
        print(print_words)
        self.logger.info(print_words)

    def _report_settings(self):
        ''' Report the settings '''

        str = '-' * 16
        self._print_log_print('%sEnvironment Versions%s' % (str, str))
        self._print_log_print("- Python    : {}".format(sys.version.strip().split('|')[0]))
        self._print_log_print("- PyTorch   : {}".format(torch.__version__))
        self._print_log_print("- TorchVison: {}".format(torchvision.__version__))
        self._print_log_print("- USE_GPU   : {}".format(self.use_gpu))
        self._print_log_print("- IS_DEBUG  : {}".format(self.args.is_debug))
        self._print_log_print('%sTraining Setting%s' % (str, str))
        self._print_log_print("- Dataset   : {}".format(self.config.data_name))
        self._print_log_print("- Protocol  : {}".format(self.config.test_type))
        self._print_log_print("- FC mode   : {}".format(self.args.fc_mode))
        self._print_log_print("- Loss mode : {}".format(self.args.loss_mode))
        self._print_log_print("- DUL using : {}".format(self.args.used_as))
        self._print_log_print("- Class num : {}".format(self.args.classnum))
        self._print_log_print('%sOutput Setting%s' % (str, str))        
        self._print_log_print("- Model path: {}".format(self.args.save_to)) 
        self._print_log_print('-' * 52)
        
        


    def _model_loader(self):

        self.model['backbone'] = mlib.UE_model(feat_dim=self.args.in_feats, \
                                                 drop_ratio=self.args.drop_ratio, \
                                                 used_as=self.args.used_as)
        self.model['fc_layer']  = mlib.FullyConnectedLayer(self.args)
        self.model['criterion'] = mlib.ClsLoss(self.args)
        self.model['optimizer'] = torch.optim.SGD(
                                      [{'params': self.model['backbone'].parameters()},
                                       {'params': self.model['fc_layer'].parameters()}],
                                      lr=self.args.base_lr,
                                      weight_decay=self.args.weight_decay,
                                      momentum=0.9,
                                      nesterov=True)
        self.model['scheduler'] = torch.optim.lr_scheduler.MultiStepLR(
                                      self.model['optimizer'], milestones=self.args.lr_adjust, gamma=self.args.gamma)
        if self.use_gpu:
            self.model['backbone']  = self.model['backbone'].cuda()
            self.model['fc_layer']  = self.model['fc_layer'].cuda()
            self.model['criterion'] = self.model['criterion'].cuda()

        if self.use_gpu and len(self.args.gpu_ids) > 1:
            self.model['backbone'] = torch.nn.DataParallel(self.model['backbone'], device_ids=self.args.gpu_ids)
            self.model['fc_layer'] = torch.nn.DataParallel(self.model['fc_layer'], device_ids=self.args.gpu_ids)
            self._print_log_print('Parallel mode was going ...')
        elif self.use_gpu:
            self._print_log_print('Single-gpu mode was going ...')
        else:
            self._print_log_print('CPU mode was going ...')

        if len(self.args.resume) > 2:
            checkpoint = torch.load(self.args.resume, map_location=lambda storage, loc: storage)
            self.args.start_epoch = checkpoint['epoch']
            self.model['backbone'].load_state_dict(checkpoint['backbone'])
            self.model['fc_layer'].load_state_dict(checkpoint['fc_layer'])
            self._print_log_print('Resuming the train process at %3d epoches ...' % self.args.start_epoch)
        self._print_log_print('Model loading was finished ...')

    
    @staticmethod
    def collate_fn_1v1(batch):
        imgs, pairs_info = [], []
        for unit in batch:
            pairs_info.append([unit['name1'], unit['name2'], unit['label']])
            imgs.append(torch.cat((unit['face1'], unit['face2']), dim=0))
        return (torch.stack(imgs, dim=0), np.array(pairs_info))
    
    
    def _data_loader(self):
        train_loader_param = self.config.load_detailGet()
        self.data['train'] = torch.utils.data.DataLoader(
            ImageList(root=train_loader_param[0], fileList=train_loader_param[1], 
            transform=transforms.Compose([ 
                transforms.Resize((self.args.img_size,self.args.img_size)),
                transforms.ToTensor(),
            ])),batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.workers, pin_memory=True)

        test_loader_param = self.config.test_loaderGet()
        self.data['test'] = torch.utils.data.DataLoader(
            ImageList(root=test_loader_param[0], fileList=test_loader_param[1], 
            transform=transforms.Compose([ 
                transforms.Resize((self.args.img_size,self.args.img_size)),
                transforms.ToTensor(),
            ])),batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.workers, pin_memory=True)

        self._print_log_print('Data loading was finished ...')

    def _UGCL_instance_sampler(self, score, epoch, batch_size):
        def _value_stretch(value):
            return (value-value.min())/(value.max()-value.min())
        if epoch <= 0:
            return None
        else:
            # stretch_score = _value_stretch(score)
            weight = torch.zeros_like(score)
            threshold = 1./(1.+math.exp(-epoch*0.5))
            _, sort_idx = score.sort()
            learning_num = min(round(batch_size*threshold),batch_size)
            weight[sort_idx[:learning_num]]=1
            
            return weight

    def _train_one_epoch(self, epoch = 0):

        self.model['backbone'].train()
        self.model['fc_layer'].train()

        loss_recorder, batch_acc = [], []
        for idx, (img, gty) in enumerate(self.data['train']):
            train_step = (epoch-1)*len(self.data['train'])+idx

            img.requires_grad = False
            gty.requires_grad = False

            if self.use_gpu:
                img = img.cuda()
                gty = gty.cuda()

            mu, logvar, embedding, score = self.model['backbone'](img)
            instance_samples = self._UGCL_instance_sampler(score, epoch, img.size(0))
            output  = self.model['fc_layer'](embedding, gty)
            loss    = self.model['criterion'](output, gty, \
                                              weight = instance_samples, \
                                              mu = mu, \
                                              logvar = logvar)
            self.model['optimizer'].zero_grad()
            loss.backward()
            self.model['optimizer'].step()
            predy   = np.argmax(output.data.cpu().numpy(), axis=1)  
            it_acc  = np.mean((predy == gty.data.cpu().numpy()).astype(int))
            batch_acc.append(it_acc)
            loss_recorder.append(loss.item())
            if (idx + 1) % self.args.print_freq == 0:
                if (instance_samples is None):
                    self._print_log_print('epoch : %2d|%2d, iter : %4d|%4d,  loss : %.4f, batch_ave_acc : %.4f, var-mean: %.4f, var-std: %.4f' % \
                        (epoch, self.args.end_epoch, idx+1, len(self.data['train']), np.mean(loss_recorder), np.mean(batch_acc), score.mean().item(),score.std().item()))
                else:
                    self._print_log_print('epoch : %2d|%2d, iter : %4d|%4d,  loss : %.4f, weight: %.4f, batch_ave_acc : %.4f, var-mean: %.4f, var-std: %.4f' % \
                        (epoch, self.args.end_epoch, idx+1, len(self.data['train']), np.mean(loss_recorder), instance_samples.mean().item(), np.mean(batch_acc), score.mean().item(),score.std().item()))
        train_loss = np.mean(loss_recorder)
        print('train_loss : %.4f' % train_loss)
        return train_loss
    
    def _test_model(self,epoch):
        self.model['backbone'].eval()
        whole_feature = 1        
        with torch.no_grad():
            for data, label in self.data['test']:           
                data.requires_grad = False
                label.requires_grad = False

                if self.use_gpu:
                    data = data.cuda()
                    label = label.cuda()
                    
                if 'baseline' in self.args.used_as:
                    _,_,feature,_ = self.model['backbone'](data)
                elif self.args.used_as=='UE':
                    feature,_,_,_ = self.model['backbone'](data)
                else:
                    return
                data_feature = feature.squeeze()
                if torch.is_tensor(whole_feature):
                    whole_feature = torch.cat((whole_feature,data_feature),0)
                    whole_label = torch.cat((whole_label,label),0)            
                else:
                    whole_feature = data_feature
                    whole_label = label
        
        whole_feature = whole_feature/whole_feature.norm(dim=1,keepdim=True)
        gallery_onehot = OneHot(whole_label)

        sim_mat = whole_feature.mm(whole_feature.t())
        sig_mat = torch.mm(gallery_onehot, gallery_onehot.t())
        scores = sim_mat.contiguous().view(-1)
        signals = sig_mat.contiguous().view(-1)
        
        ind_keep = 1. - torch.eye(whole_feature.size(0))
        ind_keep = ind_keep.contiguous().view(-1)
        scores = scores[ind_keep>0]
        signals = signals[ind_keep>0]
        score_matrix = scores.reshape((-1, ))
        label_matrix = signals.reshape((-1, ))
        
        fpr, tpr, _ = roc_curve(label_matrix.cpu(), score_matrix.cpu(), pos_label=1)
        eer = get_eer(tpr,fpr)
        prec1 = 1.-get_tpr_at_fpr(tpr, fpr, 1e-1)
        prec2 = 1.-get_tpr_at_fpr(tpr, fpr, 1e-3)
        prec3 = 1.-get_tpr_at_fpr(tpr, fpr, 1e-5)
        self._print_log_print('EPOCH-{}: EER {:.4f} | R@A1e-1 {:.4f} | R@A1e-3 {:.4f} | R@A1e-5 {:.4f}\n'.format(epoch, eer,prec1,prec2,prec3))

        out_inf = {}
        out_inf['eer'] = eer
        out_inf['RatAe-1'] = prec1
        out_inf['RatAe-3'] = prec2
        out_inf['RatAe-5'] = prec3
        return out_inf

    def _save_weights(self, testinfo = {}):
        ''' save the weights during the process of training '''
        
        if not os.path.exists(self.args.save_to):
            os.mkdir(self.args.save_to)
            
        freq_flag = self.result['epoch'] % self.args.save_freq == 0
        sota_flag = self.result['sota_eer'] > testinfo['eer']

        save_name = '%s/epoch_%02d-eer_%.4f.pth' % \
                         (self.args.save_to, self.result['epoch'], testinfo['eer'])
        if sota_flag:
            save_name = '%s/sota.pth' % self.args.save_to
            self.result['sota_eer']   = testinfo['eer']
            self._print_log_print('%s Yahoo, SOTA model was updated %s' % ('*'*16, '*'*16))
        
        if sota_flag or freq_flag:
            torch.save({
                'epoch'   : self.result['epoch'], 
                'backbone': self.model['backbone'].state_dict(),
                'fc_layer': self.model['fc_layer'].state_dict(),
                'sota_acc': testinfo['eer']}, save_name)
            
        if sota_flag:
            normal_name = '%s/epoch_%02d-eer_%.4f.pth' % \
                         (self.args.save_to, self.result['epoch'], testinfo['eer'])
            shutil.copy(save_name, normal_name)
            
            
    def _dul_training(self):
        
        self.result['sota_eer']   = 1
        for epoch in range(self.args.start_epoch, self.args.end_epoch + 1):

            start_time = time.time()
            self.result['epoch'] = epoch
            train_loss = self._train_one_epoch(epoch)
            self.model['scheduler'].step()
            eval_info = self._test_model(epoch)
            end_time = time.time()
            self._print_log_print('Single epoch cost time : %.2f mins' % ((end_time - start_time)/60))
            self._save_weights(eval_info)
            
            if self.args.is_debug:
                break


    def train_runner(self):

        self._report_settings()

        self._model_loader()

        self._data_loader()

        self._dul_training()


if __name__ == "__main__":
    
    ## data config
    # config = MobIn_config()
    # config = LmpIn_config()
    config = ThsIn_config()
    # config = DisIn_config()

    ## model config    
    input_args = arcface_UE()
        
    trainer = UETrainer(input_args, config)
    trainer.train_runner()
