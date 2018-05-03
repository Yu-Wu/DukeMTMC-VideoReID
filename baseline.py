import torch
from torch import nn
from reid import models
from reid.trainers import Trainer
from reid.evaluators import extract_features, Evaluator
from reid.dist_metric import DistanceMetric
import numpy as np
from collections import OrderedDict
import os.path as osp
import pickle
from reid.utils.serialization import load_checkpoint
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from reid.utils.data.preprocessor import Preprocessor
import random


class Baseline():
    """ The baseline model """
    def __init__(self, model_name, batch_size, num_classes, data_dir, save_path, dropout=0.5, max_frames=900):
        # "max_frames" defines the maximum frames in a tracklet in dataloader
        # More details about it can be found in ./reid/utils/data/preprocessor.py

        self.model_name = model_name
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.save_path = save_path


        self.dataloader_params = {}
        self.dataloader_params['height'] = 256
        self.dataloader_params['width'] = 128
        self.dataloader_params['batch_size'] = batch_size
        self.dataloader_params['workers'] = 6


        self.batch_size = batch_size
        self.data_height = 256
        self.data_width = 128
        self.data_workers = 6

        # batch size for eval mode. Default is 1. 
        self.eval_bs = 1
        self.dropout = dropout
        self.max_frames = max_frames


    def get_dataloader(self, dataset, training=False) :
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        if training:
            transformer = T.Compose([
                T.RandomSizedRectCrop(self.data_height, self.data_width),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalizer,
            ])
            batch_size = self.batch_size

        else:
            transformer = T.Compose([
                T.RectScale(self.data_height, self.data_width),
                T.ToTensor(),
                normalizer,
            ])
            batch_size = self.eval_bs

        data_loader = DataLoader(
            Preprocessor(dataset, root=self.data_dir,
                         transform=transformer, is_training=training, max_frames=self.max_frames),
            batch_size=batch_size, num_workers=self.data_workers,
            shuffle=training, pin_memory=True, drop_last=training)

        current_status = "Training" if training else "Test"
        print("create dataloader for {} with batch_size {}".format(current_status, batch_size))
        return data_loader




    def train(self, train_data, epochs=70, step_size=55, init_lr=0.1, dropout=0.5):

        """ create model and dataloader """
        model = models.create(self.model_name, dropout=self.dropout, num_classes=self.num_classes)
        model = nn.DataParallel(model).cuda()
        dataloader = self.get_dataloader(train_data, training=True)


        # the base parameters for the backbone (e.g. ResNet50)
        base_param_ids = set(map(id, model.module.CNN.base.parameters())) 

        # we fixed the first three blocks to save GPU memory
        base_params_need_for_grad = filter(lambda p: p.requires_grad, model.module.CNN.parameters()) 

        # params of the new layers
        new_params = [p for p in model.parameters() if id(p) not in base_param_ids]

        # set the learning rate for backbone to be 0.1 times
        param_groups = [
            {'params': base_params_need_for_grad, 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]

        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(param_groups, lr=init_lr, momentum=0.5, weight_decay = 5e-4, nesterov=True)

        # change the learning rate by step
        def adjust_lr(epoch, step_size):
            lr = init_lr / (10 ** (epoch // step_size))
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)

            if epoch % step_size == 0:
                print("Epoch {}, current lr {}".format(epoch, lr))

        """ main training process """
        trainer = Trainer(model, criterion)
        for epoch in range(epochs):
            adjust_lr(epoch, step_size)
            trainer.train(epoch, dataloader, optimizer, print_freq=10)

        torch.save(model.state_dict(), osp.join(self.save_path,  "model_{}.ckpt".format(epoch)))
        self.model = model


    def get_feature(self, dataset):
        dataloader = self.get_dataloader(dataset, training=False)
        features,_ = extract_features(self.model, dataloader)
        features = np.array([logit.numpy() for logit in features.values()])
        return features

    def evaluate(self, query, gallery):
        print("Evaluate model in {}".format(self.save_path))
        test_loader = self.get_dataloader(list(set(query) | set(gallery)), training = False)
        evaluator = Evaluator(self.model)
        evaluator.evaluate(test_loader, query, gallery)


    def resume(self, ckpt_file):
        print("continued from", ckpt_file)
        model = models.create(self.model_name, dropout=self.dropout, num_classes=self.num_classes)
        self.model = nn.DataParallel(model).cuda()
        self.model.load_state_dict(load_checkpoint(ckpt_file))
