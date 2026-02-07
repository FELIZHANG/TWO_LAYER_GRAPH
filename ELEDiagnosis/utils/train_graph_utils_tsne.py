#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch import optim
from torch_geometric.data import DataLoader
import models
import models2
import datasets
from utils.save import Save_Tool
from utils.freeze import set_freeze_by_id
from sklearn import metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_scatter import scatter
from pylab import mpl

mpl.rcParams['font.sans-serif']=['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus']=False


class train_utils(object):
    #！
    def __init__(self, args, save_dir):
    # def __init__(self, args, save_dir, pooltype1=None, pooltype2=None):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))


        # Load the datasets
        #getattr:返回对象属性值
        Dataset = getattr(datasets, args.data_name)
        self.datasets = {}
        self.datasets['train'], self.datasets['val'] = Dataset(args.sample_length,args.data_dir, args.Input_type, args.task).data_preprare()

        self.dataloaders = {x: DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['train', 'val']}
        # Define the model
        InputType = args.Input_type
        if InputType == "TD":
            feature = args.sample_length
        elif InputType == "FD":
            feature = int(args.sample_length/2)
        elif InputType == "other":
            feature = 1
        else:
            print("The InputType is wrong!!")

        if args.task == 'Node':
            self.model = getattr(models, args.model_name)(feature=feature,out_channel=5)
        elif args.task == 'Graph':
             self.model = getattr(models2, args.model_name)(feature=feature, out_channel=Dataset.num_classes,
                                                            pooltype1=args.pooltype1, pooltype2=args.pooltype2)
        else:
            print('The task is wrong!')

        if args.layer_num_last != 0:
            set_freeze_by_id(self.model, args.layer_num_last)
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        # Load the checkpoint
        self.start_epoch = 0
        if args.resume:
            suffix = args.resume.rsplit('.', 1)[-1]
            if suffix == 'tar':
                checkpoint = torch.load(args.resume)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suffix == 'pth':
                self.model.load_state_dict(torch.load(args.resume, map_location=self.device))

        # Invert the model and define the loss
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.MSELoss()



    def train(self):
        args = self.args
        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        epochs=[]
        train_loss=[]
        val_loss=[]
        train_acc=[]
        val_acc=[]

        step_start = time.time()

        save_list = Save_Tool(max_num=args.max_model_num)
        all_features=[]
        all_labels = []  # For validation phase (t-SNE)
        all_test_labels = []  # For test phase (confusion matrix)
        all_test_preds = []  # For test phase (confusion matrix)

        for epoch in range(self.start_epoch, args.max_epoch):

            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase

            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0

                # Set model to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                sample_num = 0

                for data in self.dataloaders[phase]:
                    inputs = data.to(self.device)
                    # inputs=add_gaussion_noise(inputs,snr_db=self.args.snr_db if hasattr(self.args,'snr_db') else 0)
                    labels = inputs.y

                    if args.task == 'Node':
                        bacth_num = inputs.num_nodes
                        sample_num += len(labels)
                    elif args.task == 'Graph':
                        bacth_num = inputs.num_graphs
                        sample_num += len(labels)
                    else:
                        print("There is no such task!!")
                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'train'):

                        # forward
                        if  args.task == 'Node':
                            # logits = self.model(inputs)
                            #！
                            features, logits = self.model(inputs)
                            #！
                        elif args.task == 'Graph':
                            # logits = self.model(inputs,args.pooltype)
                            #！
                            features, logits = self.model(inputs, args.pooltype)
                            #！

                            #!
                            # logits = self.model(inputs, args.pooltype1,args.pooltype2)
                            #!
                        else:
                            print("There is no such task!!")

                        #loss
                        loss = self.criterion(logits, labels)
                        pred = logits.argmax(dim=1)

                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * bacth_num
                        epoch_loss += loss_temp
                        epoch_acc += correct
                        batch_loss += loss_temp
                        batch_acc += correct
                        batch_count += bacth_num
                        batch_loss = batch_loss / batch_count
                        batch_acc = batch_acc / batch_count

                        # Calculate the training information
                        #！
                        if phase == 'val' and epoch == args.max_epoch - 1:
                            all_features.append(features.cpu().detach().numpy())
                            all_labels.append(np.atleast_1d(labels.cpu().detach().numpy()))
                        #！

                        if phase == 'train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            # Print the training information
                            if step % args.print_step == 0:
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0*batch_count/train_time
                                logging.info('Epoch: {}, Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_loss, batch_acc, sample_per_sec, batch_time
                                ))

                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1


                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # Print the train and val information via each epoch

                epoch_loss = epoch_loss / sample_num
                epoch_acc = epoch_acc / sample_num
                if phase == 'train':
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_acc)
                epochs.append(epoch)

                #phase['train', 'val']
                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.4f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time()-epoch_start
                ))

                if phase == 'val':
                    # save the checkpoint for other learning
                    model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(epoch))
                    torch.save({
                        'epoch': epoch,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'model_state_dict': model_state_dic
                    }, save_path)
                    save_list.update(save_path)
                    # save the best model according to the val accuracy
                    if epoch_acc > best_acc or epoch > args.max_epoch-2:
                        best_acc = epoch_acc
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

        Dataset = getattr(datasets, args.data_name)
        test_dataset = Dataset(args.sample_length, args.data_dir, args.Input_type, args.task).data_preprare(
            test=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=(True if self.device == 'cuda' else False))

        self.model.eval()
        test_acc = 0
        test_loss = 0.0
        sample_num = 0
        all_labels = []
        all_preds = []

        for data in test_loader:
            inputs = data.to(self.device)
            labels = inputs.y

            with torch.no_grad():
                if args.task == 'Node':
                    features, logits = self.model(inputs)
                elif args.task == 'Graph':
                    features, logits = self.model(inputs, args.pooltype)

                loss = self.criterion(logits, labels)
                pred = logits.argmax(dim=1)
                correct = torch.eq(pred, labels).float().sum().item()
                test_loss += loss.item() * (inputs.num_nodes if args.task == 'Node' else inputs.num_graphs)
                test_acc += correct
                sample_num += len(labels)
                all_test_labels.append(np.atleast_1d(labels.cpu().numpy()))
                all_test_preds.append(np.atleast_1d(pred.cpu().numpy()))

        test_loss /= sample_num
        test_acc /= sample_num
        logging.info('Test Loss: {:.4f}, Test Acc: {:.4f}'.format(test_loss, test_acc))  # Fixed typo

        # Perform t-SNE visualization after training is complete
        if all_features and all_labels:
            all_features = np.concatenate(all_features, axis=0)
            all_test_labels = np.concatenate(all_test_labels, axis=0)
            all_test_preds = np.concatenate(all_test_preds, axis=0)
            plot_matrix(torch.tensor(all_test_labels), torch.tensor(all_test_preds), range(5),
                        title="Confusion Matrix for Gearbox Dataset", axis_labels=[str(i) for i in range(5)])

            tsne = TSNE(n_components=2, random_state=42)
            tsne_features = tsne.fit_transform(all_features)

            plt.figure(figsize=(8, 8))
            for label in range(7):
                indices = all_labels == label
                plt.scatter(tsne_features[indices, 0], tsne_features[indices, 1], label=f'Fault {label}')
            plt.legend()
            plt.title('t-SNE Visualization of Final Validation Set')
            plt.show()
            plt.close()

            # Confusion matrix
        all_test_labels = np.concatenate(all_test_labels, axis=0)
        all_test_preds = np.concatenate(all_test_preds, axis=0)
        plot_matrix(torch.tensor(all_test_labels), torch.tensor(all_test_preds), range(xxx),
                    title="Confusion Matrix for Gearbox Dataset", axis_labels=[str(i) for i in range(xxx)])




def plot_matrix(y_true,y_pred,labels_name,title=None,thresh=0.8,axis_labels=None):
    y_true=y_true.cpu().numpy()
    y_pred=y_pred.cpu().numpy()
    cm=metrics.confusion_matrix(y_true,y_pred,labels=labels_name,sample_weight=None)
    cm=cm.astype(float)
    row_sum=cm.sum(axis=1)

    for i in range(np.shape(cm)[0]):
        if row_sum[i] == 0:
            continue
        cm[i, :] = cm[i, :] / row_sum[i]

    plt.imshow(cm,interpolation='nearest',cmap=plt.get_cmap('Reds'))
    plt.colorbar()

    if title is not None:
        plt.title(title)

    num_local=np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels=labels_name
    plt.xticks(num_local,axis_labels,rotation=45)
    plt.yticks(num_local,axis_labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if not np.isnan(cm[i][j]) and int(cm[i][j]*100+0.5)>0:
                plt.text(j,i,format(int(cm[i][j]*100+0.5),'d')+'%',
                         ha="center",va="center",
                         color="white" if cm[i][j]>thresh else "black")
    plt.show()
