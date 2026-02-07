import time
import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_graph_utils_tsne import train_utils
args = None
import random
import numpy as np
seed=32
np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--model_name', type=str, default='HermiNet', help='the name of the model')
    parser.add_argument('--sample_length', type=int, default=xx, help='batchsize of the training process')
    parser.add_argument('--data_name', type=str, default='xx', help='the name of the data')
    parser.add_argument('--Input_type', choices=[' TD', 'FD','other'],type=str, default='FD', help='the input type decides the length of input')
    #选择数据文件夹
    parser.add_argument('--data_dir', type=str, default="./xxx",help='the directory of the data')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument('--batch_size', type=int, default=100, help='batchsize of the training process')
    parser.add_argument('--num_workers' , type=int, default=0, help='the number of training process')


    parser.add_argument('--task', choices=['Node', 'Graph'], type=str,
                            default='Node', help='Node classification or Graph classification')
    parser.add_argument('--pooltype', choices=['TopKPool', 'EdgePool', 'ASAPool', 'SAGPool'],type=str,
                        default='EdgePool', help='For the Graph classification task')
    #高斯噪声
    # parser.add_argument('--snr_db',type=float,default=-5)

     parser.add_argument('--pooltype1', choices=['TopKPool', 'EdgePool', 'ASAPool', 'SAGPool'], type=str,
                         default='EdgePool', help='Pooling type for the first part of the graph')
    
     parser.add_argument('--pooltype2', choices=['TopKPool', 'EdgePool', 'ASAPool', 'SAGPool'], type=str,
                         default='ASAPool', help='Pooling type for the second part of the graph')

    # optimization information
    parser.add_argument('--layer_num_last', type=int, default=0, help='the number of last layers which unfreeze')
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.8, help='the momentum for opt')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='step', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.2, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='9', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--resume', type=str, default='', help='the directory of the resume training model')
    parser.add_argument('--max_model_num', type=int, default=1, help='the number of most recent models to save')
    parser.add_argument('--max_epoch', type=int, default=x, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=x, help='the interval of log training information')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    if args.task == 'Node':
        sub_dir = args.task + '_' +args.model_name+'_'+args.data_name + '_' + args.Input_type +'_'+datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    else:
        sub_dir = args.task + '_' +  args.model_name + '_' + args.pooltype + '_' + args.data_name + '_' + args.Input_type + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
        #！
        # sub_dir = args.task + '_' + args.model_name + '_' + args.pooltype1 + '_' + args.pooltype2 + '_'+args.data_name + '_' + args.Input_type + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
        #！

    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_utils(args, save_dir)
    trainer.setup()
    trainer.train()
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 打印运行时长
    print("代码运行时长：", elapsed_time, "秒")


