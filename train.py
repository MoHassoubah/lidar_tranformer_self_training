import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.SalsaNext import *
from trainer import trainer_kitti

from datasets.lidar_dataset.parser import Parser
import yaml


DATA_DIRECTORY = 'C:\lidar_datasets\kitti_data'     #'./data/GTA5' #should be the path of the kitti LiDAR data
RESTORE_FROM_DIRECTORY = 'C:\msc_codes\proj_tansUnet\model\TU_Kitti64x1024\TU_pretrain_R50-ViT-B_16_skip3_epo150_bs4_64x1024\weights'

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default=DATA_DIRECTORY, help='root dir for data')
parser.add_argument('--restore_from', type=str,
                    default=DATA_DIRECTORY, help='Which initialisation file')
parser.add_argument('--restore_from_dir', type=str,
                    default=RESTORE_FROM_DIRECTORY, help='root dir for pre-trained weights')
parser.add_argument('--dataset', type=str,
                    default='Kitti', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')

parser.add_argument('--pretrain', action='store_true', default=False, help='Enabling pretraining')
parser.add_argument('--contrastive', action='store_true', default=False, help='Enabling contrastive learning')
parser.add_argument('--use_salsa', action='store_true', default=False, help='use salsaNext module')
parser.add_argument('--train_fr_scratch', action='store_true', default=False, help='not use pre-trained model')
parser.add_argument('--use_transunet_enc_dec', action='store_true', default=False, help='use the decoder and encoder blocks from the TransUnet architecture else use those from salsaNext architecture')
parser.add_argument('--remove_Transformer', action='store_true', default=False, help='remove the Transformer from the architecture')
parser.add_argument('--low-dim', default=512, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-k', default=4096, type=int, #default=4096
                    metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-t', default=0.07, type=float, 
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    help='momentum for non-parametric updates')  
parser.add_argument(
  '--data_kitti_cfg', '-dck',
  type=str,
  required=False,
  default='datasets/lidar_dataset/config/labels/semantic-kitti.yaml',
  help='Classification yaml cfg file. See /config/labels for sample. No default!',
)
parser.add_argument(
  '--data_nuscenes_cfg', '-dcn',
  type=str,
  required=False,
  default='datasets/lidar_dataset/config/labels/semantic-nuscenes.yaml',
  help='Classification yaml cfg file. See /config/labels for sample. No default!',
)
parser.add_argument(
  '--arch_cfg', '-ac',
  type=str,
  required=False,
  default='datasets/lidar_dataset/config/arch/sensor_dataset.yaml',
  help='Architecture yaml cfg file. See /config/arch for sample. No default!',
  )
args = parser.parse_args()

def weights_init(m):
    classname = m.__class__.__name__
    if type(m) == nn.Conv2d:#classname.find('Conv') != -1 and classname!= "Conv2dReLU":
        # m.weight.data.uniform_(-0.08 , 0.08)
        nn.init.xavier_uniform_(m.weight)#, gain=nn.init.calculate_gain('relu'))
    elif classname.find('ConvTranspose') != -1:
        # m.weight.data.uniform_(-0.08 , 0.08)
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.uniform_(-0.08 , 0.08)
        # nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif type(m) == nn.Linear:#classname.find('Linear') != -1:
        # m.weight.data.uniform_(-0.08 , 0.08)
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '../data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
    }
    
    
                                   
    # open arch config file
    try:
      print("Opening arch config file %s" % args.arch_cfg)
      ARCH = yaml.safe_load(open(args.arch_cfg, 'r'))
    except Exception as e:
      print(e)
      print("Error opening arch yaml file.")
      quit()

    # open data config file
    try:
      print("Opening data config file %s" % args.data_kitti_cfg)
      DATA_kitti = yaml.safe_load(open(args.data_kitti_cfg, 'r'))
    except Exception as e:
      print(e)
      print("Error opening data yaml file.")
      quit()
      
    
    args.img_size = (ARCH["dataset_kitti"]["sensor"]["img_prop"]["height"], ARCH["dataset_kitti"]["sensor"]["img_prop"]["width"])
    
    # args.num_classes = dataset_config[dataset_name]['num_classes']
    # args.root_path = dataset_config[dataset_name]['root_path']
    # args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size[0])+'x'+str(args.img_size[1])#TU_Synapse224
    #creats a path with a name that has all the properties of the experiment
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')#../model/TU_Synapse224/TU
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+ str(args.img_size[0])+'x'+str(args.img_size[1])
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    
    if args.pretrain:
        training_sequences = DATA_kitti["split"]["pretrain"] #using labeled and unlabled data
        validation_sequences = DATA_kitti["split"]["prevalid"]
        gt=False
    else:
        training_sequences = DATA_kitti["split"]["train"]
        validation_sequences = DATA_kitti["split"]["valid"]
        gt= True
        
    
    kitti_parser = Parser(root=args.root_path,
                          train_sequences=training_sequences,
                          valid_sequences=validation_sequences,
                          test_sequences=None,
                          labels=DATA_kitti["labels"],
                          color_map=DATA_kitti["color_map"],
                          learning_map=DATA_kitti["learning_map"],
                          learning_map_inv=DATA_kitti["learning_map_inv"],
                          sensor=ARCH["dataset_kitti"]["sensor"],
                          max_points=ARCH["dataset_kitti"]["max_points"],
                          batch_size=args.batch_size,
                          workers=ARCH["train"]["workers"],
                          max_iters=None,
                          gt=gt,
                          shuffle_train=True,
                          nuscenes_dataset=False,
                          pretrain=args.pretrain)
                          
    args.num_classes = kitti_parser.get_n_classes()

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size[0] / args.vit_patches_size), int(args.img_size[1] / args.vit_patches_size))
        
    if(args.use_salsa):
        net = SalsaNext(kitti_parser.get_n_classes()).cuda()
    else:
        #after the '\' avoid adding any characters as this would raise error
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes, low_dim=args.low_dim,rm_transformer=args.remove_Transformer,\
        bn_pretrain=args.pretrain, pretrain=args.pretrain, contrastive=args.contrastive, use_tranunet_enc_dec=args.use_transunet_enc_dec, dropout_rate=0.2, eval_uncer=True).cuda()
    
    if args.pretrain or args.use_salsa or args.train_fr_scratch:
        net.apply(weights_init)
    else:
        # net.load_from(weights=np.load(config_vit.pretrained_path))
        
        net.apply(weights_init)
        # #################
        new_params = net.state_dict().copy()
        saved_state_dict = torch.load(args.restore_from_dir + '\\' +args.restore_from+'.pth')

        saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in new_params}
        new_params.update(saved_state_dict) 
        
        net.load_state_dict(new_params)
        # ################

    trainer = {'Kitti': trainer_kitti,}
    
    trainer[dataset_name](args, net, snapshot_path,kitti_parser,ARCH,DATA_kitti)