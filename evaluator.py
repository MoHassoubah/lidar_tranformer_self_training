import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms

import cv2
from matplotlib import pyplot as plt

from avgmeter import *
from ioueval import *
import os, shutil

from losses import MTL_loss
import cv2
from matplotlib import pyplot as plt



def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    return color_range.reshape(256, 1, 3)

def make_log_img(depth_gt, depth_gt_reduced, depth_pred, mask, mask_reduced, pred, color_fn,  gt, pretrain=False):
    
    if pretrain:
        # input should be [depth, pred, gt]
        # make range image (normalized to 0,1 for saving)
        depth_gt = (cv2.normalize(depth_gt, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F) * 255.0).astype(np.uint8)  
        out_img = cv2.applyColorMap(
            depth_gt, get_mpl_colormap('viridis')) * mask[..., None] 
            
        # depth_gt_reduced = (cv2.normalize(depth_gt_reduced, None, alpha=0, beta=1,
                               # norm_type=cv2.NORM_MINMAX,
                               # dtype=cv2.CV_32F) * 255.0).astype(np.uint8)  
        # depth_reduced_img = cv2.applyColorMap(
            # depth_gt_reduced, get_mpl_colormap('viridis')) * mask_reduced[..., None]  
            
        # out_img = np.concatenate([out_img, depth_reduced_img], axis=0)
        
            
        # depth_pred = (cv2.normalize(np.float32(depth_pred), None, alpha=0, beta=1,
                               # norm_type=cv2.NORM_MINMAX,
                               # dtype=cv2.CV_32F) * 255.0).astype(np.uint8)  
        # depth_pred_img = cv2.applyColorMap(
            # depth_pred, get_mpl_colormap('viridis')) * mask[..., None]  
            
        # out_img = np.concatenate([out_img, depth_pred_img], axis=0)
        
        
        # make label prediction
        pred_color = color_fn((pred * mask).astype(np.int32))
            
        out_img = np.concatenate([out_img, pred_color], axis=0)
        
        gt_color = color_fn(gt)
        out_img = np.concatenate([out_img, gt_color], axis=0)
    
    else:
        # make label prediction
        # pred_color = color_fn((pred * mask).astype(np.int32))
            
        out_img = color_fn((pred * mask).astype(np.int32))#np.concatenate([out_img, pred_color], axis=0)
        
        gt_color = color_fn(gt)
        out_img = np.concatenate([out_img, gt_color], axis=0)
        
    return (out_img).astype(np.uint8)
    
def save_img(depth_gt, depth_gt_reduced, depth_pred, proj_mask, proj_mask_reduced, seg_outputs, proj_labels, parser_to_color, i_iter, eval=False):
    
    SAVE_PATH_kitti = '../result_train'
    # if eval:
    output = seg_outputs[0].permute(1,2,0).cpu().numpy()
    output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
    mask_np = proj_mask[0].cpu().numpy()
    gt_np = proj_labels[0].cpu().numpy()
    
    depth_gt_np = depth_gt[0][0].cpu().numpy()
    # depth_gt_reduced_np = depth_gt_reduced[0][0].cpu().numpy()
    # depth_pred_np = depth_pred[0][0].cpu().numpy()
    mask_np = proj_mask[0].cpu().numpy()
    # mask_np_reduced =proj_mask_reduced[0].cpu().numpy()
    out = make_log_img(depth_gt_np, depth_gt_reduced_np, None, mask_np, mask_np_reduced, output, parser_to_color, gt_np, pretrain=eval )
    
    
    # else:
        # output = seg_outputs[0].permute(1,2,0).cpu().numpy()
        # output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        
        
        # mask_np = proj_mask[0].cpu().numpy()
        # gt_np = proj_labels[0].cpu().numpy()
        # out = make_log_img(None,None,None,mask_np,None, output, parser_to_color, gt_np, pretrain=args.pretrain )
        
    # print(name)
    name_2_save = os.path.join(SAVE_PATH_kitti, '_'+str(i_iter) + '.png')
    cv2.imwrite(name_2_save, out)
        
def eval_model(args, model, snapshot_path, parser, use_salsa=False):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
                          
    print("The length of train set is: {}".format((parser.get_valid_size())))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # trainloader = parser.get_train_set()#DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)#,
                             #worker_init_fn=worker_init_fn) #this gave the error Can't pickle local object 'trainer_synapse.<locals>.worker_init_fn'
    
    ########################                         
    valid_loader = parser.get_valid_set()
    device = torch.device("cuda")
    ignore_classes = [0]
    evaluator = iouEval(parser.get_n_classes(),device, ignore_classes)
    
    
    #Empty the TensorBoard directory
    dir_path = snapshot_path+'/log'

    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))
        
    #######################
    

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
    # dice_loss = DiceLoss(num_classes)
    # iter_num = 0
    # max_epoch = args.max_epochs
    # max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    # logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    
    iou = AverageMeter()
    accu = AverageMeter()
            
    ##############################################
    ##############################################
    
    evaluator.reset()
    iou.reset()
    accu.reset()
    
    val_losses = AverageMeter()
    
    model.eval()
    iter_num = 0
    
    with torch.no_grad():
        
        for index, batch_data in enumerate(valid_loader):
            if index % 100 == 0:
                print('%d validation iter processd' % index)
                
            (inputs, proj_mask, targets, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) = batch_data
               
                        
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True).long()
            # reduced_image_batch = reduced_image_batch.to(device, non_blocking=True) # Apply distortion 
            
            outputs = model(inputs)
            
            argmax = outputs.argmax(dim=1)
            
            evaluator.addBatch(argmax, targets)
            
            # if iter_num % 50 == 0 and iter_num != 0:
                # save_img(inputs, None, None, proj_mask, None, outputs, targets, parser.to_color, iter_num, eval=True)
                
            
            iter_num = iter_num + 1
        jaccard, class_jaccard = evaluator.getIoU()
        accura = evaluator.getacc()
        
        iou.update(jaccard.item(), args.batch_size)#in_vol.size(0)) 
        accu.update(accura.item(), args.batch_size)#in_vol.size(0)) 

    
    for i, jacc in enumerate(class_jaccard):
        print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
        i=i, class_str=parser.get_xentropy_class_string(i), jacc=round(jacc.item() * 100, 2)))
    print('===> mIoU: ' + str(round(iou.avg * 100, 2)))
    
    print('===> mAccuracy: ' + str(round(accu.avg * 100, 2)))
        
        
        ##############################################
        ##############################################


    return "model evaluation Finished!"

        
def eval_noise_robustness(args, model, snapshot_path, parser, use_salsa=False):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
                          
    print("The length of train set is: {}".format((parser.get_valid_size())))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # trainloader = parser.get_train_set()#DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)#,
                             #worker_init_fn=worker_init_fn) #this gave the error Can't pickle local object 'trainer_synapse.<locals>.worker_init_fn'
    
    ########################                         
    valid_loader = parser.get_valid_set()
    device = torch.device("cuda")
    ignore_classes = [0]
    evaluator = iouEval(parser.get_n_classes(),device, ignore_classes)
    
    
    #Empty the TensorBoard directory
    dir_path = snapshot_path+'/log'

    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))
        
    #######################
    

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
    # dice_loss = DiceLoss(num_classes)
    writer = SummaryWriter(snapshot_path + '/log')
    # iter_num = 0
    # max_epoch = args.max_epochs
    # max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    # logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    
    drop_ratio_list = [0.1,0.15,0.2,0.25]
    
    for drp_i in drop_ratio_list:
        iou = AverageMeter()
        
        parser.set_eval_drop_percentage(drp_i)
        
        ##############################################
        ##############################################
        
        evaluator.reset()
        iou.reset()
        
        val_losses = AverageMeter()
        
        model.eval()
        with torch.no_grad():
            
            for index, batch_data in enumerate(valid_loader):
                if index % 100 == 0:
                    print('%d validation iter processd' % index)

                (image_batch, proj_mask, label_batch, reduced_image_batch, reduced_proj_mask, \
                        path_seq, path_name) =  batch_data      
                        
                reduced_image_batch, label_batch = reduced_image_batch.cuda(), label_batch.cuda(non_blocking=True).long()
                # reduced_image_batch = reduced_image_batch.to(device, non_blocking=True) # Apply distortion 
                
                outputs = model(reduced_image_batch)
                
                loss_ce = ce_loss(outputs, label_batch)
                
                
                val_losses.update(loss_ce.mean().item(), args.batch_size)
                
                argmax = outputs.argmax(dim=1)
                
                evaluator.addBatch(argmax, label_batch)
                      
            jaccard, class_jaccard = evaluator.getIoU()
            
            iou.update(jaccard.item(), args.batch_size)#in_vol.size(0)) 

        writer.add_scalar('eval/iou', iou.avg, int(drp_i*100))
            
        writer.add_scalar('eval/loss', val_losses.avg, int(drp_i*100))
            
            
        ##############################################
        ##############################################


    writer.close()
    return "Training Finished!"
    
def set_training_mode_for_dropout(net, training=True):
    """Set Dropout mode to train or eval."""

    for m in net.modules():
#        print(m.__class__.__name__)
        if m.__class__.__name__.startswith('Dropout'):
            if training==True:
                m.train()
            else:
                m.eval()
    return net        

def one_hot_pred_from_label(y_pred, labels):
    y_true = torch.zeros_like(y_pred)#y_pred->batch_size*Num_classes*H*W
    ones = torch.ones_like(y_pred)
    indexes = [l for l in labels]
    y_true[torch.arange(labels.size(0)), indexes] = ones[torch.arange(labels.size(0)), indexes]
    
    return y_true

def compute_log_likelihood(y_pred, y_true, sigma): #y_pred->batch_size*num_classes
    dist = torch.distributions.normal.Normal(loc=y_pred, scale=sigma)
    log_likelihood = dist.log_prob(y_true)#log_prob->Returns the log of the probability density/mass function evaluated at value.->batch_size*num_classes
    # print('1st mean')
    # print(torch.mean(log_likelihood, dim=3).shape)
    # print('2nd mean')
    # print(torch.mean(torch.mean(log_likelihood, dim=3),dim=2).shape)
    log_likelihood = torch.mean(torch.mean(torch.mean(log_likelihood, dim=3),dim=2), dim=1)# size batch_size
    return log_likelihood

def compute_brier_score(y_pred, y_true):
    """Brier score implementation follows 
    https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf.
    The lower the Brier score is for a set of predictions, the better the predictions are calibrated."""        
        
    brier_score = torch.mean(torch.mean(torch.mean((y_true-y_pred)**2, dim=3),dim=2), dim=1)
    return brier_score

def compute_preds(args, net, inputs, use_salsa, use_mcdo=False):
    
    model_variance = None
    
    def keep_variance(x, min_variance):
        return x + min_variance

    keep_variance_fn = lambda x: keep_variance(x, min_variance=args.min_variance)
    softmax = nn.Softmax(dim=1)
    
    net.eval()
    if use_mcdo:
        net = set_training_mode_for_dropout(net, True)
        outputs = [net(inputs) for i in range(args.num_samples)]
        
        if(use_salsa):
            outputs_mean = outputs#[outs for outs in outputs]#we gone take mean after that so softmax here is must to make them same domain
        else:
            outputs_mean = [softmax(outs) for outs in outputs]#we gone take mean after that so softmax here is must to make them same domain
            
        outputs_mean = torch.stack(outputs_mean) # num_samples*batch_size*num_classes
        model_variance = torch.var(outputs_mean, dim=0)
        # Compute MCDO prediction
        outputs_mean = torch.mean(outputs_mean, dim=0)
    else:
        outputs = net(inputs)
        outputs_mean = outputs
        
    net = set_training_mode_for_dropout(net, False)
    
    return outputs_mean, model_variance


def evaluate_uncertainity(args, net, snapshot_path, parser, use_salsa=False, use_mcdo=True):
    net.eval()
    test_loss = 0
    correct = 0
    brier_score = 0
    neg_log_likelihood = 0
    total = 0
    correct_total=0
    outputs_variance = None
    
    ##############
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    criterion = torch.nn.NLLLoss(ignore_index=255)#torch.nn.CrossEntropyLoss(ignore_index=255)
    
    valid_loader = parser.get_valid_set()
    ##############
    
    with torch.no_grad():
        for index, batch_data in enumerate(valid_loader):
            if index % 100 == 0:
                print('%d validation iter processd' % index)

            (inputs, proj_mask, targets, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) = batch_data
               
                        
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True).long()
            # reduced_image_batch = reduced_image_batch.to(device, non_blocking=True) # Apply distortion 
                
            # print("labels shape")
            # print(targets.shape)
            # print("inputs shape")
            # print(inputs.shape)
            
            outputs_mean, model_variance = compute_preds(args, net, inputs, use_salsa, use_mcdo)
            
            if model_variance is not None:
                outputs_variance = model_variance + args.tau
                
            
            # print("outputs_mean shape")
            # print(outputs_mean.shape)
            # print("model_variance shape")
            # print(model_variance.shape)
            
            # one_hot_targets = one_hot_pred_from_label(outputs_mean, targets)# for each image add 1 at the column of the targer class
            one_hot_targets = torch.nn.functional.one_hot(targets,parser.get_n_classes())
            one_hot_targets=one_hot_targets.permute(0,3,1,2)
            # print("one_hot_targets shape")
            # print(one_hot_targets.shape)
            # Compute negative log-likelihood (if variance estimate available)
            if outputs_variance is not None:
                outputs_sigma = torch.pow(outputs_variance, 0.5)
                batch_log_likelihood = compute_log_likelihood(outputs_mean, one_hot_targets, outputs_sigma)
                
                # print("Last mean shape")
                # print(batch_log_likelihood.shape)
                batch_neg_log_likelihood = -batch_log_likelihood
                # Sum along batch dimension
                neg_log_likelihood += torch.sum(batch_neg_log_likelihood, 0).cpu().numpy().item()
            
            # Compute brier score
            batch_brier_score = compute_brier_score(outputs_mean, one_hot_targets)
            # Sum along batch dimension
            brier_score += torch.sum(batch_brier_score, 0).cpu().numpy().item()
            
            # Compute loss
            loss = criterion(torch.log(outputs_mean.clamp(min=1e-8)), targets)#(outputs_mean, targets)
            test_loss += loss.item()#alrady averged over batch size
            
            # Compute predictions and numer of correct predictions
            _, predicted = outputs_mean.max(1)
            # print("predicted shape")
            # print(predicted.shape)
            total += targets.size(0)
            correct_total += targets.size(0)*targets.size(1)*targets.size(2)
            correct += predicted.eq(targets).sum().item()

            # if args.show_bar and args.verbose:
                # progress_bar(index, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    # % (test_loss/(index+1), 100.*correct/total, correct, total))
                    
            
            logging.info('Batch_index %d : Loss : %f, Acc : %f, (%d/%d)' % (index, test_loss/(index+1), \
            100.*correct/correct_total, correct, correct_total))

    accuracy = 100.*correct/correct_total
    brier_score = brier_score/total
    neg_log_likelihood = neg_log_likelihood/total
    
    logging.info('>>>>>>>>>>>>>>>>>>Final Results<<<<<<<<<<<<<<<<<<<<<<')
    
    logging.info('Brier Score : %f, Acc : %f, Neg-Log-Likelihood : %f' % (brier_score, \
    accuracy, neg_log_likelihood))
    return accuracy, brier_score, neg_log_likelihood

