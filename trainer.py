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
import torch.nn.functional as F
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
from networks.Lovasz_Softmax import Lovasz_softmax
from warmupLR import *

from NCE.NCEAverage import NCEAverage

from torch.nn import functional as F


def NN(epoch, net,lower_dim, NCE_valLoader, valLoader):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    losses = AverageMeter()
    correct = 0.
    total = 0
    
    device = torch.device("cuda")
    testsize = valLoader.dataset.__len__()
    valNCEsize = NCE_valLoader.dataset.__len__()
    
    trainFeatures=torch.zeros((lower_dim,valNCEsize)).cuda()
    indicies = torch.zeros(valNCEsize).cuda()
    lossF = torch.nn.L1Loss()
    
    val_losses = AverageMeter()
    print("####################################passed the allocation ")
    
    temploader = torch.utils.data.DataLoader(NCE_valLoader.dataset, batch_size=10, shuffle=False, num_workers=1)
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(temploader):
            if batch_idx % 100 == 0:
                print('%d NCE db iter processd' % batch_idx)
            
            (index, image_batch, proj_mask, reduced_image_batch, reduced_proj_mask, path_seq, path_name) =  batch_data
            
            reduced_image_batch = reduced_image_batch.to(device, non_blocking=True) # Apply distortion
            index = index.cuda()
            
            batchSize = reduced_image_batch.size(0)
            features, _, _, _,_  = net(reduced_image_batch)#features of the training data with the transform of the testing data
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
            indicies[batch_idx*batchSize:batch_idx*batchSize+batchSize] = index
        
        print("####################################pass teh first for loop")
        end = time.time()
        for  batch_idx, batch_data in enumerate(valLoader):
            (index, image_batch, proj_mask, reduced_image_batch, reduced_proj_mask, path_seq, path_name) =  batch_data
        
            image_batch = image_batch.to(device, non_blocking=True)
            reduced_image_batch = reduced_image_batch.to(device, non_blocking=True) # Apply distortion
            index = index.cuda()
            
            batchSize = reduced_image_batch.size(0)
            features, recon_prd, _, _ ,_  = net(reduced_image_batch)#features of the training data with the transform of the testing data
            
            loss_v = lossF(recon_prd, image_batch)
            val_losses.update(loss_v.mean().item(), batchSize)
            
            net_time.update(time.time() - end)
            end = time.time()
            #features dim=(batchsize,lower_dim) * trainFeatures dim=(lower_dim,#samples in dataset)-> result dim=(batchsize, #samples in dataset)
            dist = torch.mm(features, trainFeatures)

            #(values yd, indices yi)
            yd, yi = dist.topk(1, dim=1, largest=True, sorted=True)#Returns the k largest elements of the given input tensor along a given dimension.

            total += batchSize
            
            # start_index = (index*500)+index
            # correct += torch.logical_and(start_index <= yi, (start_index +500) >=yi).sum().item()
            
            correct += torch.logical_and(index -2 <= yi, (index +2) >=yi).sum().item()
            # correct += index.eq(yi.data).sum().item()
            
            cls_time.update(time.time() - end)
            end = time.time()

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}'.format(
                  total, testsize, correct*100./total, net_time=net_time, cls_time=cls_time))
            # print("index")
            # print(index)
            # print("yi.data")
            # print(yi.data)

    return correct*100./total, val_losses



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
            
        depth_gt_reduced = (cv2.normalize(depth_gt_reduced, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F) * 255.0).astype(np.uint8)  
        depth_reduced_img = cv2.applyColorMap(
            depth_gt_reduced, get_mpl_colormap('viridis')) * mask_reduced[..., None]  
            
        out_img = np.concatenate([out_img, depth_reduced_img], axis=0)
        
            
        depth_pred = (cv2.normalize(np.float32(depth_pred), None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F) * 255.0).astype(np.uint8)  
        depth_pred_img = cv2.applyColorMap(
            depth_pred, get_mpl_colormap('viridis')) * mask[..., None]  
            
        out_img = np.concatenate([out_img, depth_pred_img], axis=0)
    
    else:
        # make label prediction
        # pred_color = color_fn((pred * mask).astype(np.int32))
            
        out_img = color_fn((pred * mask).astype(np.int32))#np.concatenate([out_img, pred_color], axis=0)
        
        gt_color = color_fn(gt)
        out_img = np.concatenate([out_img, gt_color], axis=0)
        
    return (out_img).astype(np.uint8)
    
def save_img(depth_gt, depth_gt_reduced, depth_pred, proj_mask, proj_mask_reduced, seg_outputs, proj_labels, parser_to_color, i_iter, pretrain=False):
    
    SAVE_PATH_kitti = '../result_train'
    if not pretrain:
        output = seg_outputs[0].permute(1,2,0).cpu().numpy()
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        
        
        mask_np = proj_mask[0].cpu().numpy()
        gt_np = proj_labels[0].cpu().numpy()
        out = make_log_img(None,None,None,mask_np,None, output, parser_to_color, gt_np, pretrain=args.pretrain )
    else:
        depth_gt_np = depth_gt[0][0].cpu().numpy()
        depth_gt_reduced_np = depth_gt_reduced[0][0].cpu().numpy()
        depth_pred_np = depth_pred[0][0].cpu().numpy()
        mask_np = proj_mask[0].cpu().numpy()
        mask_np_reduced =proj_mask_reduced[0].cpu().numpy()
        out = make_log_img(depth_gt_np, depth_gt_reduced_np, depth_pred_np, mask_np, mask_np_reduced, None, parser_to_color, None, pretrain=pretrain )
        
    # print(name)
    name_2_save = os.path.join(SAVE_PATH_kitti, '_'+str(i_iter) + '.png')
    cv2.imwrite(name_2_save, out)


    
def trainer_kitti(args, model, snapshot_path, parser,ARCH=None,DATA=None):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
                          
    print("The length of train set is: {}".format((parser.get_train_size())))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = parser.get_train_set()#DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)#,
                             #worker_init_fn=worker_init_fn) #this gave the error Can't pickle local object 'trainer_synapse.<locals>.worker_init_fn'
    
    ########################   
    NCEvalid_loader = parser.get_valid_set_NCE()
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
        
    # for filename in os.listdir(dir_path):
        # file_path = os.path.join(dir_path, filename)
        # try:
            # if os.path.isfile(file_path) or os.path.islink(file_path):
                # os.unlink(file_path)
            # elif os.path.isdir(file_path):
                # shutil.rmtree(file_path)
        # except Exception as e:
            # print('Failed to delete %s. Reason: %s' % (file_path, e))
    #######################
    
    
    ######################
    ######################
    if args.pretrain:
        if args.contrastive:
            train_data_size = parser.get_num_train_scans()
            # nce-k 4096 --nce-t 0.07 --nce-m 0.5 --low-dim 128
            lemniscate = NCEAverage(args.low_dim, train_data_size, args.nce_k, args.nce_t, args.nce_m).cuda()
            criterion = MTL_loss(device, args.batch_size, ndata=train_data_size, contrastive_loss=True)
        else:
            criterion = MTL_loss(device, args.batch_size)
    ######################
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    ls = Lovasz_softmax(ignore=0).to(device)
    
    if(args.use_salsa):
        epsilon_w = ARCH["train"]["epsilon_w"]
        content = torch.zeros(parser.get_n_classes(), dtype=torch.float)
        for cl, freq in DATA["content"].items():
            x_cl = parser.to_xentropy(cl)  # map actual class to xentropy class
            content[x_cl] += freq
        loss_w = 1 / (content + epsilon_w)  # get weights
        for x_cl, w in enumerate(loss_w):  # ignore the ones necessary to ignore
            if DATA["learning_ignore"][x_cl]:
                # don't weigh
                loss_w[x_cl] = 0
        print("Loss weights from content: ", loss_w.data)

        ce_loss = nn.NLLLoss(weight=loss_w).to(device)    
        
        optimizer = optim.SGD([{'params': model.parameters()}],
                                       lr=ARCH["train"]["lr"],
                                       momentum=ARCH["train"]["momentum"],
                                       weight_decay=ARCH["train"]["w_decay"])
        # Use warmup learning rate
        # post decay and step sizes come in epochs and we want it in steps
        steps_per_epoch = parser.get_train_size()
        up_steps = int(ARCH["train"]["wup_epochs"] * steps_per_epoch)
        final_decay = ARCH["train"]["lr_decay"] ** (1 / steps_per_epoch)
        scheduler = warmupLR(optimizer=optimizer,
                                  lr=ARCH["train"]["lr"],
                                  warmup_steps=up_steps,
                                  momentum=ARCH["train"]["momentum"],
                                  decay=final_decay)
    else:
        ce_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
            
        # dice_loss = DiceLoss(num_classes)
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

                              
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        
        iou = AverageMeter()
          
        for i_batch, batch_data in enumerate(trainloader):
            if args.pretrain:
                (index, image_batch, proj_mask, reduced_image_batch, reduced_proj_mask, \
                rot_ang_around_z_axis_batch,path_seq, path_name) =  batch_data
            
                if args.use_transunet_enc_dec:
                    image_batch = reduced_image_batch.to(device, non_blocking=True) # Apply distortion
                else:
                    image_batch = image_batch.to(device, non_blocking=True)
                index = index.cuda()
                
                with torch.cuda.amp.autocast():
                    if args.contrastive:
                        contrastive_prd, recon_prd, contrastive_w, recons_w, nce_converge_w = model(image_batch)
                        batchSize = image_batch.size(0)
                        weight_prev_cycle = torch.index_select(lemniscate.memory, 0, index.view(-1))
                        weight_prev_cycle.resize_(batchSize,args.low_dim)
                        w_test =weight_prev_cycle.reshape(batchSize, 1,args.low_dim)
                        x_norm = contrastive_prd.reshape(batchSize, args.low_dim, 1)
                        x_norm = F.normalize(x_norm, dim=1).data
                        out = torch.bmm(w_test, x_norm).squeeze(1).squeeze(1)
                        output_P_i_v = lemniscate(contrastive_prd, index)
                        
                        loss, (loss1, loss2, loss3) = criterion(output_P_i_v, index,
                                        recon_prd, image_batch, contrastive_w, recons_w, \
                                        contrastive_prd, weight_prev_cycle, nce_converge_w)
                    else:
                        recon_prd, rot_w, contrastive_w, recons_w = model(image_batch)
                        
                        rot_p = None#torch.cat([rot_prd, rot_prd_2], dim=0).squeeze(1)
                        rots = None#torch.cat([rot_ang_around_z_axis_batch, rot_ang_around_z_axis_batch_2], dim=0) 
                        # rots = rots.type_as(rot_p)
                        
                        # print("target rots")
                        # print(rots.shape)
                        # print(rots.dtype)
                        
                        # print("predicted rots")
                        # print(rot_p.shape)
                        
                        # imgs_recon = recon_prd#torch.cat([recon_prd, recon_prd_2], dim=0) 
                        # imgs = image_batch#torch.cat([image_batch, image_batch_2], dim=0) 
                        
                        # loss, (loss1, loss2, loss3) = criterion(rot_p, rots, 
                                                                    # None, None, 
                                                                    # imgs_recon, imgs, rot_w, contrastive_w, recons_w )
                        loss, (loss1, loss2, loss3) = criterion(None, None,
                                        recon_prd, image_batch, None, None, \
                                        None, None, None)
                  
                if args.contrastive:
                    writer.add_scalar('info/P_i_v', output_P_i_v[:,0].mean().item(), iter_num)  
                    writer.add_scalar('info/P_i_v_dash', output_P_i_v[:,1:].mean().item(), iter_num) 
                    writer.add_scalar('info/vt_x_f', out.mean().item(), iter_num)   
                    writer.add_scalar('info/loss_NCE', loss1, iter_num)
                    writer.add_scalar('info/loss_convergence', loss2, iter_num)
                    writer.add_scalar('info/loss_reconstruction', loss3, iter_num)
                # writer.add_scalar('info/loss_rotation', loss1, iter_num)
                # writer.add_scalar('info/loss_contrastive', loss2, iter_num)
                # writer.add_scalar('info/loss_reconstruction', loss3, iter_num)
                #loss1->rotation loss
                #loss2->rotation axis loss
                #loss3->contrastive loss
                #loss4->reconstruction loss
                logging.info('iteration %d : loss : %f, loss1 : %f, loss2 : %f, loss3 : %f' % (iter_num, loss.item(), \
                loss1.item(), loss2.item(), loss3.item()))
                
                # logging.info('iteration %d : rot_axis_w : %f, contrastive_w : %f, recons_w : %f' % (iter_num, \
                # rot_axis_w.item(), contrastive_w.item(), recons_w.item()))
                
                # with torch.no_grad():
                    # if iter_num % 50 == 0 and iter_num != 0:
                        # save_img(image_batch, reduced_image_batch, recon_prd, proj_mask, reduced_proj_mask, None, None, parser.to_color, iter_num, pretrain=True)
            else:
                (image_batch, proj_mask, label_batch, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) = batch_data
                
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda(non_blocking=True).long()
                outputs = model(image_batch)
            
                if(args.use_salsa):
                    loss_ce = ce_loss(torch.log(outputs.clamp(min=1e-8)), label_batch) + ls(outputs, label_batch.long())
                else:
                    loss_ce = ce_loss(outputs, label_batch) + ls(F.softmax(outputs, dim=1), label_batch)
                # loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = loss_ce 
                
                ###########################
                
                with torch.no_grad():
                    evaluator.reset() # we do this for the training as each weights of the model differ each iteration 
                    argmax = outputs.argmax(dim=1)
                    evaluator.addBatch(argmax, label_batch)
                    jaccard, class_jaccard = evaluator.getIoU()
                iou.update(jaccard.item(), args.batch_size)
                ###########################
                
                logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
                
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if(args.use_salsa):
                # step scheduler
                scheduler.step()
            else:
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                

            iter_num = iter_num + 1
            if(args.use_salsa==False):
                writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/loss', loss, iter_num)

            # if iter_num % 20 == 0:
                # image = image_batch[1, 0:1, :, :]
                # image = (image - image.min()) / (image.max() - image.min())
                # writer.add_image('train/Image', image, iter_num)
                # outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                # writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                # labs = label_batch[1, ...].unsqueeze(0) * 50
                # writer.add_image('train/GroundTruth', labs, iter_num)
                
        if not args.pretrain:
            writer.add_scalar('train/iou', iou.avg, epoch_num)
                
                
        ##############################################
        ##############################################
        if args.pretrain:
            valid_interval=5
        else:
            valid_interval=10
            
        if (epoch_num + 1) % 1 == 0:
            evaluator.reset()
            iou.reset()
            
            val_losses = AverageMeter()
            
            model.eval()
            with torch.no_grad():
                
                
                        
                if args.pretrain:
                    if args.contrastive:
                        correct_percentage,val_losses = NN(epoch_num, model,args.low_dim, NCEvalid_loader, valid_loader)
                        writer.add_scalar('valid/NCE_val', correct_percentage, epoch_num)
                    else:
                        for index, batch_data in enumerate(valid_loader):
                            if index % 100 == 0:
                                print('%d validation iter processd' % index)
                            (image_batch, proj_mask, reduced_image_batch, reduced_proj_mask, \
                            rot_ang_around_z_axis_batch,path_seq, path_name) =  batch_data
                            
                            if args.use_transunet_enc_dec:
                                image_batch = reduced_image_batch.to(device, non_blocking=True) # Apply distortion
                            else:
                                image_batch = image_batch.to(device, non_blocking=True)
                            
                            with torch.cuda.amp.autocast():
                            
                                recon_prd, rot_w, contrastive_w, recons_w = model(image_batch)
                                # rot_prd_2, contrastive_prd_2, recon_prd_2, _, _, _                         = model(reduced_image_batch_2)
                                    
                                rot_p = None#torch.cat([rot_prd, rot_prd_2], dim=0).squeeze(1)
                                rots = None#torch.cat([rot_ang_around_z_axis_batch, rot_ang_around_z_axis_batch_2], dim=0) 
                                # rots = rots.type_as(rot_p)
                                
                                # print("target rots")
                                # print(rots.shape)
                                # print(rots.dtype)
                                
                                # print("predicted rots")
                                # print(rot_p.shape)
                                
                                imgs_recon = recon_prd#torch.cat([recon_prd, recon_prd_2], dim=0) 
                                imgs = image_batch#torch.cat([image_batch, image_batch_2], dim=0) 
                                
                                loss, (loss1, loss2, loss3) = criterion(rot_p, rots, 
                                                                    None, None, 
                                                                    imgs_recon, imgs, rot_w, contrastive_w, recons_w )
                                                                            
                            val_losses.update(loss.mean().item(), args.batch_size)
                        
                else:
                
                    for index, batch_data in enumerate(valid_loader):
                        if index % 100 == 0:
                            print('%d validation iter processd' % index)
                            (image_batch, proj_mask, label_batch, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) =  batch_data               
                            image_batch, label_batch = image_batch.cuda(), label_batch.cuda(non_blocking=True).long()
                            outputs = model(image_batch)
                            
                            # loss_ce = ce_loss(outputs, label_batch)
                
                            
                            if(args.use_salsa):
                                loss_ce = ce_loss(torch.log(outputs.clamp(min=1e-8)), label_batch) + ls(outputs, label_batch.long())
                            else:
                                loss_ce = ce_loss(outputs, label_batch) + ls(F.softmax(outputs, dim=1), label_batch)
                            
                            val_losses.update(loss_ce.mean().item(), args.batch_size)
                            
                            argmax = outputs.argmax(dim=1)
                            
                            evaluator.addBatch(argmax, label_batch)
                    
                if not args.pretrain:    
                    jaccard, class_jaccard = evaluator.getIoU()
                    
                    iou.update(jaccard.item(), args.batch_size)#in_vol.size(0)) 

            if not args.pretrain: 
                writer.add_scalar('valid/iou', iou.avg, epoch_num)
                
            writer.add_scalar('valid/loss', val_losses.avg, epoch_num)
                
            
        ##############################################
        ##############################################

            
        if args.pretrain:
            save_interval = 5  # int(max_epoch/6)
        else:
            save_interval = 10 
                
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        if (epoch_num + 1) % 1 == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
    
