import os
import random
import time
import cv2
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
#import apex
from tensorboardX import SummaryWriter

from model.capl import FewShotSeg
from torch.optim.lr_scheduler import MultiStepLR
from util import dataset_panet_capl as dataset
from util import transform_capl as transform
from util import config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('--save2shm', type=bool, default=False, help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg, args.save2shm


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger
  

def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args, save2shm = get_parser()
    args.save2shm = save2shm
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    assert len(args.train_gpu) == 1, 'Error | only single GPU training is enabled.'
    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)

def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    BatchNorm = nn.BatchNorm2d


    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    model = FewShotSeg(args=args, backbone=args.backbone)
    modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]

    modules_new = [model.gamma_conv,  model.down]
    modules_new.append(model.beta_conv)     
    modules_new.append(model.attn_conv)      
    modules_new.append(model.fem_module)                       
                         
    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.lr * 10))
    params_list.append(dict(params=model.main_proto, lr=args.lr * 10))

    optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)


    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)

    model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    assert args.split in [0, 1, 2, 3, 10, 11, 999]
    trans_list = [
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
    train_transform = transform.Compose(trans_list)
    train_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root, \
                                data_list=args.train_list, transform=train_transform, mode='train', \
                                all_zero_ratio=args.all_zero_ratio, \
                                use_coco=args.use_coco, 
                                save2shm=args.save2shm)
    print('Training with all_zero_ratio: {}'.format(args.all_zero_ratio))

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    
    if args.evaluate:
        ## evaluate with the original labels
        val_transform = transform.Compose([
            transform.Resize(size=args.train_h),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])            
        val_data = dataset.SemData(split=args.split, shot=args.val_shot, data_root=args.data_root, \
                                data_list=args.val_list, transform=val_transform, mode='val', \
                                use_coco=args.use_coco, save2shm=args.save2shm)

        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    max_iou = 0.
    filename = 'happycoding.pth'
    epochs = args.n_steps // len(train_loader) + 1
    if args.use_coco:
        epochs = args.n_steps // args.epoch_size + 1
    args.epochs = epochs
    print("epochs: {}".format(args.epochs))
    print(args)
    if args.use_coco:
        test_num = 10000
    else:
        test_num = args.test_num

    if args.only_evaluate:
        loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou = validate(val_loader, model, criterion, test_num=test_num)  
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, optimizer, criterion, scheduler, epoch)
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)     

        if args.evaluate:
            loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou = validate(val_loader, model, criterion, test_num=test_num)
            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('class_miou_val', class_miou, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
            if class_miou > max_iou and epoch_log > args.save_freq:
                max_iou = class_miou
                if os.path.exists(filename):
                    os.remove(filename)            
                filename = args.save_path + '/train_epoch_' + str(epoch) + '_'+str(max_iou)+'.pth'
                logger.info('Saving checkpoint to: ' + filename)
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)

    filename = args.save_path + '/final.pth'
    logger.info('Saving checkpoint to: ' + filename)
    torch.save({'epoch': args.epochs, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)                

def train(train_loader, model, optimizer, criterion, scheduler, epoch):

    if args.manual_seed  is not None:
        torch.cuda.manual_seed(args.manual_seed + epoch)
        np.random.seed(args.manual_seed + epoch)
        torch.manual_seed(args.manual_seed + epoch)
        torch.cuda.manual_seed_all(args.manual_seed + epoch)
        random.seed(args.manual_seed + epoch)

    torch.cuda.empty_cache()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()                

    end = time.time()
    max_iter = min(args.n_steps, args.epochs*len(train_loader))
    for i, (qry_imgs, target, raw_label, s_input, s_mask, s_bg_mask, s_raw_label, subcls) in enumerate(train_loader):
        if i != 0 and i % args.epoch_size == 0:
            break

        data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1

        s_input = s_input.cuda(non_blocking=True) #[batch_size, shot, channel, h, w]
        s_mask = s_mask.cuda(non_blocking=True) #[batch_size, shot, h, w]
        s_bg_mask = s_bg_mask.cuda(non_blocking=True) #[batch_size, shot, h, w]
        s_raw_label = s_raw_label.cuda(non_blocking=True) #[batch_size, shot, h, w]
        qry_imgs = qry_imgs.cuda(non_blocking=True) # [batch_size, channel, h, w]
        target = target.cuda(non_blocking=True) #[batch_size, h, w]
        raw_label = raw_label.cuda(non_blocking=True) #[batch_size, h, w]


        s_mask = (s_mask == 1).float() 
        s_bg_mask = (s_bg_mask == 1).float() 
        supp_imgs = [[s_input[:,i,:,:,:] for i in range(args.shot)]]
        fore_mask = [[s_mask[:,i,:,:] for i in range(args.shot)]]
        back_mask = [[s_bg_mask[:,i,:,:] for i in range(args.shot)]]
        query_imgs = [qry_imgs]
        query_labels = target
        optimizer.zero_grad()      

        pred, _, align_loss, main_loss = model(supp_imgs, fore_mask, back_mask, query_imgs, s_raw_label, raw_label, i)
        query_loss = criterion(pred, query_labels)
        loss =  query_loss + align_loss + main_loss 
        if i % 10 == 0:
            print('  query:{:.4f}, align: {:.4f}, main: {:.4f}'.format(query_loss,  align_loss, main_loss))            
        loss.backward()

        optimizer.step()

        scheduler.step()

        n = args.batch_size
        assert args.batch_size == qry_imgs.size(0)

        output = pred.argmax(1)    

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        main_loss = query_loss
        aux_loss = align_loss
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter.update(aux_loss, n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        
        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            avg_Acc = intersection_meter.sum / (target_meter.sum + 1e-10)
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} ({main_loss_meter.avg:.3f}) '
                        'AuxLoss {aux_loss_meter.val:.4f} ({aux_loss_meter.avg:.3f}) '                        
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.3f}) ' 
                        'Accuracy {accuracy:.4f} ({avg_Acc:.3f}).'.format(epoch+1, args.epochs, i + 1, min(len(train_loader),args.epoch_size),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          aux_loss_meter=aux_loss_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy,
                                                          avg_Acc=np.mean(avg_Acc)))
        if main_process():
            writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))        
    return main_loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion, test_num=2000):
    
    if args.manual_seed is not None:
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    torch.cuda.empty_cache()
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    if args.use_coco:
        split_gap = 20
    else:
        split_gap = 5

    class_intersection_meter = [0]*split_gap
    class_union_meter = [0]*split_gap  

    model.eval()
    end = time.time()

    assert test_num % args.batch_size_val == 0    
    iter_num = 0
    for e in range(10):
        for i, (qry_imgs, target, raw_label, s_input, s_mask, s_bg_mask, s_raw_label, subcls, ori_label) in enumerate(val_loader):
            iter_num += 1
            if (iter_num-1) * args.batch_size_val >= test_num:
                break
            data_time.update(time.time() - end)

            s_input = s_input.cuda(non_blocking=True) #[batch_size, shot, channel, h, w]
            s_mask = s_mask.cuda(non_blocking=True) #[batch_size, shot, h, w]
            s_bg_mask = s_bg_mask.cuda(non_blocking=True) #[batch_size, shot, h, w]
            qry_imgs = qry_imgs.cuda(non_blocking=True) # [batch_size, channel, h, w]
            target = target.cuda(non_blocking=True) #[batch_size, h, w]
            ori_label = ori_label.cuda(non_blocking=True)

            s_mask = (s_mask == 1).float()
            s_bg_mask = (s_bg_mask == 1).float()
            supp_imgs = [[s_input[:,i,:,:,:] for i in range(args.val_shot)]]
            fore_mask = [[s_mask[:,i,:,:] for i in range(args.val_shot)]]
            back_mask = [[s_bg_mask[:,i,:,:] for i in range(args.val_shot)]]
            query_imgs = [qry_imgs]
            query_labels = torch.cat([target.long()], 0)
            with torch.no_grad():
                pred, align_loss = model(supp_imgs, fore_mask, back_mask, query_imgs, s_raw_label, raw_label, i)

            query_loss = criterion(pred, query_labels)
            loss = query_loss + align_loss 

            ### evaluate with original label without being resizd
            longerside = max(ori_label.size(1), ori_label.size(2))
            backmask = torch.ones(ori_label.size(0), longerside, longerside).cuda()*255
            backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
            target = backmask.clone().long()
            pred = F.interpolate(pred, size=target.size()[1:], mode='bilinear', align_corners=True)  
            output = pred.argmax(1) 

            n = args.batch_size_val
            assert n == qry_imgs.size(0)

            intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            intersection, union, target, new_target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)
                
            subcls = subcls[0].cpu().numpy()[0]
            class_intersection_meter[(subcls-1)%split_gap] += intersection[1]
            class_union_meter[(subcls-1)%split_gap] += union[1] 

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), qry_imgs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % (test_num/100) == 0) and main_process():
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num* args.batch_size_val, test_num,
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              loss_meter=loss_meter,
                                                              accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i]/(class_union_meter[i]+ 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
    class_miou = class_miou*1.0 / len(class_intersection_meter)
    logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))
    for i in range(split_gap):
        logger.info('Class_{} Result: iou {:.4f}.'.format(i+1, class_iou_class[i]))            
    

    if main_process():
        logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))

        
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc, class_miou


if __name__ == '__main__':
    main()
