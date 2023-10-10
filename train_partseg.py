"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm
from dataset import PartNormalDataset, pipe_dataset
import hydra
import omegaconf

import matplotlib.pyplot as plt

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

@hydra.main(config_path='config', config_name='partseg')
def pt_train(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    # root = hydra.utils.to_absolute_path('data/shapenetcore_partanno_segmentation_benchmark_v0_normal/')

    # added
    if not os.path.exists('/home/tasnim/from_004/Point-Transformers/outputs'):
        os.makedirs('/home/tasnim/from_004/Point-Transformers/outputs')

    test_save_path = '/home/tasnim/from_004/Point-Transformers/predictions'

    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    
    # added
    # device = torch.device("cuda" if args.cuda else "cpu")
    TRAIN_DATASET = pipe_dataset(num_points=args.num_point, partition='trainval')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    TEST_DATASET = pipe_dataset(num_points=args.num_point, partition='test')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.test_batch_size, shuffle=False, num_workers=10)

    # TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='trainval', normal_channel=args.normal)
    # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    # TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    # testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)

# added
    '''MODEL LOADING'''
    args.input_dim = 3
    args.num_class = 3
    # num_category = 16
    # num_part = args.num_class

    '''MODEL LOADING'''
    # args.input_dim = (6 if args.normal else 3) + 16
    # args.num_class = 50
    # num_category = 16
    # num_part = args.num_class

    shutil.copy(hydra.utils.to_absolute_path('models/Hengshuang/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.Hengshuang.model'.format(args.model.name)), 'PointTransformerSeg')(args).cuda()
    
# added
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.CrossEntropyLoss()

    # if you want to load from existing one
    try:
        checkpoint = torch.load('/home/tasnim/from_004/Point-Transformers/outputs/model_9.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0
    '''
    try:
        checkpoint = torch.load('best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0
    '''
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size
    '''
    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0
    '''
    # start_epoch = 0
    plt_train_loss = []
    plt_test_loss = []
    for epoch in range(start_epoch, args.epoch):
        # mean_correct = []

        logger.info('Epoch (%d/%s):' % (epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        logger.info('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        # added training
        
        train_loss = 0.0
        count = 0
        for data, seg in trainDataLoader:
            data, seg = data.float().cuda(), seg.float().cuda()
            #added 2 lines below
            # data = data.to(torch.float32)
            # seg = seg.to(torch.float32)

            #print('data type:', data.dtype)
            #print('seg type:', seg.dtype)
            # added 2 lines below
            #seg.type(torch.cuda.FloatTensor)
            
            #####################
            # data = data.permute(0, 2, 1) # torch.Size([32, 4096, 3]) to # torch.Size([32, 3, 4096])

            batch_size = data.size()[0]
            #print('batch size: ', batch_size)
            optimizer.zero_grad()

            seg_pred = classifier(data)
            #print('seg pred type:', seg_pred.dtype)

            ###############
            # seg_pred = seg_pred.permute(0, 2, 1).contiguous() #  # torch.Size([32, 3, 4096]) to torch.Size([32, 4096, 3]) 

            # print('seg_pred size:', seg_pred.size())
            # change loss to l2 loss
            # print('before loss shape: ', seg.size(), seg_pred.size())

            # ###### delta addition
            # # start
            # data = data.permute(0, 2, 1)
            # gnd_delta = (data - seg)
            # pred_delta = seg_pred
            # loss = criterion(gnd_delta, pred_delta)
            # # end
            # print('torch size:', seg.size(), seg_pred.size())
            #olde loss before delta addition
            loss = criterion(seg_pred, seg)
            # print('loss in training: ', loss.item())

            #print('loss:', loss)
            #loss = criterion(seg_pred.view(-1, 13), seg.view(-1,1).squeeze())
            #loss_meter.update(loss.item())

            loss.backward()
            optimizer.step()

            # print('predicted center shape: ', seg_pred.size())

            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)

            count += 1
            train_loss += loss.item()

            # count += batch_size
            # train_loss += loss.item()
            # train_loss += loss.item() * batch_size
            # print('train loss: ', train_loss)

            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)

        plt_train_loss.append(train_loss*1.0/count)
        #print("D_Loss :{:.4f} ".format(loss_meter.avg)
        outstr = 'Train %d, loss: %.6f' % (epoch, train_loss*1.0/count)
        print(outstr) # save in a file later

        # added testing
        
        test_loss = 0.0
        count = 0
        for data, seg in testDataLoader:
            data, seg = data.float().cuda(), seg.float().cuda()
            #added 2 lines below
            # data = data.to(torch.float32)
            # seg = seg.to(torch.float32)

            # data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred = classifier(data)
            # seg_pred = seg_pred.permute(0, 2, 1).contiguous()

            
            # ###### delta addition
            # # start
            # data = data.permute(0, 2, 1)
            # gnd_delta = (data - seg)
            # pred_delta = seg_pred
            # loss = criterion(gnd_delta, pred_delta)
            # # end

         
                      
            loss = criterion(seg_pred, seg)
            # print('loss in testing: ', loss.item())

            pred = seg_pred.max(dim=2)[1]

            count += 1
            test_loss += loss.item()
            #count += batch_size

            # test_loss += loss.item()
            # test_loss += loss.item() * batch_size

            # seg_np = seg.cpu().numpy()
            # pred_np = pred.detach().cpu().numpy()
            
            if epoch == args.epoch - 1:
                np_surface = data.cpu().detach().numpy() # (1, 4094,3)
                np_center = seg_pred.cpu().detach().numpy() # (1, 4094,3)

                # print('tensor to numpy shape: ', np.shape(np_surface), np.shape(np_center))
                np_surface = np.squeeze(np_surface) # (4094,3)
                np_center = np.squeeze(np_center) # (4094,3)
                #print('squeeze numpy shape: ', np.shape(np_surface), np.shape(np_center))


                ##################
                # for radius
                np_all_points = np.stack((np_surface,np_center), axis = 1)
                
                # print('test ', count, ' all points shape: ', np.shape(np_all_points))
                
                # save predicted points
                #save numpy points
                numpy_filename = 'output_' + format(int(count-1), '05d') + '.npy'
                # numpy_filename = 'output_' + format(int(count), '05d') + '_radius_'+ str(np.mean(np_center))+'.npy'
                
                np.save(os.path.join(test_save_path, numpy_filename), np_all_points) # sget rotated values
                print('saving pipe number as np array', int(count-1))
            
            
            
            
            

        plt_test_loss.append(test_loss*1.0/count)
        outstr = 'Test %d, loss: %.6f' % (epoch, test_loss*1.0/count)
        print(outstr) # save in a file later

        logger.info('Save model...')
        
        savepath = '/home/tasnim/from_004/Point-Transformers/outputs/model_'+ str(epoch) + '.pth'
        logger.info('Saving at %s' % savepath)
        state = {
            'epoch': epoch,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)
        logger.info('Saving model....')

        # if epoch == args.epoch - 1:
        #     torch.save(classifier.state_dict(), '/home/tasnim/from_004/Point-Transformers/outputs/model_%s.t7' % (str(epoch)))



        
            



    # plt the loss in this indent
    plt.plot(np.array(plt_train_loss), color='r', label='training loss')
    plt.plot(np.array(plt_test_loss), color='g', label='testing loss')
    plt.legend(loc='best')
    plt.savefig('/home/tasnim/from_004/Point-Transformers/loss_plt/delta_mean_std_min_max_loss_plot.png')









    '''
    for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
        points = points.data.numpy()
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)

        points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
        optimizer.zero_grad()

        seg_pred = classifier(torch.cat([points, to_categorical(label, num_category).repeat(1, points.shape[1], 1)], -1))
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]

        correct = pred_choice.eq(target.data).cpu().sum()
        mean_correct.append(correct.item() / (args.batch_size * args.num_point))
        loss = criterion(seg_pred, target)
        loss.backward()
        optimizer.step()

    train_instance_acc = np.mean(mean_correct)
    logger.info('Train accuracy is: %.5f' % train_instance_acc)

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()

        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            seg_pred = classifier(torch.cat([points, to_categorical(label, num_category).repeat(1, points.shape[1], 1)], -1))
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=float))
        for cat in sorted(shape_ious.keys()):
            logger.info('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

    logger.info('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
        epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
    if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
        logger.info('Save model...')
        savepath = 'best_model.pth'
        logger.info('Saving at %s' % savepath)
        state = {
            'epoch': epoch,
            'train_acc': train_instance_acc,
            'test_acc': test_metrics['accuracy'],
            'class_avg_iou': test_metrics['class_avg_iou'],
            'inctance_avg_iou': test_metrics['inctance_avg_iou'],
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)
        logger.info('Saving model....')

    if test_metrics['accuracy'] > best_acc:
        best_acc = test_metrics['accuracy']
    if test_metrics['class_avg_iou'] > best_class_avg_iou:
        best_class_avg_iou = test_metrics['class_avg_iou']
    if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
        best_inctance_avg_iou = test_metrics['inctance_avg_iou']
    logger.info('Best accuracy is: %.5f' % best_acc)
    logger.info('Best class avg mIOU is: %.5f' % best_class_avg_iou)
    logger.info('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)

    '''
'''
@hydra.main(config_path='config', config_name='partseg')
def pt_test(args, io):
    omegaconf.OmegaConf.set_struct(args, False)

   
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)


    test_save_path = '/home/tasnim/from_004/Point-Transformers/predictions'

    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    
  
    TEST_DATASET = pipe_dataset(num_points=args.num_point, partition='test')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.test_batch_size, shuffle=False, num_workers=10)

                
    #Try to load models
    # semseg_colors = test_loader.dataset.semseg_colors
    if args.model == 'dgcnn':
        model = DGCNN_semseg_s3dis(args).to(device)
    else:
        raise Exception("Not implemented")
    

    model = nn.DataParallel(model)
    # model.load_state_dict(torch.load(os.path.join(args.model_root, 'model.t7' )))

    ##################
    # SEE
    # test area is the number of epoch
    ##################
    model.load_state_dict(torch.load('outputs/%s/models/model_%s.t7' % (args.exp_name,str(args.test_area))))

    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
    for data, seg in test_loader:
        data, seg = data.to(device), seg.to(device)

        #added 2 lines below
        data = data.to(torch.float32)
        seg = seg.to(torch.float32)

        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        seg_pred = model(data)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()

        # tensor to numpy
        data = data.permute(0, 2, 1)
        np_surface = data.cpu().detach().numpy() # (1, 4094,3)
        np_center = seg_pred.cpu().detach().numpy() # (1, 4094,3)

        # print('tensor to numpy shape: ', np.shape(np_surface), np.shape(np_center))
        np_surface = np.squeeze(np_surface) # (4094,3)
        np_center = np.squeeze(np_center) # (4094,3)
        #print('squeeze numpy shape: ', np.shape(np_surface), np.shape(np_center))


        ##################
        # for radius
        np_all_points = np_surface
        # print('test ', count, ' all points shape: ', np.shape(np_all_points))
        
        # save predicted points
        #save numpy points
        numpy_filename = 'output_' + format(int(count), '05d') + '_radius_'+ str(np.mean(np_center))+'.npy'
        # numpy_filename = 'dgcnn_numpy_pipe_and_axis_' + format(int(count), '05d') + '_radius_'+ str(np.mean(np_center))+'.npy'
        np.save(os.path.join(test_save_path, numpy_filename), np_all_points) # sget rotated values
        print('saving pipe number as np array', int(count), ' radius: ', np.mean(np_center))
        count += 1
        ################


    outstr = 'Test :: test area: %s' % (str(args.test_area))
    print(outstr)

''' 

if __name__ == '__main__':
    pt_train()
    # main()