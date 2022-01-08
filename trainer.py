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
from utils import test_single_volume
from metrics import StreamSegMetrics # TODO add new
from metrics.topo_loss import getTopoLoss # TODO add new
from datetime import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import test_single_volume
from torchsummary import summary


def sum_dict(a,b):
    temp = dict()
    for key in a.keys() | b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp

def validate(args,db_test,cur_epochs, model, loader, device, metrics,snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    torch.cuda.empty_cache()
    prec_time = datetime.now()
    metrics.reset()
    metric_list, score_list, score_iou, score_dice = 0.0, {}, {}, {}

    with torch.no_grad():
        for i, (image_data) in tqdm(enumerate(loader)):
            images, labels = image_data['image'], image_data['label']
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            case_name = image_data['case_name'][0]
            metric_i, score_new = test_single_volume(images, labels, model, classes=args.num_classes,
                                                     patch_size=[args.img_size, args.img_size],
                                                     test_save_path='../predictions', case=case_name,
                                                     z_spacing=1, dataset_name=args.dataset)
            metric_list += np.array(metric_i)
            score_iou = sum_dict(score_iou, score_new['Class IoU']);
            score_new.pop('Class IoU')  # TODO sum iou
            score_dice = sum_dict(score_dice, score_new['Class Dice']);
            score_new.pop('Class Dice')  # TODO sum dice
            score_list = sum_dict(score_list, score_new)  # TODO sum all except iou and dice
            #print ('idx %d case %s mean_dice %f mean_hd95 %f mean_IoU %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], score_new['Mean IoU']))
        metric_list = metric_list / len(db_test)
        for i in range(1, 2):
            print ('Mean class %d mean_dice %f mean_hd95 %f mean_IoU %f mean_dice %f' % (
            i, metric_list[i - 1][0], metric_list[i - 1][1], score_iou[i] / len(db_test),
            score_dice[str(i)] / len(db_test)))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        print ('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
        return {'Mean IoU': score_iou[i]}
def validate_synapse(args,db_test,cur_epochs, model, loader, device, metrics,snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    torch.cuda.empty_cache()
    prec_time = datetime.now()
    metrics.reset()
    metric_list, score_list, score_iou, score_dice = 0.0, {}, {}, {}

    with torch.no_grad():
        for i, (image_data) in tqdm(enumerate(loader)):
            images, labels = image_data['image'], image_data['label']
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            case_name = image_data['case_name'][0]
            metric_i, score_new = test_single_volume(images, labels, model, classes=args.num_classes,
                                                     patch_size=[args.img_size, args.img_size],
                                                     test_save_path='./output_synapse/predictions', case=case_name,
                                                     z_spacing=1, dataset_name=args.dataset)
            metric_list += np.array(metric_i)

        metric_list = metric_list / len(db_test)

        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        print ('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
        return {'mean_dice': performance}

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_test = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn) ## TODO new
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    best_val_miou=0 # TODO add new
    metrics = StreamSegMetrics(num_classes) # TODO add new
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        if epoch_num % 1 == 0:
            print("************* Validation *************")
            model.eval()
            val_score = validate_synapse(args,db_test,cur_epochs=epoch_num, model=model, loader=val_loader, device=device,
                                        metrics=metrics,snapshot_path=snapshot_path)
            if val_score["mean_dice"] > best_val_miou:
                best_val_miou = val_score["mean_dice"]
                save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info('Conf | Model saved as {}'.format(save_mode_path))
            model.train() # TODO add new
        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
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
def trainer_glas(args, model, snapshot_path):
    from datasets.dataset_GlaS import GlaS_dataset, RandomGenerator

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = GlaS_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    
    db_test =GlaS_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                          transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn) ## TODO new
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    best_val_miou=0 # TODO add new
    metrics = StreamSegMetrics(num_classes) # TODO add new
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            if epoch_num > int(max_epoch * 1.1):
                loss_topo = getTopoLoss(outputs, label_batch)  # TODO new
            loss = 0.4 * loss_ce + 0.6 * loss_dice if epoch_num <= int(
                max_epoch * 1.1) else 0.4 * loss_ce + 0.6 * loss_dice + 0.0005 * loss_topo  # TODO optimazation 1 int(max_epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            if epoch_num > int(max_epoch * 1.1):
                writer.add_scalar('info/loss_topo', loss_topo, iter_num)  # TODO add new
                logging.info('iteration %d : loss : %f, loss_ce: %f, loss_topo: %f' % (
                    iter_num, loss.item(), loss_ce.item(), loss_topo.item()))  # TODO add new
            else:
                logging.info('iteration %d : loss : %f, loss_ce: %f' % (
                    iter_num, loss.item(), loss_ce.item()))

            if iter_num % 5 == 0: # TODO show result
                image = image_batch[0, :, :, :] #[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                # writer.add_image('train/Prediction', outputs[0, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
                writer.add_image('train/Prediction', outputs[0, ...] * 50, iter_num)
                writer.add_image('train/Prediction', outputs[0, ...] * 50, iter_num)

        if epoch_num % 1 == 0:
            print("************* Validation *************")
            model.eval()
            val_score = validate(args,db_test,cur_epochs=epoch_num, model=model, loader=val_loader, device=device,
                                        metrics=metrics,snapshot_path=snapshot_path)

            if val_score["Mean IoU"] > best_val_miou:
                best_val_miou = val_score["Mean IoU"]
                save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info('Conf | Model saved as {}'.format(save_mode_path))
            model.train() # TODO add new

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
    # summary(model, input_size=(3, 224, 224), batch_size=-1)
    writer.close()
    return "Training Finished!"

