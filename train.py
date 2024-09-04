import argparse
import os
import random


import torch.optim

import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn

from mmcls.build import crossswin_base

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from data.Datasets import DataPrepare, fetch_dataloaders
from mmcls.swin.swin import swin_b_w7_224, swin_b_w12_384
from utils.acc_function import *
from utils.save_photo import save_crm
from utils.con_matrix import compute_matrix
from tqdm import tqdm
np.set_printoptions(threshold=np.inf,linewidth=2000)

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def main(args):

    pth_save_path = args.pthpath

    if not os.path.exists(pth_save_path):
        os.makedirs(pth_save_path)
    print("weights save：",pth_save_path)
    acc_n = []
    # 循环训练不同数据集
    for _ in range(args.max_epoch):
        acc = []
        for data_use, ratio in data_use_ratio:
            
            if data_use == 'aid':
                data_path = "F:/~/AID_dataset/AID"         
            elif data_use == 'nwpu': 
                data_path = "F:/~/NWPU-RESISC45/NWPU-RESISC45"            
            elif data_use == 'ucm':
                data_path = "F:/~/UCMerced_LandUse/UCM"
            elif data_use == 'siriwhu':
                data_path = "F:/~/SIRI-WHU/12class_tif"              
            else:
                print('Please choose datasets for training use!')
         
            print(data_use, ratio)

            train_transforms = T.Compose([
                T.Resize((256, 256)),
                # T.Resize((288, 288)),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.ColorJitter(0.4),
                # T.RandomRotation(90),
                T.ToTensor(),
                T.Normalize(dataset_mean, dataset_std),
                ])
   
            test_transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(dataset_mean, dataset_std)
                ])
            print("Loading dataset...")

            train_inputs, train_labels, test_inputs, test_labels, num_classes = fetch_dataloaders(data_path, data_use, float(ratio))
            train_loader = DataLoaderX(
                    DataPrepare(filenames=train_inputs, labels=train_labels, num_classes=num_classes,
                                transforms=train_transforms),
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=False)
            val_loader = DataLoaderX(
                    DataPrepare(filenames=test_inputs, labels=test_labels, num_classes=num_classes,
                                transforms=test_transforms),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True, drop_last=False)
            # loading model
            model = crossswin_base(num_classes=num_classes) 
            #model = swin_b_w7_224(pretrained=True) 
            #model = swin_b_w12_384(pretrained=True)
            if torch.cuda.is_available():
                model.cuda()
            else:
                print("cpu ----")
            criterion = nn.CrossEntropyLoss().cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
            scheduler = lr_scheduler.MultiStepLR(optimizer, [int(args.epochs * 0.33), int(args.epochs * 0.67)], gamma=0.1)
            num_epochs = args.epochs

            best_prec = 0
            print("Start training...")
            print("numbers of epochs : " + str(num_epochs))
            start_epoch = time.time()

            for epoch in range(0, num_epochs):
                # train for one epoch
                start = time.time()
                train(train_loader, model, criterion, optimizer, num_classes, epoch)
                prec1, test_loss = validate(val_loader, model, criterion,num_classes,epoch, args)
                print('Test epoch: {} Test set: Average loss: {:.6f}, Accuracy: {:.6f}'.format(epoch, test_loss, prec1))
                # remember best prec@1 and save checkpoint
                if prec1 > best_prec:
                    best_prec = prec1
                    torch.save(model, pth_save_path + str(data_use) + '_' + '_' + args.name + '.pth')
                end = time.time()
                print("max acc: {:.6f}".format(best_prec))
                print("time for one epoch:%.2fs \n" % ((end - start)))
                # for scheduler in schedulers:
                scheduler.step()
                if epoch%10==9:
                    torch.save(model, pth_save_path +str(epoch)+"_"+str(data_use) + '_' + '_' + args.name + '.pth')
            acc.append(best_prec)
            end_epoch = time.time()
            print(best_prec)
            print("Training Final Total Time :%.2fs \n" % ((end_epoch - start_epoch)))
        acc_n.append(acc)
    for acc in acc_n:
        print(acc)
    [mean, std] = compute_average(acc_n)

def train(train_loader, model, criterion, optimizer, num_classes, epoch):
    """
        Run train epoch
    """
    # scaler = torch.cuda.amp.GradScaler()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    error = AverageMeter()
    model.train()
    end = time.time()
    for i in range(1):
        for _, (input, target, _, img_path) in enumerate(tqdm(train_loader)):
            # measure data loading time
            data_time.update(time.time() - end)
            if torch.cuda.is_available():
                target = target.cuda()
                input = torch.autograd.Variable(input).cuda()
            else:
                input = torch.autograd.Variable(input)
            with torch.no_grad():
                target_var = torch.autograd.Variable(target)
            # with torch.cuda.amp.autocast():
            output = model(input)
            loss = criterion(output, target_var)
                # loss_base = torchvision.ops.sigmoid_focal_loss(out, target_onehot.float(), reduction='mean')
                # loss = loss_base + loss_crsa
            ######################################################

            ######################################################
            prec1 = accuracy(output.data, target)[0]
            batch_size = target_var.size(0)
            _, pred = output.float().data.cpu().topk(1, dim=1)
            error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.float()
            # measure accuracy and record loss
            top1.update(prec1.item(), input.size(0))
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # step = i+len(train_loader)*epoch+1
            # if step % 20 == 0:
            #     print('step {}: Average loss: {:.6f}, Accuracy: {:.6f}'.format(step, loss.data.cpu().numpy(),top1.avg))

    print(('Train epoch: {} Train set: Average loss: {:.6f}, Accuracy: {:.6f}'.format(epoch, losses.avg,top1.avg)))
     
def validate(val_loader, model, criterion, num_classes,epoch, args):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    test_pred = []
    test_target = []
    end = time.time()
    with torch.no_grad():
        for i, (input, target, _, img_path) in enumerate(tqdm(val_loader)):
            if torch.cuda.is_available():
                target = target.cuda()
                input = input.cuda()
            target_var = torch.autograd.Variable(target)
            input = torch.autograd.Variable(input)

            # with torch.cuda.amp.autocast():
                # compute output
            output = model(input)
            loss = criterion(output, target_var)
            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            _, cindex = torch.max(output.squeeze(), dim=-1)
            test_target.extend(target.tolist())
            test_pred.extend(cindex.tolist())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        if args.save_matrix is not None and epoch == 99:
            compute_matrix(args.save_matrix, test_target, test_pred, top1, num_classes,epoch)
    return top1.avg, losses.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=3e-4, type=float, help="learning rate")
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--batch_size', '--bs', default=8, type=int, help="batch_size")
    parser.add_argument('--name', default='test')
    parser.add_argument('--num_workers', default=0, type=int, help="workers number")
    # parser.add_argument('--step', default=20, type=int, help="print this time loss")
    parser.add_argument('--GPU_ID', '--gpu', default='0', type=str, help="use GPU ID")
    parser.add_argument('--max_epoch', '--me', default=1, type=int, help="mean acc")
    parser.add_argument('--save_matrix', '--sm', default=True, type=str, help="valid_size")
    parser.add_argument('--save_img', '--si', default=None, type=str, help="valid_size")
    parser.add_argument('--data_use', '--ds', default="ucm", type=str, help="valid_size")
    parser.add_argument('--data_ratio', '--d2', default="0.5", type=str, help="valid_size")
    parser.add_argument('--pthpath', '--pp', default='./pth/', type=str, help="valid_size")
    # parser.add_argument('--seed', default=5, type=int)
    args = parser.parse_args()
    print('--loading parameters--')
    print(args)

    dataset_mean = [0.485, 0.456, 0.406]
    dataset_std = [0.229, 0.224, 0.225]

    data_use_ratio = [[args.data_use,args.data_ratio]]
    # data_use_ratio = [['ucm', '0.5', 0.05], ['ucm', '0.8', 0.05], ['aid', '0.2', 0.05], ['aid', '0.5', 0.05],  ['nwpu', '0.2', 0.05],  ['nwpu', '0.1', 0.05]]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID

    main(args)

