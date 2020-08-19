import os
import os.path as osp
import numpy as np
import tqdm
import glob
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable

import datetime
import pytz
import sys
sys.path.append("../")
from semnet.models.resnet import SEMNet
from semnet.datasets.semdataset import SEMDataset

def _fast_hist(label_true, label_pred, n_class):
    hist = np.bincount(
        n_class * label_true.astype(int) +
        label_pred.astype(int), minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class=2):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     acc_cls = np.diag(hist) / hist.sum(axis=1)
    # acc_cls = np.nanmean(acc_cls)
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ## Define hyper parameters
    parser.add_argument('--epochs', type=int, default=500, help='epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    args = parser.parse_args()
    args.model = 'semnet'

    here = osp.dirname(osp.abspath(__file__))
    log_dir = osp.join(here, 'log')
    runs = sorted(glob.glob(os.path.join(log_dir, 'experiment_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    output_dir = os.path.join(log_dir, 'experiment_{}'.format(str(run_id)))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(osp.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    print('Argument Parser: ')
    print(args.__dict__)

    cuda = torch.cuda.is_available()
    print('Cuda available: ', cuda)
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    ## Define data loader
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(SEMDataset(split='train'), batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(SEMDataset(split='valid'), batch_size=args.batch_size,shuffle=False, **kwargs)
    ## Define neural network model
    model = SEMNet()
    if cuda:
        model = model.cuda()
    # Define optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Define loss function
    loss_fn = nn.BCELoss()

    if not osp.exists(osp.join(output_dir, 'log.csv')):
        with open(osp.join(output_dir, 'log.csv'), 'w') as f:
            header = ['epoch', 'iteration', 'MSE', 'elapsed_time']
            header = map(str, header)
            f.write(','.join(header) + '\n')

    if not osp.exists(osp.join(output_dir, 'val_log.csv')):
        with open(osp.join(output_dir, 'val_log.csv'), 'w') as f:
            header = ['epoch', 'MSE', 'Acc', 'elapsed_time']
            header = map(str, header)
            f.write(','.join(header) + '\n')
    timestamp_start = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    best_acc = 0.0
    for epoch in range(args.epochs):
        iteration = 0
        model.train()
        for batch_idx, sample in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), ncols=80, desc='Train epoch=%d' % epoch, leave=False):
            assert model.training
            img, lbl = sample['img'], sample['lbl']
            if cuda:
                img, lbl = img.cuda(), lbl.cuda()
            img, lbl  = Variable(img), Variable(lbl)
            optim.zero_grad()
            pred = model(img)
            loss = loss_fn(pred, lbl)
            train_loss = loss.data.item()
            loss.backward()
            optim.step()
            iteration = iteration + 1
            with open(osp.join(output_dir, 'log.csv'), 'a') as f:
                elapsed_time = (datetime.datetime.now(pytz.timezone('Asia/Tokyo')) - timestamp_start).total_seconds()
                log = [epoch, iteration, train_loss, elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')
            print('Train Epoch: ', epoch, 'Iteration: ', iteration, 'Loss: ', train_loss)

        model.eval()
        val_loss = 0.0
        label_trues, label_preds = [], []
        for batch_idx, sample in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), ncols=80, desc='Validation epoch=%d' % epoch, leave=False):
            img, lbl = sample['img'], sample['lbl']
            if cuda:
                img, lbl = img.cuda(), lbl.cuda()
            img, lbl  = Variable(img), Variable(lbl)
            with torch.no_grad():
                pred = model(img)
            loss = loss_fn(pred, lbl)
            val_loss =  val_loss + loss.data.item()
            lbl = lbl.data.cpu().squeeze(dim=1).numpy()
            pred = pred.data.cpu().squeeze(dim=1).numpy()
            label_trues = np.concatenate((label_trues, lbl), axis=0)
            label_preds = np.concatenate((label_preds, pred), axis=0)

        label_preds = np.where(label_preds>0.5, 1.0, 0.0)
        val_loss = val_loss / len(test_loader)
        acc = label_accuracy_score(label_trues, label_preds)
        with open(osp.join(output_dir, 'val_log.csv'), 'a') as f:
            elapsed_time = (datetime.datetime.now(pytz.timezone('Asia/Tokyo')) - timestamp_start).total_seconds()
            log = [epoch, val_loss, acc, elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')
        print('Eval Epoch: ', epoch, 'Loss: ', val_loss, 'Acc: ', acc)
        checkpoint = {
            'epoch': epoch,
            'model': model,
            'optim_state_dict': optim.state_dict(),
            'model_state_dict': model.state_dict()
        }
        torch.save(checkpoint, osp.join(output_dir,'checkpoint.pth.tar'))
        if acc > best_acc:
            best_acc = acc
            checkpoint = {
                'epoch': epoch,
                'best_acc': best_acc,
                'model': model,
                'optim_state_dict': optim.state_dict(),
                'model_state_dict': model.state_dict()
            }
            torch.save(osp.join(output_dir,'best_model.pth.tar'))




