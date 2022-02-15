import logging
import math
import operator
import time
import torch
import torch.nn.functional as F
import numpy as np
import random
from util import AverageMeter
from timm.utils import reduce_tensor

__all__ = ['train', 'validate', 'PerformanceScoreboard']

logger = logging.getLogger()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def update_meter(meter, loss, acc1, acc5, size, batch_time, world_size):
    reduced_loss = reduce_tensor(loss.data, world_size)
    reduced_top1 = reduce_tensor(acc1, world_size)
    reduced_top5 = reduce_tensor(acc5, world_size)
    meter['loss'].update(reduced_loss.item(), size)
    meter['top1'].update(reduced_top1.item(), size)
    meter['top5'].update(reduced_top5.item(), size)
    meter['batch_time'].update(batch_time)


def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, monitors, args):
    meters = {
        'loss': AverageMeter(),
        'top1': AverageMeter(),
        'top5': AverageMeter(),
        'batch_time': AverageMeter()
    }

    total_sample = len(train_loader.sampler)
    batch_size = args.dataloader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    if args.local_rank == 0:
        logger.info('Training: %d samples (%d per mini-batch)',
                    total_sample, batch_size)
    num_updates = epoch * len(train_loader)
    seed = num_updates
    model.train()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        optimizer.zero_grad()
        seed = seed + 1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        start_time = time.time()
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        update_meter(meters, loss, acc1, acc5, inputs.size(0), time.time() - start_time, args.world_size)
        loss.backward()

        for group in optimizer.param_groups:
            for p in group['params']:
                if not p.grad is None and torch.sum(torch.abs(p.grad.data)) == 0.0:
                    p.grad = None
                    
        optimizer.step()
        num_updates += 1

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=meters['loss'].avg)
        
        if args.local_rank == 0 and (batch_idx + 1) % args.log.print_freq == 0:
            for m in monitors:
                m.update(epoch, batch_idx + 1, steps_per_epoch, 'Training', {
                    'Loss': meters['loss'],
                    'Top1': meters['top1'],
                    'Top5': meters['top5'],
                    'BatchTime': meters['batch_time'],
                    'LR': optimizer.param_groups[0]['lr']
                })
            logger.info(
                "--------------------------------------------------------------------------------------------------------------")
    if args.local_rank == 0:
        logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f', meters['top1'].avg, meters['top5'].avg, meters['loss'].avg)
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
    return meters['top1'].max, meters['top5'].max, meters['loss'].avg


def validate(data_loader, model, criterion, epoch, monitors, args):
    meters = {
        'loss': AverageMeter(),
        'top1': AverageMeter(),
        'top5': AverageMeter(),
        'batch_time': AverageMeter()
    }

    total_sample = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    if args.local_rank == 0:
        logger.info('Validation: %d samples (%d per mini-batch)',
                    total_sample, batch_size)

    model.eval()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            start_time = time.time()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            update_meter(meters, loss, acc1, acc5, inputs.size(0), time.time() - start_time, args.world_size)
            
            if args.local_rank == 0:
                if (batch_idx + 1) % args.log.print_freq == 0:
                    for m in monitors:
                        m.update(epoch, batch_idx + 1, steps_per_epoch, 'Val', {
                            'Loss': meters['loss'],
                            'Top1': meters['top1'],
                            'Top5': meters['top5'],
                            'BatchTime': meters['batch_time'],
                        })
                    logger.info(
                        "--------------------------------------------------------------------------------------------------------------")
    if args.local_rank == 0:
        logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f', meters['top1'].avg, meters['top5'].avg, meters['loss'].avg)
    return meters['top1'].max, meters['top5'].max, meters['loss'].avg



class PerformanceScoreboard:
    def __init__(self, num_best_scores):
        self.board = list()
        self.num_best_scores = num_best_scores

    def update(self, top1, top5, epoch):
        """ Update the list of top training scores achieved so far, and log the best scores so far"""
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch})

        # Keep scoreboard sorted from best to worst, and sort by top1, top5 and epoch
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board,
                            key=operator.itemgetter('top1', 'top5', 'epoch'),
                            reverse=True)[0:curr_len]
        for idx in range(curr_len):
            score = self.board[idx]
            logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                        idx + 1, score['epoch'], score['top1'], score['top5'])

    def is_best(self, epoch):
        return self.board[0]['epoch'] == epoch