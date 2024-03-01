from pathlib import Path

from coco import build
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from unet import UNet, SumUNet
import argparse

from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import numpy as np
import random
from PIL import Image



seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def sigmoid_focal_loss(pred, targets, alpha: float = 0.25, gamma: float = 2):
    prob = pred.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(prob, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean() #.sum()


def calc_loss(pred, target, metrics, sf_weight=0.67):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    # sfocal = sigmoid_focal_loss(pred, target)

    loss = bce * sf_weight + dice * (1 - sf_weight)

    # metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(args, model, dataloaders, optimizer, scheduler):
    best_loss = 1e10
    print_freq = 10

    for epoch in range(args.epochs):
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Epoch: [{}]'.format(epoch)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            ious, dices = [], []
            for samples, targets in metric_logger.log_every(dataloaders[phase], print_freq, header):
                # for inputs, labels in :
                inp = samples.tensors
                seg_out = torch.stack([t['masks'] for t in targets], dim=0)
                inputs = inp.to(device)
                labels = seg_out.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # print(inputs.shape)
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels*1.0, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        pred = outputs.sigmoid() > 0.5
                        ious.append(compute_iou(pred, labels))
                        dices.append(compute_dice_coefficient(pred, labels))

                    # statistics
                epoch_samples += inputs.size(0)

            if phase == 'val':
                metrics['mIoU'] = (sum(ious)/len(ious)) * epoch_samples
                metrics['mDice'] = (sum(dices)/len(dices)) * epoch_samples
            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

            # save the model weights
            if phase == 'val' and epoch_loss < best_loss:
                print(f"saving best model to {args.checkpoint}")
                best_loss = epoch_loss
                torch.save(model.state_dict(), args.checkpoint)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(torch.load(args.checkpoint))
    return model


def main(args):
    coco_path = args.coco_path
    batch_size = args.batch_size
    num_class = 1
    if args.skip == 'concat':
        model = UNet(num_class, args.load_pretrained).to(device)
    elif args.skip == 'sum':
        model = SumUNet(num_class, args.load_pretrained).to(device)
    else:
        exit(1)

    dataset_train = build('train', coco_path)
    dataset_val = build('val', coco_path)

    image_datasets = {
        'train': dataset_train, 'val': dataset_val
    }

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=0)
    data_loader_val = DataLoader(dataset_val, batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=0)

    dataloaders = {
        'train': data_loader_train,
        'val': data_loader_val
    }
    # freeze backbone layers
    for l in model.base_layers:
        for param in l.parameters():
            param.requires_grad = False

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5*1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    train_model(args, model, dataloaders, optimizer_ft, exp_lr_scheduler)


def evaluate(args):
    num_class = 1
    if args.skip == 'concat':
        model = UNet(num_class, args.load_pretrained).to(device)
    elif args.skip == 'sum':
        model = SumUNet(num_class, args.load_pretrained).to(device)
    else:
        exit(1)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()  # Set model to the evaluation mode

    # Create a new simulation dataset for testing
    dataset_val = build('val', args.coco_path)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=0)
    # Predict
    # pred = model(inputs)
    mIoU, mDice = evaluate_segmentation(model, data_loader_val, save_images=True)
    print(f'Performance on validation dataset: mIoU:{mIoU} , mDice:{mDice}')


def compute_iou(pred, target):
    keep = target.sum(-1).sum(-1).flatten() > 0
    pred = pred[keep]
    target = target[keep]
    intersection = (pred & target).sum(-1).sum(-1)
    union = (pred | target).sum(-1).sum(-1)
    iou = torch.div(intersection, union).mean()
    return iou


def compute_dice_coefficient(pred, target):
    keep = target.sum(-1).sum(-1).flatten() > 0
    pred = pred[keep]
    target = target[keep]
    intersection = (pred & target).sum(-1).sum(-1)
    dice = 2 * intersection / (pred.sum(-1).sum(-1) + target.sum(-1).sum(-1))
    dice = dice.mean()
    return dice


def evaluate_segmentation(model, data_loader, save_images=False):
    ious = []
    dices = []
    for samples, targets in data_loader:
        inp = samples.tensors
        seg_out = torch.stack([t['masks'] for t in targets], dim=0)
        inputs = inp.to(device)
        labels = seg_out.to(device)
        pred = model(inputs)
        pred = pred.sigmoid() > 0.5
        ious.append(compute_iou(pred, labels))
        dices.append(compute_dice_coefficient(pred, labels))
        # img = inputs[0].permute(1,2,0).cpu().numpy()*1.0 +  np.transpose(pred[0].cpu().numpy()*(np.array([20, 0, 0])[:, None, None]), (1,2,0))
        if not save_images: continue
        for i in range(len(targets)):
            im = highlight_mask(inputs[i].permute(1, 2, 0).cpu().numpy(), pred[i, 0].cpu().numpy())
            Image.fromarray(im).save(f'images/{targets[i]["image_id"].item()}.png')
    mean_iou = torch.mean(torch.stack(ious))
    mean_dice = torch.mean(torch.stack(dices))
    return mean_iou, mean_dice


def highlight_mask(image, mask, color=(20, 0, 0)):
    # mask = mask*20
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # image[mask == 1] += np.array()
    return np.array(np.clip(256*(image.astype(np.float32)*std + mean) + np.transpose(mask[None,:,:]*np.asarray([0.0,120.0,0])[:,None,None], (1,2,0)), 0, 255), dtype=np.uint8)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    parser = argparse.ArgumentParser(description='Train a segmentation model on COCO data.')

    parser.add_argument('--coco_path', type=str, default='/home/kalyan/data/sandbox/minicoco/coco',
                        help='Path to COCO dataset directory.')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Epochs for training.')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Whether to evaluate the model on the validation set.')
    parser.add_argument('--load_pretrained', action='store_true', default=False,
                        help='Whether to evaluate the model on the validation set.')
    # parser.add_argument('--save_images', action='store_true', default=False,
    #                     help='Whether to save predicted masks of validation set.')
    parser.add_argument('--skip', type=str, choices=['concat', 'sum'], default='concat',
                        help='Skip connection strategy: either concatenation (concat) or summation (sum).')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth',
                        help='Path to checkpoint file.')

    args = parser.parse_args()
    args.checkpoint = f'{args.skip}_{args.checkpoint}'
    print(args)
    if not args.eval:
        main(args)
    else:
        Path('images').mkdir(parents=True, exist_ok=True)
        evaluate(args)
