import os
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
import cv2
from seg_metric import SegmentationMetric
import random
import shutil
import setproctitle
import time
import logging
from models.cmsnet import create_model

from dataset import RemoteData
from custom_transforms import Mixup, edge_contour
from loss import CrossEntropyLoss, Edge_loss, Edge_weak_loss

# ================= Multiclass Dice loss =================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, output, target):
        # output: [B, C, H, W]
        # target: [B, H, W]
        num_classes = output.shape[1]
        
        # Convert the target to one-hot format: [B, C, H, W].
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax over the class dimension.
        output_softmax = F.softmax(output, dim=1)
        
        dice_loss = 0.0
        # Compute Dice for each class and average the losses.
        for i in range(num_classes):
            o_c = output_softmax[:, i, ...]
            t_c = target_one_hot[:, i, ...]
            
            intersection = torch.sum(o_c * t_c, dim=(1, 2))
            union = torch.sum(o_c, dim=(1, 2)) + torch.sum(t_c, dim=(1, 2))
            
            dice_c = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1.0 - dice_c)
            
        return (dice_loss / num_classes).mean()

# ================= Training wrapper with the compound loss =================
class FullModel(nn.Module):
    def __init__(self, model, args2):
        super(FullModel, self).__init__()
        self.model = model
        self.use_mixup = args2.use_mixup
        self.use_edge = args2.use_edge

        self.ce_loss = CrossEntropyLoss()
        self.edge_loss = Edge_loss()
        self.dice_loss = DiceLoss()

        if self.use_mixup:
            self.mixup = Mixup(use_edge=args2.use_edge)

    def forward(self, input, label=None, train=True):
        if train and self.use_mixup and label is not None:
            if self.use_edge:
                loss = self.mixup(input, label, [self.ce_loss, self.edge_loss], self.model)
            else:
                loss = self.mixup(input, label, self.ce_loss, self.model)
            return loss

        output = self.model(input)
        if train:
            losses = 0
            if isinstance(output, (list, tuple)):
                # CMSNet returns [main_output, cnn_aux_output, vmamba_aux_output].
                if len(output) == 3: 
                    main_output, cnn_aux_output, vmamba_aux_output = output
                    main_loss = (
                        self.ce_loss(main_output, label)
                        + self.dice_loss(main_output, label)
                    )
                    auxiliary_loss = 0.4 * (
                        self.ce_loss(cnn_aux_output, label)
                        + self.ce_loss(vmamba_aux_output, label)
                    )
                    losses = main_loss + auxiliary_loss
                        
                elif self.use_edge:
                    for i in range(len(output) - 1):
                        losses += self.ce_loss(output[i], label)
                    losses += self.edge_loss(output[-1], edge_contour(label).long())
                else:
                    for i in range(len(output)):
                        losses += self.ce_loss(output[i], label)
            else:
                losses = self.ce_loss(output, label) + self.dice_loss(output, label)
            return losses
        else:
            # During evaluation, return the main output only.
            return output[0] if isinstance(output, (list, tuple)) else output

MODEL_CHOICES = (
    'cmsnet', 'vmamba', 'rs3mamba', 'cctnet', 'deeplabv3', 'unet', 'danet',
    'unetmamba', 'unettran',
)


def get_model(args2, device, models='cmsnet'):
    """Build one of the models retained in the revised comparison protocol."""
    if models not in MODEL_CHOICES:
        raise ValueError(f"Unsupported model '{models}'. Choose from: {MODEL_CHOICES}")

    nclass = 4

    if models == 'cmsnet':
        print("Initializing CMSNet (Dual-Branch + Multi-Head)...")
        model = create_model(num_classes=nclass, input_size=args2.crop_size[0])
    elif models == 'vmamba':
        print("Initializing the VMamba-only baseline...")
        model = create_model(
            num_classes=nclass,
            input_size=args2.crop_size[0],
            mode='mamba_only',
        )
    elif models == 'danet':
        from models.danet import DANet
        model = DANet(nclass=nclass, backbone='resnet50', pretrained_base=True)
    elif models == 'deeplabv3':
        from models.deeplabv3 import DeepLabV3
        model = DeepLabV3(nclass=nclass, backbone='resnet50', pretrained_base=True)
    elif models == 'unet':
        from models.unet import UNet
        model = UNet(nclass=nclass)
    elif models == 'cctnet':
        from models.cctnet import CCTNet
        model = CCTNet(transformer_name=args2.trans_cnn[0], cnn_name=args2.trans_cnn[1], nclass=nclass,
                       img_size=args2.crop_size[0],
                       pretrained=True, aux=True, head=args2.head, edge_aux=args2.use_edge)
    elif models in {'rs3mamba', 'unetmamba', 'unettran'}:
        raise ImportError(
            f"The {models} implementation is not included in this repository. "
            "Add its model file under models/ before selecting this option."
        )

    model = FullModel(model, args2)
    model = model.to(device)
    return model

class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def parse_args():
    parser = argparse.ArgumentParser(description='Train CMSNet')
    parser.add_argument("--data_dir", type=str, default='./data', help="Path to the dataset root")
    parser.add_argument("--save_dir", type=str, default='./newwork_dir', help="Directory for checkpoints and logs")
    parser.add_argument("--dataset", type=str, default='barley', choices=['barley'])
    parser.add_argument("--end_epoch", type=int, default=100)
    parser.add_argument("--warm_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--train_batchsize", type=int, default=8)
    parser.add_argument("--val_batchsize", type=int, default=4)
    parser.add_argument("--crop_size", type=int, nargs='+', default=[512, 512], help='H, W')
    parser.add_argument("--information", type=str, default='RS')
    parser.add_argument("--models", type=str, default='cmsnet', choices=MODEL_CHOICES)
    parser.add_argument("--head", type=str, default='seghead')
    parser.add_argument("--trans_cnn", type=str, nargs='+', default=['cswin_tiny', 'resnet50'], help='transformer, cnn')
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--use_edge", type=int, default=0)
    parser.add_argument("--use_mixup", type=int, default=0)
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    args2 = parser.parse_args()
    return args2

def save_model_file(save_dir, save_name):
    save_dir = os.path.join(save_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir + '/weights/')
        os.makedirs(save_dir + '/outputs/')
    for file in os.listdir('.'):
        if os.path.isfile(file):
            shutil.copy(file, save_dir)
    if not os.path.exists(os.path.join(save_dir, 'models')):
        try:
            shutil.copytree('./models', os.path.join(save_dir, 'models'))
        except FileNotFoundError:
            pass
    logging.basicConfig(filename=save_dir + '/train.log', level=logging.INFO)

def to_float(value):
    return value.item() if torch.is_tensor(value) else value

def train():
    args2 = parse_args()

    torch.manual_seed(args2.seed)
    torch.cuda.manual_seed_all(args2.seed)
    random.seed(args2.seed)
    np.random.seed(args2.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    save_name = "{}_lr{}_epoch{}_batchsize{}_{}".format(args2.models, args2.lr, args2.end_epoch,
                                                        args2.train_batchsize, args2.information)
    save_dir = args2.save_dir

    save_model_file(save_dir=save_dir, save_name=save_name)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = get_model(args2, device, models=args2.models)

    remotedata_train = RemoteData(base_dir=args2.data_dir, split='train', dataset=args2.dataset, crop_size=args2.crop_size)
    dataloader_train = DataLoader(
        remotedata_train,
        batch_size=args2.train_batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True)

    remotedata_val = RemoteData(base_dir=args2.data_dir, split='val', dataset=args2.dataset, crop_size=args2.crop_size)
    dataloader_val = DataLoader(
        remotedata_val,
        batch_size=args2.val_batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    # Use one learning-rate schedule for all trainable parameters.
    optimizer = torch.optim.AdamW([{'params':
                                        filter(lambda p: p.requires_grad,
                                               model.parameters()),
                                    'lr': args2.lr}],
                                  lr=args2.lr,
                                  betas=(0.9, 0.999),
                                  weight_decay=0.01,
                                  )

    start = time.time()
    miou = 0
    acc = 0
    f1 = 0
    precision = 0
    recall = 0
    best_miou = 0
    best_acc = 0
    best_f1 = 0
    last_epoch = 0
    test_epoch = args2.end_epoch - 10
    ave_loss = AverageMeter()

    weight_save_dir = os.path.join(save_dir, save_name + '/weights')
    model_state_file = weight_save_dir + "/{}_lr{}_epoch{}_batchsize{}_{}.pkl.tar" \
        .format(args2.models, args2.lr, args2.end_epoch, args2.train_batchsize, args2.information)

    if os.path.isfile(model_state_file):
        print('loaded successfully')
        logging.info("=> loading checkpoint '{}'".format(model_state_file))
        checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
        checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
        best_miou = checkpoint['best_miou']
        best_acc = checkpoint['best_acc']
        best_f1 = checkpoint['best_f1']
        last_epoch = checkpoint['epoch']
        
        if torch.is_tensor(best_miou): best_miou = best_miou.item()
        if torch.is_tensor(best_acc): best_acc = best_acc.item()
        if torch.is_tensor(best_f1): best_f1 = best_f1.item()
        
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(
            model_state_file, checkpoint['epoch']))

    for epoch in range(last_epoch, args2.end_epoch):
        model.train()
        setproctitle.setproctitle("xzy:" + str(epoch) + "/" + "{}".format(args2.end_epoch))

        for i, sample in enumerate(dataloader_train):
            image, label = sample['image'], sample['label']
            image, label = image.to(device), label.to(device)
            label = label.long().squeeze(1)
            losses = model(image, label)

            loss = losses.mean()
            ave_loss.update(loss.item())

            lenth_iter = len(dataloader_train)
            lr = adjust_learning_rate(optimizer,
                                      args2.lr,
                                      args2.end_epoch * lenth_iter,
                                      i + epoch * lenth_iter,
                                      args2.warm_epochs * lenth_iter
                                      )
            if i % 50 == 0:
                print_loss = ave_loss.average()
                time_cost = time.time() - start
                start = time.time()
                print("epoch:[{}/{}], iter:[{}/{}], loss:{:.4f}, time:{:.4f}, lr:{:.4f}, "
                      "best_miou:{:.4f}, miou:{:.4f}, acc:{:.4f}, f1:{:.4f}, precision:{:.4f}, recall:{:.4f}".
                      format(epoch, args2.end_epoch, i, len(dataloader_train), print_loss, time_cost, lr,
                             best_miou, to_float(miou), to_float(acc), to_float(f1), 
                             to_float(precision), to_float(recall)))
                logging.info(
                    "epoch:[{}/{}], iter:[{}/{}], loss:{:.4f}, time:{:.4f}, lr:{:.4f}, "
                    "best_miou:{:.4f}, miou:{:.4f}, acc:{:.4f}, f1:{:.4f}, precision:{:.4f}, recall:{:.4f}".
                    format(epoch, args2.end_epoch, i, len(dataloader_train), print_loss, time_cost, lr,
                           best_miou, to_float(miou), to_float(acc), to_float(f1), 
                           to_float(precision), to_float(recall)))

            model.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch > test_epoch:
            miou, acc, f1, precision, recall = validate(dataloader_val, device, model, args2)

        if epoch > test_epoch and epoch != 0:
            print('miou:{}, acc:{}, f1:{}, precision:{}, recall:{}'.format(
                to_float(miou), to_float(acc), to_float(f1), 
                to_float(precision), to_float(recall)))

        if to_float(miou) >= best_miou and to_float(miou) != 0:
            best_miou = to_float(miou)
            best_acc, best_f1 = to_float(acc), to_float(f1)
            best_weight_name = weight_save_dir + '/{}_lr{}_epoch{}_batchsize{}_{}_best_epoch_{}.pkl'.format(
                args2.models, args2.lr, args2.end_epoch, args2.train_batchsize, args2.information, epoch)
            torch.save(model.model.state_dict(), best_weight_name)
            torch.save(model.model.state_dict(), weight_save_dir + '/best_weight.pkl')

        torch.save({
            'epoch': epoch + 1,
            'best_miou': best_miou,
            'best_acc': best_acc,
            'best_f1': best_f1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, weight_save_dir + '/{}_lr{}_epoch{}_batchsize{}_{}.pkl.tar'
           .format(args2.models, args2.lr, args2.end_epoch, args2.train_batchsize, args2.information))

    logging.info("***************super param*****************")
    logging.info("dataset:{} information:{} lr:{} epoch:{} batchsize:{} best_miou:{} best_acc:{} best_f1:{}"
                 .format(args2.dataset, args2.information, args2.lr, args2.end_epoch, args2.train_batchsize,
                         best_miou, best_acc, best_f1))
    logging.info("***************end*************************")

def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, warmup_iter=None, power=0.9):
    if warmup_iter is not None and cur_iters < warmup_iter:
        lr = base_lr * cur_iters / (warmup_iter + 1e-8)
    elif warmup_iter is not None:
        lr = base_lr * ((1 - float(cur_iters - warmup_iter) / (max_iters - warmup_iter)) ** (power))
    else:
        lr = base_lr * ((1 - float(cur_iters / max_iters)) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    return lr

def validate(dataloader_val, device, model, args2):
    model.eval()
    MIOU, ACC, F1, Precision, Recall = 0.0, 0.0, 0.0, 0.0, 0.0
    nclass = 4

    metric = SegmentationMetric(nclass)
    with torch.no_grad():
        for i, sample in enumerate(dataloader_val):
            image, label = sample['image'], sample['label']
            alpha = sample.get('alpha')
            image, label = image.to(device), label.to(device)
            label = label.long().squeeze(1)
            
            logit = model(image, label, train=False)
            logit = logit.argmax(dim=1)
            logit = logit.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            
            # Use the alpha channel to exclude invalid pixels when available.
            if alpha is None:
                valid = np.ones_like(label, dtype=bool)
            else:
                valid = alpha.numpy() > 0
            metric.addBatch(logit[valid], label[valid])

    iou = metric.IntersectionOverUnion()
    acc = metric.Accuracy()
    precision = metric.Precision()
    recall = metric.Recall()

    print('miou:{}, precision:{}, recall:{}'.format(iou, precision, recall))
    logging.info('miou:{}, precision:{}, recall:{}'.format(iou, precision, recall))

    miou = np.nanmean(iou)
    mprecision = np.nanmean(precision)
    mrecall = np.nanmean(recall)

    MIOU = MIOU + miou
    ACC = ACC + acc
    Recall = Recall + mrecall
    Precision = Precision + mprecision
    F1 = F1 + 2 * Precision * Recall / (Precision + Recall)
    
    MIOU = torch.from_numpy(np.array(MIOU)).to(device)
    ACC = torch.from_numpy(np.array(ACC)).to(device)
    F1 = torch.from_numpy(np.array(F1)).to(device)
    Recall = torch.from_numpy(np.array(Recall)).to(device)
    Precision = torch.from_numpy(np.array(Precision)).to(device)

    return MIOU.item(), ACC.item(), F1.item(), Precision.item(), Recall.item()

if __name__ == '__main__':
    cudnn.enabled = True
    train()
