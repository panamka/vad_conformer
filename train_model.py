import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn import DataParallel as DP
from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np
import os
from shutil import rmtree
import math
from TensorboardLogger import TensorboardLogger

from dataset import VadDataset, DataLoader
from metrics.loss import CrossEntropy as Loss
from metrics.metrics_vad import Accuracy
from metrics.metrics_vad import MseLoss

from model.conformer.stft import StftHandler
from model.conformer.model_casual import ConformerVad

def to_numpy(x):
    return x.detach().cpu().numpy()

def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % (2 ** 32 -1))


def train_epoch(model, loader, optimizer, scheduler, criterion, metric_dict, max_norm_grad, device):
    log = {}
    losses = []
    for mixture, vad_mask in tqdm(loader):
        mixture = mixture.to(device, dtype=torch.float)
        vad_mask = vad_mask.to(device, dtype=torch.float)

        optimizer.zero_grad()
        for param_group in optimizer.param_groups:
            log['lr_train'] = param_group['lr']

        pred_mask = model(mixture)
        loss = criterion(pred_mask, vad_mask)
        loss = loss.mean()
        loss.backward()
        grad_norm = clip_grad_norm_(
            model.parametrs(), max_norm_grad).item(
        )
        log['grad_norm'] = grad_norm
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        for name, metrics in metric_dict.items():
            values = metrics(pred_mask, vad_mask)
            values = to_numpy(values)
            log[name] = np.mean(values)

    return {'train_loss' : np.mean(losses)} | log


@torch.inference_mode()
def val_epoch(model, loader, criterion, metric_dict, device):
    losses = []
    log = {}
    for mixture, vad_mask in tqdm(loader):
        mixture = mixture.to(device, dtype=torch.float)
        vad_mask = vad_mask.to(device, dtype=torch.float)

        pred_mask = model(mixture)
        loss = criterion(pred_mask, vad_mask)
        loss = loss.mean()
        losses.append(loss.item())

        for name, metrics in metric_dict.items():
            values = metrics(pred_mask, vad_mask)
            values = to_numpy(values)
            log[name] = np.mean(values)

    return {'val_loss' : np.mean(losses)} | log

def main():
    logdir = 'tensorboard_logging'
    model_name = 'vad_conformer'

    logger_tb = TensorboardLogger(os.path.join(logdir,model_name))

    train_history = defaultdict(list)
    val_history = defaultdict(list)

    max_grad_norm = 5
    cuda_visible_devices = '0,1,2,3'
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    devices = [1]
    device = 'cuda:1'
    n_epochs = 600
    batch_size = 192 * len(devices)
    lr =1e-4
    start_epoch = 0
    best_loss = float('inf')

    metric_dict = {
                'accuracy' : Accuracy(),
                'mse' : MseLoss(),
                }

    criterion = Loss()

    conf_kwargs = dict(
        dim=256,
        dim_head=64,
        heads=4,
        ff_mult=2,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.1,
        ff_dropout=0.1,
        conv_dropout=0.1,
        look_ahead=6,
    )

    model = ConformerVad(
        stft=StftHandler(),
        num_layers=12,
        inp_dim=257,
        out_dim=257,
        conformer_kwards=conf_kwargs,
    )

    model = DP(model, device_ids=devices)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    lr_scheduler = build_sch(optimizer)

    saveroot = 'path'
    if os.path.exists(saveroot):
        ans = input(f'{saveroot} already exists. Do you want to rewtire it? Y/n: ').lower()
        if ans == 'y':
            rmtree(saveroot)
            os.makedirs(saveroot)
            if os.path.exists(os.path.join(logdir, model_name)):
                print('del TB folder')
                rmtree(os.path.join(logdir, model_name))
                print('create new TB folder')
                logger_tb = TensorboardLogger(os.path.join(logdir, model_name))
        else:
            print('Continue to train')
            checkpoint_path = os.path.join(saveroot, 'last_snapshot.tar')
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['opt'])
            lr_scheduler.load_state_dict(checkpoint['sch'])
            train_history = checkpoint['train_history']
            val_history = checkpoint['val_history']
            best_loss = min(val_history['val_loss'])
            start_epoch = len(train_history['train_loss'])
            print(f'Train loss: {train_history["train_loss"][-1]}')
            print(f'Val loss: {val_history["val_loss"][-1]}')
            model.train()
    else:
        os.makedirs(saveroot)

    root_train = '/home/train-clean-360'
    root_test = '/home/train-clean-100'
    root_noise = '/home/noise_upgrade'

    train_dataset = VadDataset(root_train, root_noise)
    test_dataset = VadDataset(root_test, root_noise, train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
        num_workers=8, worker_init_fn=worker_init_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
        num_workers=8, worker_init_fn=worker_init_fn
    )
    for epoch in tqdm(range(start_epoch, n_epochs + start_epoch)):
        train_dict = train_epoch(model, train_loader, optimizer, lr_scheduler, criterion, metric_dict,
                                 max_grad_norm, device)
        val_dict = val_epoch(model, test_loader, criterion, metric_dict, device)
        print('train metrics')
        for key, value in train_dict.items():
            logger_tb.log(epoch, value, key, 'train')
            print(f'{key}: {value}')
            train_history[key].append(value)
        print('val metrics')
        for key, value in val_dict.items():
            logger_tb.log(epoch, value, key, 'val')
            print(f'{key}: {value}')
            val_history[key].append(value)
        snapshot = {
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            'sch': lr_scheduler.state_dict(),
            'train_history': train_history,
            'val_history': val_history,
        }

        last_snapshot_path = os.path.join(saveroot, 'last_snapshot.tar')
        torch.save(snapshot, last_snapshot_path)

        cur_loss = val_dict['val_loss']
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_snapshot_path = os.path.join(saveroot, 'best_snapshot.tar')
            torch.save(snapshot, best_snapshot_path)

start_lr = 1e-5
main_lr = 1e-5

# main_lr = 4e-3

def build_sch(optimizer):
    def schedule(step):
        warm_up_steps = 1000

        if step >= warm_up_steps:
            return 1

        start_mult = start_lr / main_lr
        fraction = step / warm_up_steps * math.pi
        stair = (1 - math.cos(fraction)) / 2
        stair = start_mult + (1 - start_mult) * stair

        return stair

    sch = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=schedule
    )
    return sch

if __name__ == '__main__':
        main()