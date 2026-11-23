import argparse
import json
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import build_model
from utils.dataset import get_voc
from utils.kd import response_kd_loss, feature_kd_loss
from utils.metrics import update_confusion, mean_iou
from utils.teacher import load_teacher, teacher_forward


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['student', 'custom'], default='student',
                        help='student = provided StudentNet, custom = our LiteSegNet (Step 3)')
    parser.add_argument('--kd', choices=['none', 'response', 'feature', 'both'], default='none')
    parser.add_argument('--data', default='./data')
    parser.add_argument('-e', '--epochs', type=int, default=3)
    parser.add_argument('-b', '--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--size', type=int, default=320)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=4.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lr-patience', type=int, default=2)
    parser.add_argument('--lr-factor', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--tol', type=float, default=0.1, help='min mIoU gain (points)')
    parser.add_argument('--aug', action='store_true',
                        help='random scale (0.5-2.0) + crop + horizontal flip on the train set')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--iters', type=int, default=0,
                        help='kick-start: stop after this many iterations (0 = full training)')
    return parser.parse_args()


def main():
    args = get_args()
    tag = args.kd if args.model == 'student' else f'custom_{args.kd}'

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    train_loader = DataLoader(get_voc(args.data, 'train', args.size, train=args.aug),
                              batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(get_voc(args.data, 'val', args.size), batch_size=args.batch,
                            shuffle=False, num_workers=args.workers,
                            persistent_workers=args.workers > 0)

    student = build_model(args.model).to(device)
    params = list(student.parameters())
    teacher = load_teacher(device) if args.kd != 'none' else None

    # 1x1 adapters map student features to the teacher's channel counts (training only)
    adapters = None
    if args.kd in ('feature', 'both'):
        adapters = nn.ModuleDict({k: nn.Conv2d(c, t, 1) for k, (c, t) in
                                  student.kd_channels.items()}).to(device)
        params += list(adapters.parameters())

    def feat_loss(out, f_t):
        return sum(feature_kd_loss(adapters[k](out[k]), f_t[layer])
                   for k, layer in student.kd_taps.items())

    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.lr_factor, patience=args.lr_patience)
    n_params = sum(p.numel() for p in student.parameters())
    print(f'Training {args.model} ({n_params:,} params) with kd={args.kd} on {device}' +
          (f' for {args.iters} iterations' if args.iters else f' for up to {args.epochs} epochs'))
    print(vars(args), flush=True)

    losses, history = [], []
    best, best_epoch, bad_epochs = -1.0, 0, 0
    os.makedirs('checkpoints', exist_ok=True)
    start = time.time()
    for epoch in range(1, (10**6 if args.iters else args.epochs) + 1):
        student.train()
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            out = student(img)
            if args.kd == 'none':
                loss = F.cross_entropy(out['out'], mask, ignore_index=255)
            else:
                with torch.no_grad():
                    z_t, f_t = teacher_forward(teacher, img)
                if args.kd == 'response':
                    loss = response_kd_loss(out['out'], z_t, mask, args.alpha, args.beta, args.tau)
                elif args.kd == 'feature':
                    loss = F.cross_entropy(out['out'], mask, ignore_index=255) + \
                        args.gamma * feat_loss(out, f_t)
                else:  # both
                    loss = response_kd_loss(out['out'], z_t, mask, args.alpha, args.beta,
                                            args.tau) + args.gamma * feat_loss(out, f_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if args.iters:
                if len(losses) % 5 == 0:
                    print(f'iter {len(losses):3d}/{args.iters}  loss {loss.item():.3f}')
                if len(losses) >= args.iters:
                    break
        if args.iters:
            if len(losses) >= args.iters:
                break
            continue

        student.eval()
        conf = torch.zeros(21, 21, dtype=torch.long)
        with torch.no_grad():
            for img, mask in val_loader:
                pred = student(img.to(device))['out'].argmax(1).cpu()
                conf = update_confusion(conf, pred, mask)
        miou = 100 * mean_iou(conf)
        ep_loss = sum(losses[-len(train_loader):]) / len(train_loader)
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start
        history.append({'epoch': epoch, 'loss': ep_loss, 'miou': miou, 'lr': lr,
                        'time': elapsed})
        print(f'epoch {epoch}/{args.epochs}  loss {ep_loss:.3f}  val mIoU {miou:.1f}'
              f'  lr {lr:.2e}  ({elapsed:.0f} s)', flush=True)

        scheduler.step(miou)
        if miou > best + args.tol:
            best, best_epoch, bad_epochs = miou, epoch, 0
            torch.save(student.state_dict(), f'checkpoints/student_{tag}.pth')
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f'Early stopping: no improvement > {args.tol} for {args.patience} epochs')
                break

    os.makedirs('plots', exist_ok=True)
    if args.iters:
        first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
        print(f'Kick-start finished: average loss {first:.3f} (first 5) -> {last:.3f} (last 5)')
    else:
        total = time.time() - start
        print(f'Best val mIoU {best:.1f} at epoch {best_epoch}; {len(history)} epochs, '
              f'{total:.0f} s total. Saved checkpoints/student_{tag}.pth')
        os.makedirs('logs', exist_ok=True)
        with open(f'logs/history_{tag}.json', 'w') as f:
            json.dump({'args': vars(args), 'params': n_params, 'best_miou': best,
                       'best_epoch': best_epoch, 'epochs': len(history),
                       'train_time_s': total, 'history': history}, f, indent=1)

    ipe = len(train_loader)
    fig, ax = plt.subplots(1, 2 if history else 1, figsize=(10 if history else 6, 4),
                           squeeze=False)
    ax[0, 0].plot([i / ipe for i in range(len(losses))], losses, lw=0.5, alpha=0.5,
                  label='iteration')
    if history:
        ax[0, 0].plot([h['epoch'] for h in history], [h['loss'] for h in history], 'o-',
                      ms=3, label='epoch mean')
        ax[0, 0].legend()
        ax[0, 1].plot([h['epoch'] for h in history], [h['miou'] for h in history], 'o-', ms=3,
                      color='tab:green')
        ax[0, 1].axvline(best_epoch, ls='--', color='gray')
        ax[0, 1].set_xlabel('epoch')
        ax[0, 1].set_ylabel('val mIoU (%)')
        ax[0, 1].set_title('Validation mIoU')
    ax[0, 0].set_xlabel('epoch')
    ax[0, 0].set_ylabel('loss')
    ax[0, 0].set_title(f'Training loss ({args.model}, kd={args.kd})')
    fig.tight_layout()
    fig.savefig(f'plots/loss_{tag}.png', dpi=120)
    print(f'Saved plots/loss_{tag}.png')


if __name__ == '__main__':
    main()
