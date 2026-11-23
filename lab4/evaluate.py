import argparse
import json
import os
import torch
from torch.utils.data import DataLoader, Subset

from models import build_model
from utils.dataset import get_voc
from utils.metrics import update_confusion, mean_iou, iou_per_class
from utils.profiler import ms_per_image
from utils.teacher import load_teacher
from utils.visualization import save_comparison

CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

RUNS = [('student', 'none', 'Student, no KD'),
        ('student', 'response', 'Student, response KD'),
        ('student', 'feature', 'Student, feature KD'),
        ('custom', 'none', 'LiteSeg, no KD'),
        ('custom', 'both', 'LiteSeg, resp+feat KD')]


def image_miou(pred, mask, n=21):
    """mIoU of one image over the classes present in its ground truth or prediction."""
    conf = update_confusion(torch.zeros(n, n, dtype=torch.long), pred, mask)
    return mean_iou(conf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data')
    parser.add_argument('--size', type=int, default=320)
    parser.add_argument('-b', '--batch', type=int, default=16)
    parser.add_argument('--limit', type=int, default=0, help='evaluate only the first N images')
    parser.add_argument('--examples', type=int, default=4,
                        help='number of best / worst images to save for the custom model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    ds = get_voc(args.data, 'val', args.size)
    if args.limit:
        ds = Subset(ds, range(args.limit))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)

    models = {'Teacher (FCN-ResNet50)': load_teacher(device)}
    for arch, kd, name in RUNS:
        tag = kd if arch == 'student' else f'custom_{kd}'
        path = f'checkpoints/student_{tag}.pth'
        if os.path.exists(path):
            m = build_model(arch, pretrained=False).to(device).eval()
            m.load_state_dict(torch.load(path, map_location=device))
            models[name] = m
        else:
            print(f'Skipping {name}: {path} not found')

    rows, examples, per_image, per_class = [], {}, {}, {}
    all_preds = {}
    with torch.no_grad():
        for name, model in models.items():
            conf = torch.zeros(21, 21, dtype=torch.long)
            scores, preds = [], []
            for i, (img, mask) in enumerate(loader):
                pred = model(img.to(device))['out'].argmax(1).cpu()
                conf = update_confusion(conf, pred, mask)
                scores += [image_miou(p, m) for p, m in zip(pred, mask)]
                preds.append(pred.to(torch.uint8))
                if i == 0:
                    examples[name] = pred[:4]
                    first_img, first_mask = img[:4], mask[:4]
            per_image[name] = scores
            all_preds[name] = torch.cat(preds)
            per_class[name] = {c: 100 * v for c, v in zip(CLASSES, iou_per_class(conf).tolist())}
            params = sum(p.numel() for p in model.parameters())
            ms_dev = ms_per_image(model, device, args.size)
            ms_cpu = ms_per_image(model.cpu(), torch.device('cpu'), args.size)
            model.to(device)
            rows.append((name, 100 * mean_iou(conf), params, ms_dev, ms_cpu))
            print(f'{name}: mIoU {rows[-1][1]:.1f}', flush=True)

    print(f'\n{"Model":<26}{"mIoU":>8}{"Params":>14}{f"ms/img ({device.type})":>16}'
          f'{"ms/img (cpu)":>14}')
    for name, miou, params, ms, ms_cpu in rows:
        print(f'{name:<26}{miou:>8.1f}{params:>14,}{ms:>16.1f}{ms_cpu:>14.1f}')

    os.makedirs('plots', exist_ok=True)
    save_comparison(first_img, first_mask, examples, 'plots/compare.png')
    print('\nSaved plots/compare.png')

    # Extra qualitative results: best and worst images for our best custom model.
    focus = next((n for n in ['LiteSeg, resp+feat KD', 'LiteSeg, no KD', 'Student, no KD']
                  if n in per_image), None)
    if focus:
        scores = torch.tensor(per_image[focus])
        order = scores.argsort()
        for label, idx in [('success', order.flip(0)[:args.examples]),
                           ('failure', order[:args.examples])]:
            imgs = torch.stack([ds[int(i)][0] for i in idx])
            masks = torch.stack([ds[int(i)][1] for i in idx])
            preds = {n: all_preds[n][idx].long() for n in models}
            save_comparison(imgs, masks, preds, f'plots/{label}.png')
            print(f'Saved plots/{label}.png ({focus}, val idx {idx.tolist()}, '
                  f'per-image mIoU {[round(100 * scores[i].item(), 1) for i in idx]})')

    os.makedirs('logs', exist_ok=True)
    with open('logs/eval_results.json', 'w') as f:
        json.dump({'device': device.type,
                   'rows': [dict(zip(['model', 'miou', 'params', 'ms_dev', 'ms_cpu'], r))
                            for r in rows],
                   'per_class': per_class}, f, indent=1)
    print('Saved logs/eval_results.json')


if __name__ == '__main__':
    main()
