import argparse

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from model import autoencoderMLP4Layer

DATA_DIR = './data'   # change if using Colab

CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def main():
    print('running main ...')

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-z', metavar='bottleneck size', type=int, help='int [8]')

    args = argParser.parse_args()

    save_file = args.s
    bottleneck_size = args.z

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    test_transform = transforms.Compose([transforms.ToTensor()])
    test_set = FashionMNIST(DATA_DIR, train=False, download=False, transform=test_transform)

    N_input = 28 * 28
    N_output = N_input
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
    model.load_state_dict(torch.load(save_file, weights_only=True))
    model.to(device)
    model.eval()

    print('enter an index from 0 to', len(test_set) - 1, '(or -1 to quit)')

    while True:
        try:
            idx = int(input('index > '))
        except ValueError:
            print('not an integer')
            continue

        if idx < 0:
            break
        if idx >= len(test_set):
            print('index out of range')
            continue

        label = test_set.targets[idx].item()
        print('label =', label, '(' + CLASSES[label] + ')')

        img = test_set.data[idx].type(torch.float32) / 255.0
        img = img.view(1, N_input).to(device=device)

        with torch.no_grad():
            output = model(img)

        img = img.view(28, 28).cpu()
        output = output.view(28, 28).cpu()

        f = plt.figure()
        f.add_subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('input: ' + CLASSES[label])
        f.add_subplot(1, 2, 2)
        plt.imshow(output, cmap='gray')
        plt.title('reconstruction')
        plt.show()


if __name__ == '__main__':
    main()
