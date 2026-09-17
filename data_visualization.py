import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

DATA_DIR = './data'

CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_set = FashionMNIST(DATA_DIR, train=True, download=False,
transform=transforms.Compose([transforms.ToTensor()]))

idx = int(input('index > '))
img = train_set.data[idx]
label = train_set.targets[idx].item()

print('label =', label, '(' + CLASSES[label] + ')')
plt.imshow(img, cmap='gray')
plt.title(CLASSES[label])
plt.show()