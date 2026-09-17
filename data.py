from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms

DATA_DIR = './data'
transform = transforms.Compose([transforms.ToTensor()])
train_set = FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
test_set = FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)

print('train:', len(train_set), 'test:', len(test_set))
img, label = train_set[0]
print('shape:', img.shape, 'dtype:', img.dtype)
print('min:', img.min().item(), 'max:', img.max().item())
print('raw dtype:', train_set.data.dtype, 'raw max:', train_set.data.max().item())