from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
import cv2
from glob import glob

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((256 + 30, 256 + 30)),
    transforms.RandomCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

trainB = ImageFolder(os.path.join('dataset', 'training', 'trainB'), train_transform)

trainB_loader = DataLoader(trainB, batch_size=2, shuffle=True)

trainB_iter = iter(trainB_loader)
real_B, _ = next(trainB_iter)

noise_level = 100 / 100 * 0
noise = torch.randn_like(real_B) * noise_level
real_B_with_noise_0 = real_B + noise

noise_level = 100 / 100 * 0.2
noise = torch.randn_like(real_B) * noise_level
real_B_with_noise_1 = real_B + noise

noise_level = 100 / 100 * 1
noise = torch.randn_like(real_B) * noise_level
real_B_with_noise_2 = real_B + noise

noise_level = 100 / 100 * 5
noise = torch.randn_like(real_B) * noise_level
real_B_with_noise_5 = real_B + noise

noise_level = 100 / 100 * 10
noise = torch.randn_like(real_B) * noise_level
real_B_with_noise_10 = real_B + noise




img_0 = RGB2BGR(tensor2numpy(denorm(real_B_with_noise_0[0])))
img_1 = RGB2BGR(tensor2numpy(denorm(real_B_with_noise_1[0])))
img_2 = RGB2BGR(tensor2numpy(denorm(real_B_with_noise_2[0])))
img_5 = RGB2BGR(tensor2numpy(denorm(real_B_with_noise_5[0])))
img_10 = RGB2BGR(tensor2numpy(denorm(real_B_with_noise_10[0])))

cv2.imwrite(os.path.join("test0.png"), img_0 * 255.0)
cv2.imwrite(os.path.join("test1.png"), img_1 * 255.0)
cv2.imwrite(os.path.join("test2.png"), img_2 * 255.0)
cv2.imwrite(os.path.join("test5.png"), img_5 * 255.0)
cv2.imwrite(os.path.join("test10.png"), img_10 * 255.0)