from scipy import misc
import os, cv2, torch
import numpy as np

def load_test_data(image_path, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img

def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')

def cam(x, size = 256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0

def imagenet_norm(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.224, 0.225]
    mean = torch.FloatTensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    std = torch.FloatTensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    return (x - mean) / std

def denorm(x):
    return x * 0.5 + 0.5

def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def create_binary_mask(img):
    """
       Преобразует все ненулевые (не черные) пиксели тензора изображения в белые.
       Предполагается, что тензор изображения имеет формат (B, C, H, W) и находится на устройстве GPU или CPU.

       Args:
       tensor (torch.Tensor): Тензор изображения.

       Returns:
       torch.Tensor: Тензор изображения с преобразованными пикселями.
       """
    # Создаем маску для всех ненулевых пикселей
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, th1 = cv2.threshold(gray, 16, 255, cv2.THRESH_BINARY)
    return th1



def apply_mask(tensor, mask):
    # Assuming mask is a PyTorch tensor, ensure it's on the same device as tensor.
    mask_tensor = mask.to(tensor.device).float()

    # Expand the mask to match the number of channels of the tensor.
    mask_tensor = mask_tensor.expand(tensor.size(0), tensor.size(1), -1, -1)

    # Apply the mask to the tensor.
    masked_tensor = tensor * mask_tensor
    return masked_tensor


def resize_images(path1):

    for img_path in os.listdir(path1):
        if img_path.endswith((".png", ".jpg", ".jpeg", ".tif")):  # Проверяем, что файл является изображением.
            img = cv2.imread(os.path.join(path1, img_path))
            mask = create_binary_mask(img)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                cnt = max(contours, key=cv2.contourArea)  # Берем наибольший контур.

                # Находим экстремальные точки на контуре
                leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
                rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
                topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
                bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

                # Отображаем экстремальные точки на изображении
                cv2.drawContours(mask, [cnt], -1, (255, 0, 0), 3)
                cv2.circle(mask, leftmost, 8, (255, 0, 0), -1)
                cv2.circle(mask, rightmost, 8, (255, 0, 0), -1)
                cv2.circle(mask, topmost, 8, (255, 0, 0), -1)
                cv2.circle(mask, bottommost, 8, (255, 0, 0), -1)

                cv2.imwrite(r"C:\Users\senic\PycharmProjects\BP_last\test\{}".format(img_path), mask)

if __name__ == "__main__":
    resize_images(r"C:\Users\senic\PycharmProjects\BP_last\UGATIT-pytorch\dataset\training\testB")

