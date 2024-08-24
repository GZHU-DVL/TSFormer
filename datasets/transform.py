from PIL import Image

from torchvision import transforms


def image_transforms(load_size):

    return transforms.Compose([
        # transforms.CenterCrop(size=(178, 178)),  # for CelebA
        transforms.Resize(size=load_size, interpolation=Image.BILINEAR),   
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def mask_transforms(load_size):

    return transforms.Compose([
        transforms.Resize(size=load_size, interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

# def image_transforms_1024(load_size):
#     def resize_and_random_crop(image):
#         width, height = image.size
#         if width < 1024:
#             # If width is less than 256, stretch width while keeping height unchanged
#             image = transforms.functional.resize(image, (1024, height), interpolation=Image.BILINEAR)
#         elif height < 1024:
#             # If height is less than 256, stretch height while keeping width unchanged
#             image = transforms.functional.resize(image, (width, 1024), interpolation=Image.BILINEAR)
        
#         if width < 1024 or height < 1024:
#             image = transforms.functional.resize(image, (1024, 1024), interpolation=Image.BILINEAR)
#         # 随机裁剪
#         i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(1024, 1024))
#         image = transforms.functional.crop(image, i, j, h, w)
#         return image

#     return transforms.Compose([
#         # transforms.CenterCrop(size=(1024, 1024)),  # for CelebA
#         transforms.Lambda(resize_and_random_crop),
#         transforms.Resize(size=1024, interpolation=Image.BILINEAR),   
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])

# def mask_transforms_1024(load_size):
#     return transforms.Compose([
#         transforms.Resize(size=1024, interpolation=Image.NEAREST),
#         transforms.ToTensor()
#     ])
