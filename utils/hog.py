import numpy as np
from PIL import Image

from torchvision import transforms

from skimage.feature import hog


def tensor_to_image():

    return transforms.ToPILImage()


def image_to_tensor():

    return transforms.ToTensor()


def image_to_hog(image):

    image = np.array(tensor_to_image()(image))

    hog_image = image_to_tensor()(Image.fromarray(hog(image, orientations=9, pixels_per_cell=(8, 8),
                                                      cells_per_block=(8, 8), visualize=True, multichannel=True)[1]))

    return hog_image

