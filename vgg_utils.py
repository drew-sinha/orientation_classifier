import numpy as np
import torch
from torchvision import transforms
from torchvision.models import vgg
from torch import nn


def get_VGG_image(image):
    '''Converts a grayscale image as numpy array to an appropriately rescaled RGB torch Tensor for VGG'''
    image = np.stack((image,)*3, 0)   # Replicate in 3 channels

    # Normalize appropriately for VGG
    image = torch.from_numpy(image)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # Normalization per docs
                                 std=[0.229, 0.224, 0.225])
    return normalize(image)


def make_compatible_VGG(input_shape):
    '''Creates a modified VGG_19 batch-normalized network usuable for binary classification;
        this CNN is compatible with arbitrary-sized images as input
        (as opposed to the 224x224 images originally used by VGG)

    Parameters:
        input_shape: Desired shape of input image

    Returns:
        torch.nn.Module 19-layer VGG model; this model includes batch normalization
            and is derived from the pretrained torch VGG model
    '''

    base_nn = vgg.vgg19_bn(pretrained=True) # Setting num_classes here blows up importing pretrained model

    # Freeze the parameters for the convolutional part of the network; will be modifying only fully-connected layers
    for param in base_nn.features.parameters():
        param.requires_grad = False

    # Modify dimensions of first linear layer based on input image
    # First calculate final feature map size after several max pools (VGG's conv2ds don't change size)
    out_size = []
    for dim_size in input_shape:
        size = dim_size
        for i in range(5):
            size -= 2
            size /= 2
            size = int(size)+1
        out_size.append(size)

    base_nn.classifier[0] = nn.Linear(512 * out_size[0] * out_size[1], 4096)
    nn.init.normal_(base_nn.classifier[0].weight, 0, 0.01)
    nn.init.constant_(base_nn.classifier[-1].bias,0)

    # Now set the number of classes
    base_nn.classifier[-1] = nn.Linear(4096,2)
    nn.init.normal_(base_nn.classifier[-1].weight, 0, 0.01)
    nn.init.constant_(base_nn.classifier[-1].bias, 0)

    return base_nn
