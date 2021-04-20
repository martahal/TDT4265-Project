import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms


def show_training_image(images, targets):
    num_images = 1
    # helper function
    image = images[0]
    np_image = image.numpy()
    plt.imshow((np.transpose(np_image, (1,2,0))* 255).astype(np.uint8))

    #image = torch.empty(3, 224, 224).uniform_(0, 1)
    #mean = (0.5, 0.5, 0.5)
    #std = (0.5, 0.5, 0.5)
    #norm = transforms.Normalize(mean, std)
    #image_norm = norm(image)
    #image_restore = image_norm * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)

    #plt.imshow(np_image)

    #def imshow(image):
    #    # unnormalize image
    #    #image = image / 2 + 0.5
    #    # convert from Tensor image
    #    image = image.permute(1, 2, 0)
    #    plt.imshow(image)
    #imshow(images[0])
    #fig, axes = plt.subplots(nrows=2, ncols=int(np.ceil(num_images / 2)),
    #                         figsize=(num_images, 4))  # , sharex=True, sharey=True)#, figsize=(num_images + 4, 4))
    #fig.tight_layout()
    #for idx in np.arange(num_images):
    #    ax = fig.add_subplot(2, int(np.ceil(num_images / 2)), idx + 1, xticks=[], yticks=[], frame_on=False)
    #    imshow(images[idx])

    pass