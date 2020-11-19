import numpy as np
import matplotlib.pyplot as plt

def imageFromTensor(img, mean):
    assert len(img) == 3 and img.shape[0] == 3
    img = np.array(img).transpose((1, 2, 0))
    img = np.clip(img + mean, 0, 1)
    return img

def showImage(img, text=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    if text is not None:
        plt.title(text)
    plt.show()