import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def show_pca_D():
    csv_path = './result/pca.csv'
    res = pd.read_csv(csv_path)

    ori = []
    pca = []
    ds = []
    for l in range(len(res)):
        record = res.loc[l]
        d = record['n_features']
        pca_acc = float(record['pca_acc'][7:13])
        original_acc = float(record['original_acc'][7:13])
        ori.append(original_acc)
        pca.append(pca_acc)
        ds.append(d)

    ori = [sum(ori) / len(ori)] * len(ori)

    plt.figure(figsize=(8, 4))
    plt.plot(ds, pca, label='PCA dim reduced')
    plt.plot(ds, ori, label='Original', ls='--')
    plt.title('Comparison of the accuracy of dimension reduced result and original result')
    plt.xlabel('D')
    plt.ylabel('acc')
    plt.legend()
    plt.show()

def show_pca_res():
    log_path = './result/pca.log'
    with open(log_path, 'r') as fin:
        res = fin.readlines()
    res = [[float(y[0]), float(y[1])] for y in (t.strip().split(', ') for t in res)]
    res = np.array(res)

    plt.figure(figsize=(5, 4))
    plt.boxplot(res, labels=['PCA dim reduced', 'Original'])
    plt.ylabel('acc')
    plt.title('Comparison of the classifier accuracy')
    plt.show()

def show_model_performances():
    labels = ['LeNet', 'AlexNet', 'VGGNet', 'ResNet']
    res = [0.62470, 0.84480, 0.85580, 0.82194]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, height=res, width=0.4)
    plt.ylim(0.6, 0.9)
    plt.ylabel('acc')
    plt.xlabel('models')
    plt.title('Accuracy performance of models on the test set')
    plt.show()