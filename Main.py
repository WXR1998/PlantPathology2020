'''
    PCA:
        特征提取网络-全连接分类器这部分的参数数量过多，而其中有用的参数部分只占一小部分
        所以利用PCA可以将这部分的参数提取出来

        训练时，使用全量网络进行训练。之后得到正常的特征提取器，和针对此特征提取器输出的
        全连接分类器。然后将特征提取器的输出在训练集全集上的输出拿出来做PCA：得到一个
        (train_sample, feature_dimension)的矩阵，对此矩阵做PCA，得到一个
        (train_sample, reduced_dimension)的矩阵，且feature_dimension >> reduced_dimension.
        再针对这个新的数据集训练一个简单的全连接分类器即可。

        ** 在显存受限场景下，进行Inference只需保存特征提取器(GPU)、简单分类器(GPU)、PCA降维
        矩阵(CPU)即可。 **
'''

from torchvision import transforms
import torch
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import os
import argparse
from sklearn import decomposition
import pynvml
import pandas as pd
import pickle
pynvml.nvmlInit()

from loader.PlantPathology_torch import PlantPathology_torch as dataset
from models import import_model, _model_dict
from src import Logger as Log
from src import Visualize

class Operation:
    log_path = './logs/'

    def get_available_device(self):
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_ratio = mem_info.used / mem_info.total
            if used_ratio < 0.1:
                Log.log(Log.INFO, f'Use GPU:{i} for running.')
                return torch.device(f'cuda:{i}')
        Log.log(Log.INFO, 'Use CPU:0 for running.')
        return torch.device('cpu:0')

    def __init__(self, model, vr):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip()])

        self.data_train = dataset(subset='Train', transform=transform, valid_ratio=vr)
        self.data_valid = dataset(subset='Valid', transform=transform, valid_ratio=vr)
        self.data_test = dataset(subset='Test', transform=transform)

        print(self.data_train)
        print(self.data_valid)
        print(self.data_test)

        self.model = import_model(model)
        Log.log(Log.INFO, f'Running on model [ {model} ].')

        self.data_loader_train = torch.utils.data.DataLoader(dataset=self.data_train,
                                                             batch_size=self.model.batch_size,
                                                             shuffle=True)
        self.data_loader_valid = torch.utils.data.DataLoader(dataset=self.data_valid,
                                                             batch_size=self.model.batch_size,
                                                             shuffle=True)
        self.data_loader_test = torch.utils.data.DataLoader(dataset=self.data_test,
                                                            batch_size=self.model.batch_size,
                                                            shuffle=False)

        self.device = self.get_available_device()
        dataset.clear()

        if not os.path.exists(os.path.join(self.log_path, model)):
            os.makedirs(os.path.join(self.log_path, model))
        self.full_log_path = os.path.join(self.log_path, model, 'log_data.csv')

    def result_path(self, model):
        root_dir = './result/%s' % model.model_name
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        return os.path.join(root_dir, 'result.csv')

    def train(self, path=None, epochs=None):
        model = self.model().to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        cost = torch.nn.CrossEntropyLoss()

        n_epochs = self.model.epoch_num if epochs is None else epochs
        start_epoch = 0
        if path is not None:
            start_epoch = model.load(path)

        training_log = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'val_acc'])

        Log.log(Log.INFO, 'Start training...')
        for epoch in range(start_epoch, n_epochs):
            Log.log(Log.INFO, '-'*20)
            Log.log(Log.INFO, 'Epoch %d/%d'%(epoch+1, n_epochs))
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_valid_acc = 0.0

            train_iter = tqdm(self.data_loader_train)
            iter_cnt = 0
            for data in train_iter:
                x_train, y_train = data
                y_train = np.argmax(y_train, axis=1)
                x_train, y_train = Variable(x_train).to(self.device), \
                                   Variable(y_train).to(self.device)
                batch_size = x_train.shape[0]
                outputs = model(x_train)
                optimizer.zero_grad()
                loss = cost(outputs, y_train)

                loss.backward()
                optimizer.step()
                batch_loss = loss.data * batch_size

                _, pred = torch.max(outputs.data, 1)
                gt = y_train.data
                batch_acc = torch.sum(gt == pred)

                epoch_loss += batch_loss
                epoch_acc += batch_acc
                iter_cnt += batch_size

                train_iter.set_description('[T] acc %.3f, loss %.3f' % (epoch_acc / iter_cnt, epoch_loss / iter_cnt))

            torch.cuda.empty_cache()

            iter_cnt = 0
            with torch.no_grad():
                valid_iter = tqdm(self.data_loader_valid)
                for data in valid_iter:
                    x_valid, y_valid = data
                    y_valid = np.argmax(y_valid, axis=1)
                    x_valid, y_valid = Variable(x_valid).to(self.device), \
                                       Variable(y_valid).to(self.device)
                    batch_size = x_valid.shape[0]
                    outputs = model(x_valid)

                    _, pred = torch.max(outputs.data, 1)
                    gt = y_valid.data
                    batch_valid_acc = torch.sum(gt == pred)

                    epoch_valid_acc += batch_valid_acc
                    iter_cnt += batch_size
                    valid_iter.set_description('[V] acc %.3f' % (epoch_valid_acc / iter_cnt))

            torch.cuda.empty_cache()

            training_log.loc[len(training_log)] = {
                'epoch': int(epoch+1),
                'train_loss': (epoch_loss / len(self.data_train)).item(),
                'train_acc': (epoch_acc / len(self.data_train)).item(),
                'val_acc': (epoch_valid_acc / len(self.data_valid)).item()
            }
            Log.log(Log.INFO, 'Epoch loss = %.5f, Epoch train acc = %.5f, Epoch valid acc = %.5f' %
                    (epoch_loss / len(self.data_train), epoch_acc / len(self.data_train), epoch_valid_acc / len(self.data_valid)))
            if (epoch+1) % model.save_interval == 0 or (epoch+1) == n_epochs:
                model.save('%03d.pth' % (epoch+1))

        training_log.to_csv(self.full_log_path, index=False, float_format='%.3f')
        return model

    def test(self, path=None, trained_model=None):
        if trained_model is None:
            model = self.model().to(self.device)
            epoch = model.load(path)
        else:
            model = trained_model
            epoch = 'trained'

        Log.log(Log.INFO, f'Start testing with epoch [ {epoch} ] ...')
        result = pd.DataFrame(columns=['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab'])
        image_id = 0

        with torch.no_grad():
            for data in tqdm(self.data_loader_test):
                x_test, y_test = data
                x_test = Variable(x_test).to(self.device)
                batch_size = x_test.shape[0]
                outputs = model(x_test)
                _, pred = torch.max(outputs.data, 1)
                pred = pred.cpu().numpy()
                pred = np.eye(self.data_train.class_num)[pred]
                for i in range(batch_size):
                    result.loc[result.shape[0]] = {'image_id': 'Test_%d' % image_id,
                                                   'healthy': pred[i][0],
                                                   'multiple_diseases': pred[i][1],
                                                   'rust': pred[i][2],
                                                   'scab': pred[i][3]}
                    image_id += 1

        result.to_csv(self.result_path(model), float_format='%.0f', index=False)
        Log.log(Log.INFO, 'Evaluation success.')

    def get_features(self, path=None, trained_model=None):
        '''
        Get the feature outputs and the ground truths of the model.

        :param path: The path to the pth file.
        :param trained_model: If use trained model, pass this model to this method.
        :return: features, gt, model
            features: (n, 98304)
            gt: (n, )
            model: The loaded model
        '''
        if trained_model is None:
            model = self.model().to(self.device)
            epoch = model.load(path)
        else:
            model = trained_model
            epoch = 'trained'

        assert 'VGG' in model.model_name
        Log.log(Log.INFO, f'Start getting features with epoch [ {epoch} ] ...')

        features = None
        gt = None
        with torch.no_grad():
            for data in tqdm(self.data_loader_train):
                x, y = data
                y = np.argmax(y, axis=1)
                x, y = Variable(x).to(self.device), \
                       Variable(y).to(self.device)

                batch_gt = y.data
                if gt is None:
                    gt = batch_gt.cpu().numpy()
                else:
                    gt = np.concatenate((gt, batch_gt.cpu().numpy()), axis=0)

                batch_features = model.get_features(x)
                batch_features = batch_features.cpu().numpy()
                if features is None:
                    features = batch_features
                else:
                    features = np.concatenate((features, batch_features), axis=0)

        return features, gt, model

    def PCA(self, path=None, trained_model=None, use_train_set=False):
        '''
        Compare the original accuracy and PCA-dimension-reduced accuracy results.
        Here, the PCA-dimension-reduced results are represented in the original
        dimension basis. i.e. inverse transform the dimension-reduced results.

        :param path: The path to the pth file.
        :param trained_model: If use trained model, pass this model to this method.
        :param use_train_set: Which set to use, trainset or validset.
        :return:
        '''

        # Now feature is a (n, 98304) matrix. Perform PCA on it to extract useful dimension information.
        features, gt, model = self.get_features(path, trained_model)
        result_dir = './result/pca.csv'

        result = pd.DataFrame(columns=['n_features', 'pca_acc', 'original_acc'])

        ori_features = features
        for components in range(2, ori_features.shape[0]):
            pca_acc = 0
            ori_acc = 0
            features = ori_features
            pca = decomposition.PCA(n_components=components)
            pca_features = pca.fit_transform(features)

            # Here feature is a (n, 98304) matrix which has been performed PCA.
            features = pca.inverse_transform(pca_features)

            with torch.no_grad():
                features_tensor = torch.Tensor(features).to(self.device)
                ori_features_tensor = torch.Tensor(ori_features).to(self.device)
                gt_tensor = torch.Tensor(gt).to(self.device)
                batch_outputs = model.get_classification(features_tensor)
                batch_ori_outputs = model.get_classification(ori_features_tensor)

                _, pred = torch.max(batch_outputs.data, 1)
                _, ori_pred = torch.max(batch_ori_outputs.data, 1)
                batch_gt = gt_tensor
                batch_acc = torch.sum(batch_gt == pred)
                batch_ori_acc = torch.sum(batch_gt == ori_pred)
                pca_acc += batch_acc
                ori_acc += batch_ori_acc

            sample_count = len(self.data_train) if use_train_set else len(self.data_valid)
            print('%03d, %.5f | %.5f' % (components, pca_acc / sample_count, ori_acc / sample_count))
            result.loc[len(result)] = {
                'n_features': components,
                'pca_acc': pca_acc / sample_count,
                'original_acc': ori_acc / sample_count
            }

        result.to_csv(result_dir, float_format='%.3f', index=False)

parser = argparse.ArgumentParser(description='Plant Pathology 2020.')
parser.add_argument('--mode', default='PCA', choices=['Train', 'Test', 'PCA'])
parser.add_argument('--ckpt', default=None)
parser.add_argument('--model', default='VGG11',
                    help=str(list(_model_dict.keys())))
parser.add_argument('--epoch', default=None, type=int)
parser.add_argument('--vr', default=0.2, type=float)

if __name__ == '__main__':
    args = parser.parse_args()
    oper = Operation(args.model, args.vr)

    if args.mode == 'Train':
        oper.train(path=args.ckpt, epochs=args.epoch)
        oper.test(path=args.ckpt)
    elif args.mode == 'Test':
        oper.test(path=args.ckpt)
    elif args.mode == 'PCA':
        oper.PCA(path=args.ckpt)
