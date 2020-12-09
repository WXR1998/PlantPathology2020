from torchvision import transforms
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse

# from models.TestModel import TestModel as Model
# from models.AlexNet import AlexNet as Model
# from models.LeNet import LeNet as Model
from models.VGG import VGG13 as Model

from src import Logger as Log
from src import Visualize
from loader.PlantPathology_torch import PlantPathology_torch as dataset

class Operation:
    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip()])

        self.data_train = dataset(subset='Train', transform=transform)
        self.data_valid = dataset(subset='Valid', transform=transform)
        self.data_test = dataset(subset='Test', transform=transform)

        print(self.data_train)
        print(self.data_valid)
        print(self.data_test)

        self.data_loader_train = torch.utils.data.DataLoader(dataset=self.data_train,
                                                             batch_size=Model.batch_size,
                                                             shuffle=True)
        self.data_loader_valid = torch.utils.data.DataLoader(dataset=self.data_valid,
                                                             batch_size=Model.batch_size,
                                                             shuffle=True)
        self.data_loader_test = torch.utils.data.DataLoader(dataset=self.data_test,
                                                            batch_size=Model.batch_size,
                                                            shuffle=False)

    def result_path(self, model):
        root_dir = './result/%s' % model.model_name
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        return os.path.join(root_dir, 'result.csv')

    def train(self, path=None):
        model = Model()
        device = model.device
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters())
        cost = torch.nn.CrossEntropyLoss()

        n_epochs = Model.epoch_num
        start_epoch = 0
        if path is not None:
            model.load(path)
            start_epoch = int(path[:-4]) + 1

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
                x_train, y_train = Variable(x_train).to(device), Variable(y_train).to(device)
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

            iter_cnt = 0
            valid_iter = tqdm(self.data_loader_valid)
            for data in valid_iter:
                x_valid, y_valid = data
                y_valid = np.argmax(y_valid, axis=1)
                x_valid, y_valid = Variable(x_valid).to(device), Variable(y_valid).to(device)
                batch_size = x_valid.shape[0]
                outputs = model(x_valid)

                _, pred = torch.max(outputs.data, 1)
                gt = y_valid.data
                batch_valid_acc = torch.sum(gt == pred)

                epoch_valid_acc += batch_valid_acc
                iter_cnt += batch_size
                valid_iter.set_description('[V] acc %.3f' % (epoch_valid_acc / iter_cnt))

            Log.log(Log.INFO, 'Epoch loss = %.5f, Epoch train acc = %.5f, Epoch valid acc = %.5f' %
                    (epoch_loss / len(self.data_train), epoch_acc / len(self.data_train), epoch_valid_acc / len(self.data_valid)))
            if (epoch+1) % model.save_interval == 0 or (epoch+1) == n_epochs:
                model.save('%03d.pth' % (epoch+1))

    def test(self, path=None):
        model = Model()
        device = model.device
        model = model.to(device)
        model.load(path)

        Log.log(Log.INFO, 'Start testing...')
        result = pd.DataFrame(columns=['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab'])
        image_id = 0

        for data in tqdm(self.data_loader_test):
            x_test, y_test = data
            x_test = Variable(x_test).to(device)
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

parser = argparse.ArgumentParser(description='Plant Pathology 2020.')
parser.add_argument('--mode', default='Train', choices=['Train', 'Test'])
parser.add_argument('--ckpt', default='')

if __name__ == '__main__':
    oper = Operation()
    args = parser.parse_args()

    if args.mode == 'Train':
        oper.train()

    if args.ckpt != '':
        oper.test(args.ckpt)
    else:
        oper.test()
