from torchvision import transforms
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse
import pynvml
import pandas as pd
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
                Log.log(Log.INFO, f'Use GPU:{i} for training.')
                return torch.device(f'cuda:{i}')
        Log.log(Log.INFO, 'Use CPU:0 for training.')
        return torch.device('cpu:0')

    def __init__(self, model):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip()])

        self.data_train = dataset(subset='Train', transform=transform)
        self.data_valid = dataset(subset='Valid', transform=transform)
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

    def train(self, path=None, epoch=None):
        model = self.model().to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        cost = torch.nn.CrossEntropyLoss()

        n_epochs = self.model.epoch_num if epoch is None else epoch
        start_epoch = 0
        if path is not None:
            model.load(path)
            start_epoch = int(path[:-4]) + 1

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

            training_log.loc[len(training_log)] = [epoch+1,
                                                   (epoch_loss / len(self.data_train)).item(),
                                                   (epoch_acc / len(self.data_train)).item(),
                                                   (epoch_valid_acc / len(self.data_valid)).item()]
            Log.log(Log.INFO, 'Epoch loss = %.5f, Epoch train acc = %.5f, Epoch valid acc = %.5f' %
                    (epoch_loss / len(self.data_train), epoch_acc / len(self.data_train), epoch_valid_acc / len(self.data_valid)))
            if (epoch+1) % model.save_interval == 0 or (epoch+1) == n_epochs:
                model.save('%03d.pth' % (epoch+1))

        training_log.to_csv(self.full_log_path, index=False)

    def test(self, path=None):
        model = self.model().to(self.device)
        model.load(path)

        Log.log(Log.INFO, 'Start testing...')
        result = pd.DataFrame(columns=['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab'])
        image_id = 0

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

parser = argparse.ArgumentParser(description='Plant Pathology 2020.')
parser.add_argument('--mode', default='Train', choices=['Train', 'Test'])
parser.add_argument('--ckpt', default=None)
parser.add_argument('--model', default='ResNet18',
                    help=str(list(_model_dict.keys())))
parser.add_argument('--epoch', default=None, type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    oper = Operation(args.model)

    if args.mode == 'Train':
        oper.train(path=args.ckpt, epoch=args.epoch)

    if args.ckpt != '':
        oper.test(path=args.ckpt)
    else:
        oper.test()
