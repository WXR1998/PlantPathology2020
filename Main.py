from torchvision import transforms
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from models.TestModel import TestModel as Model
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
            for data in tqdm(self.data_loader_train):
                x_train, y_train = data
                y_train = np.argmax(y_train, axis=1)
                x_train, y_train = Variable(x_train), Variable(y_train)
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

            for data in tqdm(self.data_loader_valid):
                x_valid, y_valid = data
                y_valid = np.argmax(y_valid, axis=1)
                x_valid, y_valid = Variable(x_valid), Variable(y_valid)
                batch_size = x_valid.shape[0]
                outputs = model(x_valid)

                _, pred = torch.max(outputs.data, 1)
                gt = y_valid.data
                batch_valid_acc = torch.sum(gt == pred)

                epoch_valid_acc += batch_valid_acc

            Log.log(Log.INFO, 'Epoch loss = %.5f, Epoch train acc = %.5f, Epoch valid acc = %.5f' %
                    (epoch_loss / len(self.data_train), epoch_acc / len(self.data_train), epoch_valid_acc / len(self.data_valid)))
            if epoch % model.save_interval == 0 or epoch == n_epochs - 1:
                model.save('%03d.pth' % epoch)

    def test(self, path=None):
        model = Model()

        Log.log(Log.INFO, 'Start testing...')
        result = pd.DataFrame(columns=['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab'])
        image_id = 0
        for data in tqdm(self.data_loader_test):
            x_test, y_test = data
            x_test = Variable(x_test)
            batch_size = x_test.shape[0]
            outputs = model(x_test)
            _, pred = torch.max(outputs.data, 1)
            pred = pred.numpy()
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

if __name__ == '__main__':
    oper = Operation()
    # oper.train()
    oper.test()