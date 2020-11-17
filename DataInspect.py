from loader.PlantPathology_torch import PlantPathology_torch as dataset

data_train = dataset(train=True)
data_test = dataset(train=False)

print(data_train.data.shape)
print(data_test.data.shape)
print(data_train.targets.shape)
