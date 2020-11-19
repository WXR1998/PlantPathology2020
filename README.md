# Plant Pathology 2020

Final Project of Machine Learning Course

## Download Dataset

_under ./data:_ 
    
    $> kaggle competitions download -c plant-pathology-2020-fgvc7
    $> unzip *.zip
    
## File Usage

### loader
#### PlantPathology
- 从数据集中读取数据，并转化成便于操作的格式。
- 读取数据有`pickle`的缓存，当磁盘上存在缓存文件时将默认优先使用缓存。
- 默认缩放图片到`384*512`的大小。
- 为了节省内存占用，图片像素点取值范围`uint8` `[0, 255]`。
- 图片读入后会将其rescale到`[0, 1]`并减去训练集上的均值。
#### PlantPathology\_torch
- 包装为`pytorch`的数据集类，实现了对应的方法，可以和`pytorch`的`dataloader`对接。
- 将`train.csv`按比例分为训练集和验证集。

### src
#### Logger
- 日志函数。
#### Visualize
- 把减去均值的`tensor`格式的图`([3, H, W])`还原成原图`([H, W, 3])`并加上均值后clip到`[0, 1]`中。
- 调用`matplotlib`的相关方法显示图片。

### models
#### Model
- 所有模型的基类，定义了储存和读取`state_dict`的方法。
- 和模型训练相关的参数，如`epoch_num`等也在这个类中定义。
#### TestModel
- 测试用的模型。4个卷积层和2个全连接层。 测试集上能达到`0.83`。

### Main
- 训练和测试的入口。
- 测试会把输出放到`result/ModelName/result.csv`下。
- 训练时会把checkpoint放到`ckpt/ModelName/xxx.pth`下，注意读取checkpoint文件时一定要以`.pth`结尾。