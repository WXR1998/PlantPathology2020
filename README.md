# Plant Pathology 2020

Final Project of Machine Learning Course

## Download Dataset

_under ./data:_ 
    
    $> kaggle competitions download -c plant-pathology-2020-fgvc7
    $> unzip *.zip
    
## File Usage

### DataLoader
- 从数据集中读取数据，并转化成便于操作的格式。
- 读取数据有pickle的缓存，当磁盘上存在缓存文件时将默认优先使用缓存。
- 默认缩放图片到384 $ \times $ 512的大小。
- 图片像素点取值范围[0, 1]，且已经减去其训练集通道均值，故获得的像素点亮度需要加上均值才能还原成原图。
