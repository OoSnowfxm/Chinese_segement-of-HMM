1、实现部分

data_utils.py 包含数据集处理相关函数接口

train.py 包含从训练集中获取参数的部分实现

test.py 包含对训练集的评估以及对测试集的分词过程



2、数据集部分

msr_training.txt 是根据utf8文件处理后的msr分词训练集

msr_test_gold.txt是msr数据集中和训练集最相似的测试集，分词效果最好



3、其他部分

output.txt 是HMM模型分词后生成的结果

\__pycache__是python文件缓存