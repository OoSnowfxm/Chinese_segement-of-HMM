# 引用自己写的python文件
from data_utils import *

# 训练参数，本质上是通过训练集统计得到参数
def TrainArg(filename):
    TrainSet = open(filename, encoding='gb18030')         # 读取训练集
    global LineNum, StatusSet, ObserveSet, TransProbMatrix
    global StatusDictionary, InitStatus, EmitProbMatrix
    for line in TrainSet:
        line = line.strip()
        LineNum += 1
        WordList = []
        for pos in range(len(line)):                    # 跳过空格
            if line[pos] == ' ': 
                continue
            WordList.append(line[pos])
        ObserveSet = ObserveSet | set(WordList)         # 训练集所有字的集合

        line = line.split(' ')
        LineState = []                                  # 这句话的状态序列  
        
        for word in line:
            if(len(word)==0):continue
            #print(word,len(word), getTag(word),end="")
            LineState.extend(getTag(word))
        #print(LineState)
        if(len(LineState)==0):
            continue
        InitStatus[LineState[0]] += 1                   # 计算初始状态概率分布，目前只取数值，在计算对数概率的时候变为概率

        for pos in range(len(LineState)-1):
            TransProbMatrix[LineState[pos]][LineState[pos+1]] += 1  # 状态转移概率的计算

        for pos in range(len(LineState)):
            StatusDictionary[LineState[pos]] += 1                   # 记录每一个状态的出现次数
            for st in StatusSet:
                if WordList[pos] not in EmitProbMatrix[st]:
                    EmitProbMatrix[st][WordList[pos]] = 0.0         # 如果发射为这个字的概率为0，也要在发射概率矩阵中体现出来

            EmitProbMatrix[LineState[pos]][WordList[pos]] += 1      # array_B用于计算发射概率

    #print(TransProbMatrix)
    setArg(ObserveSet, TransProbMatrix, EmitProbMatrix, InitStatus, LineNum, StatusDictionary)
    getLogOfPr()                                                    # 取对数概率

    # 输出参数估计结果
    print('参数估计结果:')
    print('初始状态分布')
    print(InitStatus)
    print('状态转移概率矩阵:')
    print(TransProbMatrix)
    # print('发射概率矩阵')
    # print(EmitProbMatrix)