# 引用自己写的python文件
from train import *
'''
    函数实现部分
'''
# 处理未在训练集中出现的字的发射概率
def initEmitProbMatrix():
    # 所有的字都对应两种情况，要么是B，要么是E
    MIN = -3.14e+100
    global EmitProbMatrix
    EmitProbMatrix['B']['begin'] = 0
    EmitProbMatrix['B']['end'] = MIN
    EmitProbMatrix['S']['begin'] = MIN
    EmitProbMatrix['S']['end'] = MIN
    EmitProbMatrix['M']['begin'] = MIN
    EmitProbMatrix['M']['end'] = MIN
    EmitProbMatrix['E']['begin'] = MIN
    EmitProbMatrix['E']['end'] = 0
    setArg(emitProbMatrix=EmitProbMatrix)

# 维特比解码过程
def ViterBi(line):
    global LineNum, StatusSet, ObserveSet, TransProbMatrix
    global StatusDictionary, InitStatus, EmitProbMatrix
    Tab = [{}]  #动态规划表
    Path = {}
    MIN = -3.14e+100

    if line[0] not in EmitProbMatrix['B']:
        for st in StatusSet:
            if st == 'S':
                EmitProbMatrix[st][line[0]] = 0
            else:
                EmitProbMatrix[st][line[0]] = MIN

    for st in StatusSet:
        Tab[0][st] = InitStatus[st] + EmitProbMatrix[st][line[0]]
        Path[st] = [st]

    for t in range(1,len(line)):
        Tab.append({})
        NewPath = {}

        for st1 in StatusSet:
            Items = []
            for st2 in StatusSet:
                if line[t] not in EmitProbMatrix[st1]:                      # 如果这个字不在观测集合内，就按照非B即E的情况处理
                    if line[t-1] not in EmitProbMatrix[st1]:
                        Prob = Tab[t-1][st2] + TransProbMatrix[st2][st1] + EmitProbMatrix[st1]['end']
                    else:
                        Prob = Tab[t-1][st2] + TransProbMatrix[st2][st1] + EmitProbMatrix[st1]['begin']
                else:
                    # 计算的是对数概率，所以概率相乘即对数概率相加
                    Prob = Tab[t-1][st2] + TransProbMatrix[st2][st1] + EmitProbMatrix[st1][line[t]]
                Items.append((Prob,st2))

            Best = max(Items)  
            Tab[t][st1] = Best[0]
            NewPath[st1] = Path[Best[1]] + [st1]
        Path = NewPath

    Prob, State = max([(Tab[len(line) - 1][state], state) for state in StatusSet])
    return Path[State]


# 根据状态序列进行分词
def tagSeg(line, tag):
    WordList = []
    Start = -1
    Started = False

    if len(tag) != len(line):
        return None
    if len(tag) == 1:                                           # 如果长度为一，说明单字成词
        WordList.append(line[0])   
    else:
        if tag[-1] == 'B' or tag[-1] == 'M':                    # 去除一个词的最后一个字为B、M的情况(这些情况是不应该存在的)
            if tag[-2] == 'B' or tag[-2] == 'M':
                tag[-1] = 'S'
            else:
                tag[-1] = 'E'

        for t in range(len(tag)):                               # 对于一个普通的词而言，首尾是BE，其他都是M
            if tag[t] == 'S':
                if Started:
                    Started = False
                    WordList.append(line[Start:t])
                WordList.append(line[t])
            elif tag[t] == 'B':
                if Started:
                    WordList.append(line[Start:t])
                Start = t
                Started = True
            elif tag[t] == 'E':
                Started = False
                word = line[Start:t+1]
                WordList.append(word)
            elif tag[t] == 'M':
                continue
    return WordList

# 根据状态序列进行分词
def tagSeg(line, tag):
    WordList = []
    Start = -1
    Started = False

    if len(tag) != len(line):
        return None
    if len(tag) == 1:                                           # 如果长度为一，说明单字成词
        WordList.append(line[0])   
    else:
        if tag[-1] == 'B' or tag[-1] == 'M':                    # 去除一个词的最后一个字为B、M的情况(这些情况是不应该存在的)
            if tag[-2] == 'B' or tag[-2] == 'M':
                tag[-1] = 'S'
            else:
                tag[-1] = 'E'

        for t in range(len(tag)):                               # 对于一个普通的词而言，首尾是BE，其他都是M
            if tag[t] == 'S':
                if Started:
                    Started = False
                    WordList.append(line[Start:t])
                WordList.append(line[t])
            elif tag[t] == 'B':
                if Started:
                    WordList.append(line[Start:t])
                Start = t
                Started = True
            elif tag[t] == 'E':
                Started = False
                word = line[Start:t+1]
                WordList.append(word)
            elif tag[t] == 'M':
                continue
    return WordList


# 测试集测试
def test(filename): 
    TestSet = open(filename, encoding='gb18030') 
    output = ''
    for line in TestSet:
        line = line.strip()
        tag = ViterBi(line)
        # print(tag)
        seg = tagSeg(line, tag)
        # print(seg)
        list = ''
        for i in range(len(seg)):
            list = list + seg[i] + ' '
        # print(list)
        output = output + list + '\n'

    outputfile = open('output.txt', mode='w', encoding='gb18030')
    outputfile.write(output)

# 评估HMM的效果,计算F1-score
def F1OfHMM(filename):
    global StatusSet
    output = ''
    TestSet = open(filename, encoding='gb18030')
    TotalNUM = 0
    F1 = {}     # F1集合
    TP = {}     # 真阳情况对应的字典    预测为正，实际为正
    FP = {}     # 假阳情况对应的字典    预测为正，实际为负
    FN = {}     # 真阴情况对应的字典    预测为负，实际为正
    Precision = {}      # 精确率对应的字典
    Recall = {}         # 召回率对应的字典
    F1['B'] = F1['M'] = F1['S'] = F1['E'] = 0
    TP['B'] = TP['M'] = TP['S'] = TP['E'] = 0
    FP['B'] = FP['M'] = FP['S'] = FP['E'] = 0
    FN['B'] = FN['M'] = FN['S'] = FN['E'] = 0
    Precision['B'] = Precision['M'] = Precision['S'] = Precision['E'] = 0
    Recall['B'] = Recall['M'] = Recall['S'] = Recall['E'] = 0
    for line in TestSet:
        line = line.strip()
        line = line.split(' ')
        LineState = []                  # 这句话真实的状态序列
        Line = []
        for word in line:
            if(len(word)==0):
                continue
            for j in range(len(word)):
                Line.append(word[j])
            LineState.extend(getTag(word))
        if(len(LineState)==0):continue

        PreLineState = ViterBi(Line)    # 预测出来的这句话的状态序列

        # print(Line,len(Line))
        # print(LineState,len(LineState))
        # print(PreLineState,len(PreLineState))
        for i in range(len(LineState)):
            if(LineState[i] == PreLineState[i]):
                TP[LineState[i]] += 1
            else:
                FP[PreLineState[i]] += 1
                FN[LineState[i]] += 1
        TotalNUM += 1
        print(TotalNUM)
        # for i in range(len(line)):
        #     output = output + line[i]
        # output = output + '\n'
        # for i in range(len(PreLineState)):
        #     output = output + PreLineState[i]
        # output = output + '\n'
    
    # 计算精确率和召回率
    for state in StatusSet:
        Precision[state] = float(TP[state] / (TP[state] + FP[state]))
        Recall[state] = float(TP[state] / (TP[state] + FN[state]))
        F1[state] = 2 * (Precision[state] * Recall[state]) / (Precision[state] + Recall[state])
    
    # 输出每个状态的F1的平均值
    F1_score = 0
    for state in StatusSet:
        F1_score += F1[state]
    F1_score = F1_score / 4
    print('每个状态的精确率为：', Precision)
    print('每个状态的召回率为：', Recall)
    print('最终的F1得分为：', F1_score)
    return F1_score
    # outputfile = open('output1.txt', mode='w', encoding='gb18030')
    # outputfile.write(output)

if __name__ == '__main__':
    TrainFile = 'msr_training.txt'              # 训练集文件
    TestFile = 'msr_test_gold.txt'              # 测试集文件
    initArg()                                   # 初始化参数
    TrainArg(TrainFile)                         # 训练参数
    initEmitProbMatrix()
    F1 = F1OfHMM(TrainFile)
    #print(F1)
    test(TestFile)
