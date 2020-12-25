'''
    头文件引用部分
'''
import re
import sys
import codecs
import numpy as np
import pandas as pd

'''
    参数设置
'''
StatusSet = ['B', 'M', 'E', 'S']    # 状态值集合
ObserveSet = set()                  # 观察值集合
TransProbMatrix = {}                # 状态转移概率矩阵   
EmitProbMatrix = {}                 # 发射概率矩阵
InitStatus = {}                     # 初始状态分布
LineNum = 0                         # 训练集句子的数量
StatusDictionary = {}               # 每个状态在训练集中出现的次数的字典


'''
    数据处理部分
'''

# 初始化参数
def initArg():
    for st1 in StatusSet:
        TransProbMatrix[st1] = {}                   # 初始化状态转移概率矩阵
        InitStatus[st1] = 0.0                       # 初始化初始状态分布
        EmitProbMatrix[st1] = {}                    # 初始化发射概率矩阵
        StatusDictionary[st1] = 0                   # 初始化字典
        for st2 in StatusSet:
            TransProbMatrix[st1][st2] = 0.0


# 从训练集每个词语中获取对应的状态标签
def getTag(word):
    Tag = []                                        # 每个词语对应的标签序列
    if len(word) == 1:                              # 词语长度为1，则为单字成词
        Tag = ['S']
    elif len(word) == 2:                            # 词语长度为2，则为一B一E
        Tag = ['B', 'E']
    else:                                           # 其他情况，词语开始为B，结束为E，中间为M
        Tag.append('B')
        Tag.extend(['M'] * (len(word)-2))
        Tag.append('E')
    return Tag


#将参数估计的概率取对数
def getLogOfPr():
    MIN = -3.14e+100                                # 定义对数概率的最小值
    for st in InitStatus:
        if InitStatus[st] == 0:                     # 如果概率为0，说明不可能有此初始状态，则为MIN
            InitStatus[st] = MIN
        else:                                       # 反之将概率取对数
            InitStatus[st] = np.log(InitStatus[st] / LineNum)
    for st1 in TransProbMatrix:
        for st2 in TransProbMatrix[st1]:            # 如果概率为0，说明不可能发生从st1到st2的转移，为MIN
            if TransProbMatrix[st1][st2] == 0.0:
                TransProbMatrix[st1][st2] = MIN
            else:                                   # 反之将概率取对数
                TransProbMatrix[st1][st2] = np.log(TransProbMatrix[st1][st2] / StatusDictionary[st1])
    for st in EmitProbMatrix:
        for word in EmitProbMatrix[st]:             # 如果概率为0，说明不可能有从st状态到word字的观测情况，为MIN
            if EmitProbMatrix[st][word] == 0.0:
                EmitProbMatrix[st][word] = MIN
            else:                                   # 反之将概率取对数
                EmitProbMatrix[st][word] = np.log(EmitProbMatrix[st][word] / StatusDictionary[st])


# 全局变量接口，通过两个接口实现多文件共享全局变量
def setArg(observeSet=None, transProbMatrix=None, emitProbMatrix=None, initStatus=None, lineNum=None, statusDictionary=None):
    global ObserveSet, TransProbMatrix, EmitProbMatrix, InitStatus, LineNum, StatusDictionary
    if(observeSet is not None):
        ObserveSet = observeSet
    if(transProbMatrix is not None):
        TransProbMatrix = transProbMatrix
    if(emitProbMatrix is not None):
        EmitProbMatrix = emitProbMatrix
    if(initStatus is not None):
        InitStatus = initStatus
    if(lineNum is not None):
        LineNum = lineNum
    if(statusDictionary is not None):
        statusDictionary = StatusDictionary

def getArg():
    return ObserveSet, TransProbMatrix, EmitProbMatrix, InitStatus, LineNum, StatusDictionary