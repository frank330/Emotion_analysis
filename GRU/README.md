实验流程：

（1）数据获取：使用微博情感分析数据集，包含neutral（无情绪）、happy（积极）、angry（愤怒）、sad（悲伤）、fear（恐惧）、surprise（惊奇）等6种情感，共3万条数据（https://smp2020ewect.github.io/）

（2）数据预处理：包括数据清洗和词向量

（3）搭建模型：pytorch框架搭建GRU神经网络

（4）模型训练：设置bach_size、epoch、GPU或CPU等 

（5）模型评估：使用混淆矩阵来评估
