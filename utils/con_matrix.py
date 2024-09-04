"""
混淆矩阵的计算
"""
from sklearn.metrics import confusion_matrix
import os
import time
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf,linewidth=2000)
import seaborn as sns

def compute_matrix(matrix_name, test_target, test_pred, top1,num_classes, epoch):
    matrix = confusion_matrix(test_target, test_pred)
    # 转化为百分比
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    # 写入混淆矩阵
    os.makedirs('matrix', exist_ok=True)
    log = 'matrix/' + str(matrix_name) + '.txt'
    f = open(log, 'a')
    f.write(str(matrix) + os.linesep)
    f.write(str(top1.avg) + os.linesep)
    f.write(str(epoch) + os.linesep)
    f.close()

    nan_matrix = np.full(matrix.shape, np.nan)
    # 只在原矩阵非零位置填入概率值
    nan_matrix[matrix != 0.] = matrix[matrix != 0.]
    # 可视化混淆矩阵
    labels = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential',
              'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
              'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt']
    '''labels = ['Airport','BareLand','BaseballField','Beach','Bridge','Center','Church','Commercial','DenseResidential',
              'Desert','Farmland','Forest','Industrial','Meadow','MediumResidential','Mountain','Park','Parking',
              'Playground','Pond','Port','RailwayStation','Resort','River','School','SparseResidential','Square',
              'Stadium','StorageTanks','Viaduct']'''
    cell_size = min(1.0 / len(labels), 0.5)
    font_size = max(min(int(100 * cell_size ** 2), 8), 4)
    plt.figure(figsize=(8,8))
    sns.heatmap(nan_matrix,fmt='.2f',cmap=plt.cm.Blues,annot=True,cbar=False,xticklabels=labels,yticklabels=labels,annot_kws={"size":font_size})
    plt.title('confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('/home/u2208183008/zyf/code/crossswin/ucmMatrix', transparent=True, dpi=1000)
    plt.show()

    
    '''plt.figure()
    plt.imshow(matrix, cmap=plt.cm.Blues)
    # 计算百分比
    matrix=matrix.astype('float')/matrix.sum(axis=1)[:,np.newaxis]
    # 遍历每个格子
    for i in range(num_classes):
        for j in range(num_classes):
            # 获取值
            value=matrix[i,j]
            # 显示概率
            text=plt.text(j,i,f'{value:.2%}',ha='center',va='center',color='black')
            # 显示数量
            #text=plt.text(j,i,f'{matrix[i,j]}',ha='center',va='center',color='black')

    plt.xticks(range(num_classes),labels,rotation=45)
    plt.yticks(range(num_classes),labels)
    plt.title('confusion matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.colorbar()
    # 保存可视化结果
    plt.savefig('matrix/{}01.png'.format(matrix_name))

    # 记录指标到文件
    with open('matrix/{}.txt'.format(matrix_name), 'a') as f:
        f.write(str(matrix) + '\n')
        f.write(str(top1.avg) + '\n')
        f.write(str(epoch) + '\n')'''
        
