import pandas as pd
from sklearn import svm
import joblib
from config import train_file, train_srcfolder, train_dstfolder, train_model
from tools import extract_numbers_all_to_folder,gencsv
from time import *
import warnings
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    begin_time=time()
    #提取该路径下的图片中所有数字，并输出到指定文件内
    extract_numbers_all_to_folder(train_srcfolder, train_dstfolder)
    #生成csv文件
    gencsv(train_dstfolder, train_file)
    #读取csv文件
    train = pd.read_csv(train_file)
    
    #第一列为真值
    trainlabel = train.values[:, 0]
    #第二列到最后为图片信息
    traindata = train.values[:, 1:]

    # SVM训练
    svc = svm.SVC(kernel='rbf', C=5)
    svc.fit(traindata, trainlabel)

    # 保存模型
    joblib.dump(svc, train_model)
    end_time=time()
    print(r'运行时间：',end_time-begin_time)