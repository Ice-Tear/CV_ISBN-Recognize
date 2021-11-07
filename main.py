import os
import shutil
import cv2
import pandas as pd
import numpy as np
import joblib
from config import train_model, test_srcfolder, test_dstfolder,test_error_digits_srcfolder,test_error_srcfolder
from tools import extract_numbers_to_folder,gencsv
from time import *

if __name__ == '__main__':
    begin_time=time()
    svc = joblib.load(train_model)
    pic_right = 0
    pic_count = 100
    char_right = 0
    char_count = 0
    shutil.rmtree(test_dstfolder)
    os.makedirs(test_dstfolder)
    count = 0
    for dirpath, dirnames, filenames in os.walk(test_srcfolder):
        for filename in filenames:
            count += 1
            print(count, ':', filename)

            imgloc = os.path.join(dirpath, filename)
            subfolder = os.path.join(test_dstfolder, filename)
            os.makedirs(subfolder)
            extract_numbers_to_folder(imgloc, subfolder)
            testfile = os.path.join(subfolder, 'testsrc.csv')
            names = os.listdir(subfolder)
            if len(names) == 0:
                #获取错误图片信息
                img=cv2.imread(imgloc)
                cv2.imwrite(r'.\resource\test_resource\test_Error\origin\{}'.format(filename),img)
                error_src=os.path.join(test_error_digits_srcfolder, filename)
                os.makedirs(error_src)
                extract_numbers_to_folder(imgloc, error_src)
                print(count, ':', filename)
                print(r'error!')
            if len(names) > 0:
                gencsv(subfolder, testfile)
                test = pd.read_csv(testfile)
                testdata = test.values[:, 1:]
                testlabel = test.values[:, 0]
                prediction = svc.predict(testdata)
                #打印真值
                print(testlabel)
                #打印预测值
                print(prediction)
                local_char_right = 0
                num = len(testlabel)
                for i in range(num):
                    if prediction[i] == testlabel[i]:
                        local_char_right += 1
                char_right += local_char_right
                char_count += num
                if local_char_right == num:
                    pic_right += 1
                else:
                    #获取错误图片信息
                    img=cv2.imread(imgloc)
                    cv2.imwrite(r'.\resource\test_resource\test_Error\origin\photo_{}.jpg'.format(count),img)
                    error_src=os.path.join(test_error_digits_srcfolder, filename)
                    os.makedirs(error_src)
                    extract_numbers_to_folder(imgloc, error_src)
                    print(count, ':', filename)
                    print(testlabel)
                    print(prediction)
                    print(r'error!')
    
    print('图片识别准确率: {:.2f}%'.format(pic_right / pic_count*100))
    if char_count==0 :
        print('字符识别准确率: 0.00%')
    else :
        print('字符识别准确率: {:.2f}%'.format(char_right / char_count*100))
    end_time=time()
    print(r'运行时间：',end_time-begin_time)