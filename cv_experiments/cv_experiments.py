from math import pi
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def toGray(truecolorimg):
    """
    真彩图转灰度图
        参数：
        - `truecolorimg`:真彩图
    
        返回值：
        - `gray`:灰度图(二维np矩阵)
    """
    row,column,depth=truecolorimg.shape
    #新建灰度图（二维数组）
    gray=np.zeros((row,column),dtype=np.uint8)
    #给灰度图赋值
    for i in range(row):
            for j in range(column):
                #opencv读取顺序是GBR
                r=truecolorimg[i][j][2]
                g=truecolorimg[i][j][0]
                b=truecolorimg[i][j][1]
                gray[i][j]= (r * 299 + g * 587 + b * 114) // 1000
    #返回灰度图
    return gray

def hist(gray):
    """
    根据给定灰度图(gray)绘制灰度直方图
        参数：
        - `gray`:灰度图

        返回值：
        - `NULL`
    """
    #将灰度图化为一维数组
    gray_onedimension=gray.flatten()

    #使用plt绘制直方图
    plt.figure("灰度直方图")
    plt.hist(gray_onedimension, 255, [0, 255])
    plt.show()

def otsu(gray):
    """
    大津算法
        参数：
        - `gray`:灰度图

        返回值：
        - `threshold`:灰度图的阈值(int)
    """
    #获取灰度图的行列数
    row,column=gray.shape

    ##获取灰度值分布数组
    #新建一维矩阵(hist)
    hist=np.zeros((256,), dtype=int, order='c')
    for i in range(row):
        for j in range(column):
            #计算灰度值分布
            hist[gray[i][j]]+=1

    ##获取阈值
    #初始化阈值(threshold)与方差(dx)
    threshold=dx=0

    #从灰度值0开始到255计算阈值
    for i in range(256):

        #   w0:前景占图像比例
        #   u0:前景平均灰度
        #   w1:背景景占图像比例
        #   u1:背景平均灰度
        w0=u0=w1=u1=0

        #获取前景信息
        u_front=front_size=0
        for j in range(i):
            #获取前景总像素点
            front_size+=hist[j]
            #获取前景总像灰度
            u_front+=hist[j]*j
        #计算前景占图像比例
        w0=front_size/(row*column)
        #计算前景平均灰度
        if(front_size!=0):
            u0=u_front/front_size

        #获取背景信息
        u_back=back_size=0
        for j in range(i,256):
            #获取背景总像素点
            back_size+=hist[j]
            #获取背景总像灰度
            u_back+=hist[j]*j
        #计算背景占图像比例
        w1=back_size/(row*column)
        #计算背景平均灰度
        if(back_size!=0):
            u1=u_back/back_size
        
        #计算前景和背景图像的方差
        g=w0*w1*(u0-u1)*(u0-u1)

        #获取阈值
        if(dx<g):
            dx=g
            threshold=i
    return threshold

def toBinary(gray,threshold):
    """
    灰度图转二值图
        参数：
        - `gray`:灰度图
        - `threshold`:灰度图的阈值

        返回值：
        - `binary`:二值图(二维np矩阵)
    """
    #获取灰度图的行列数
    row,column=gray.shape

    #新建二值图（二维数组）
    binary=np.zeros((row,column),dtype=np.uint8)

    #通过阈值转为二值图
    for i in range(row):
        for j in range(column):
            #大于阈值的等于255，小于阈值的等于0
            if(gray[i][j]>=threshold):
                binary[i][j]=255
            else:
                binary[i][j]=0
    return binary

def GaussFilter(ksize):
    """
    创建高斯滤波器
        参数：
        - `ksize`:窗口大小

        返回值：
        - `Gaussfilter`:高斯滤波器(二维np矩阵)
    """
    Gaussfilter=np.zeros((ksize,ksize))
    #初始化标准差sigma：σ=0.3×((ksize−1)×0.5−1)+0.8
    sigma=0.3*((ksize-1)*0.5-1)+0.8
    #初始化高斯滤波器
    for i in range(ksize):
        for j in range(ksize):
            Gaussfilter[i][j]=(1/(2*np.pi*sigma*sigma))*np.exp(-((i-ksize//2)*(i-ksize//2)+(j-ksize//2)*(j-ksize//2))/(2*sigma*sigma))
    return Gaussfilter

def GaussBlur(gray):
    """
    高斯滤波
        参数：
        - `gray`:灰度图

        返回值：
        - `Gaussblur`:灰度图(二维np矩阵)
    """
    #获取灰度图的行列数
    row,column=gray.shape

    #初始化高斯滤波器，窗口大小为5
    gaussfilter=GaussFilter(5)

    #扩充原图
    exGray=np.full((row+4,column+4),127,dtype=np.uint8)
    for i in range(row):
        for j in range(column):
            exGray[i+2][j+2]=gray[i][j]

    Gaussblur=np.zeros((row,column),dtype=np.uint8)

    #与滤波器卷积
    for i in range(0,row):
        for j in range(0,column):
            Gaussblur[i][j]=(exGray[i:i+5,j:j+5]*gaussfilter).sum().astype(np.uint8)
    return Gaussblur

def grad_X(gray):
    """
    计算x方向梯度
        参数：
        - `gray`:灰度图

        返回值：
        - `grad_x`:x方向梯度(二维np矩阵)
    """
    #获取灰度图的行列数
    row,column=gray.shape

    #定义算子
    G_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    
    #扩充原图
    exGray=np.full((row+2,column+2),127,dtype=np.uint8)
    for i in range(row):
        for j in range(column):
            exGray[i+1][j+1]=gray[i][j]

    #新建梯度矩阵
    grad_x=np.zeros((row,column))
    for i in range(row):
        for j in range(column):
            #与算子卷积
            grad_x[i][j]=(exGray[i:i+3,j:j+3]*G_x).sum()

    return grad_x

def grad_Y(gray):
    """
    计算y方向梯度
        参数：
        - `gray`:灰度图

        返回值：
        - `grad_y`:y方向梯度(二维np矩阵)
    """
    #获取灰度图的行列数
    row,column=gray.shape

    #定义算子
    G_y=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    
    #扩充原图
    exGray=np.full((row+2,column+2),127,dtype=np.uint8)
    for i in range(row):
        for j in range(column):
            exGray[i+1][j+1]=gray[i][j]

    #新建梯度矩阵
    grad_y=np.zeros((row,column))
    for i in range(row):
        for j in range(column):
            #与算子卷积
            grad_y[i][j]=(exGray[i:i+3,j:j+3]*G_y).sum()

    return grad_y    

def grad(grad_x,grad_y):
    """
    计算梯度幅值
        参数：
        - `grad_x`:水平梯度矩阵
        - `grad_y`:竖直梯度矩阵

        返回值：
        - `grad`:梯度幅值(二维np矩阵)
    """

    grad=np.zeros((grad_x.shape[0],grad_x.shape[1]))
    for i in range(grad_x.shape[0]):
        for j in range(grad_x.shape[1]):
            #计算每个像素的梯度幅值
            grad[i][j]=np.sqrt(grad_x[i][j]**2+grad_y[i][j]**2)

    return grad

def angle(grad_x,grad_y):
    """
    计算方向
        参数：
        - `grad_x`:水平梯度矩阵
        - `grad_y`:竖直梯度矩阵

        返回值：
        - `angle`:梯度方向(二维np矩阵)
    """
    #获取灰度图的行列数
    row=grad_x.shape[0]
    column=grad_y.shape[1]

    #新建梯度矩阵
    angle=np.zeros((row,column),dtype=float)
    for i in range(row):
        for j in range(column):
            #加π/2是为了方便计算，将角度从(-π/2,π/2)拉到(0,π)
            #加0.0000001为了避免分母为0的warning
            angle[i][j]=np.arctan(grad_y[i][j]/(0.0000001+grad_x[i][j]))+pi/2
    
    return angle

def nms(grad,angle):
    """
    非极大值抑制
        参数：
        - `grad`:梯度幅值矩阵
        - `angle`:角度矩阵

        返回值：
        - `nms`:灰度图(二维np矩阵)
    """
    #获取行列数
    row,column=grad.shape

    nms=np.zeros((row,column),dtype=np.uint8)
    #失败的算法，比对点选取有问题
    # for i in range(1,row-1):
    #     for j in range(1,column-1):
    #         if(0<=angle[i][j] and angle[i][j]<(pi/4)):
    #             if(grad[i][j]>grad[i][j-1] and grad[i][j]>grad[i][j+1]):
    #                 nms[i][j]=grad[i][j]
    #         elif((pi/4)<=angle[i][j] and angle[i][j]<(pi/2)):
    #             if(grad[i][j]>grad[i-1][j-1] and grad[i+1][j+1]>grad[i][j+1]):
    #                 nms[i][j]=grad[i][j]
    #         elif((pi/2)<=angle[i][j] and angle[i][j]<((3*pi)/4)):
    #             if(grad[i][j]>grad[i-1][j] and grad[i][j]>grad[i+1][j]):
    #                 nms[i][j]=grad[i][j]
    #         elif(((3*pi)/4)<=angle[i][j] and angle[i][j]<pi):
    #             if(grad[i][j]>grad[i-1][j+1] and grad[i][j]>grad[i+1][j-1]):
    #                 nms[i][j]=grad[i][j]

    #成功的算法，改进了比对点的选取
    for i in range(1,row-1):
        for j in range(1,column-1):
            #在四个方向上选取比对点
            #因为加了π/2，故选取对比点稍有区别，这也是上面的错误原因
            #若造成困扰，请自行将角度减去π/2
            if(0<=angle[i][j] and angle[i][j]<(pi/4)):
                #在(0,π/4)内，选上下两点对比
                if(grad[i][j]>grad[i-1][j] and grad[i][j]>grad[i+1][j]):
                    nms[i][j]=grad[i][j]
            elif((pi/4)<=angle[i][j] and angle[i][j]<(pi/2)):
                #在(π/4,π/2)内，选左下和右上两点对比
                if(grad[i][j]>grad[i-1][j+1] and grad[i+1][j-1]>grad[i][j+1]):
                    nms[i][j]=grad[i][j]
            elif((pi/2)<=angle[i][j] and angle[i][j]<((3*pi)/4)):
                #在(π/2,3π/4)内，选左右两点对比
                if(grad[i][j]>grad[i][j-1] and grad[i][j]>grad[i][j+1]):
                    nms[i][j]=grad[i][j]
            elif(((3*pi)/4)<=angle[i][j] and angle[i][j]<pi):
                #在(3π/4,π)内，选左上和右下两点对比
                if(grad[i][j]>grad[i+1][j+1] and grad[i][j]>grad[i-1][j-1]):
                    nms[i][j]=grad[i][j]

    return nms

def doubleThreshold(nms,min,max):
    """
    双阈值检测+连通
        参数：
        - `nms`:非极大值抑制后的图
        - `min`:最小阈值
        - `max`:最大阈值

        返回值：
        - `doubleThreshold`:灰度图(二维np矩阵)
    """
    #获取行列数
    row,column=nms.shape

    doubleThreshold=np.zeros((row,column),dtype=np.uint8)
    #因为不想扩充矩阵，故从1开始到row-1
    for i in range(1,row-1):
        for j in range(1,column-1):
            #小于最小阈值舍弃
            if(nms[i][j]<min):
                doubleThreshold[i][j]=0
            #大于最大阈值保留
            elif(nms[i][j]>max):
                doubleThreshold[i][j]=255
            #若介于双阈值之间，可连接到高阈值的点保留(连通)
            #原理:若以[i,j]为中心的3x3矩阵中的最大值大于阈值则保留
            elif(nms[i-1:i+2,j-1:j+2].max()>max):
                doubleThreshold[i][j]=255
    
    return doubleThreshold

def Canny(gray,min,max):
    """
    Canny算法
        参数：
        - `gray`:灰度图
        - `min`:最小阈值
        - `max`:最大阈值

        返回值：
        - `canny`:边缘图(二维np矩阵)
    """
    #step1.消除噪声(高斯滤波)
    gaussBlur=GaussBlur(gray)

    #step2.计算梯度幅值与方向角
    grad_x=grad_X(gaussBlur) #grad_x:水平梯度
    grad_y=grad_Y(gaussBlur) #grad_y:竖直梯度
    grad_re=grad(grad_x,grad_y) #grad:梯度幅值
    angle_re=angle(grad_x,grad_y) #angle:方向角

    # cv2.imshow('grad',grad_re)
    # cv2.imshow('angle',angle_re)

    #step3.非极大值抑制
    nms_re=nms(grad_re,angle_re)

    #step4.滞后阈值
    canny=doubleThreshold(nms_re,min,max)
    return canny


def hproject(binary, forecolor=255):
    """
    获取水平投影
        参数：
        - `binary`:二值图
        - `forecolor`:前景色 默认255
    
        返回值：
        - `stats`:水平投影信息(一维np数组),元素值表示对应行黑色像素个数
    """
    row, column = binary.shape
    stats = np.zeros((row,), dtype=int, order='c')
    for i in range(row):
        for j in range(column):
            if forecolor != 255:
                stats[i] += 1 if binary[i, j] else 0
            else:
                stats[i] += 0 if binary[i, j] else 1
    return stats

def vproject(binary, forecolor=255):
    """
    获取垂直投影
        参数：
        - `binary`:二值图
        - `forecolor`:前景色 默认255

        返回值：
        - `stats`:水平投影信息(一维np数组),元素值表示对应行黑色像素个数
    """
    row, column = binary.shape
    stats = np.zeros((column,), dtype=int, order='c')
    for j in range(column):
        for i in range(row):
            if forecolor != 255:
                stats[j] += 1 if binary[i, j] else 0
            else:
                stats[j] += 0 if binary[i, j] else 1
    return stats

def findChar(stats):
    """
    分割，用于找到字符行
        参数：
        - `stats`:投影信息(一维np数组),元素值表示对应行黑色像素个数

        返回值：
        - `arr`:分割位置(list)
    """
    arr = []
    sign = 1
    for i in range(len(stats)-1):
        if stats[i] == 0 and sign == 1:
            continue
        elif sign == 1:
            arr.append(i)
            sign *= -1
        elif stats[i] == 0:
            arr.append(i)
            sign *= -1
    return arr

def find_Gravity_core(binary):
    """
    找重心
        参数：
        - `binary`:二值图

        返回值：
        - `point`:重心坐标
    """
    height,width = binary.shape
    # 垂直方向投影
    x = [0]*height
    # 水平方向投影
    y = [0]*width
    # 像素点总数
    count = 0
    point = []
    for i in range(width):
        for j in range(height):
            if binary[j, i] == 0:
                y[i] += 1
                x[j] += 1
                count += 1
    county = count // 2
    for i in range(width):
        county -= y[i]
        if county <= 0:
            point.append(i)
            break
    countx = count // 2
    # 求投影中心 投影中心即重心坐标
    for i in range(height):
        countx -= x[i]
        if countx <= 0:
            point.append(i)
            break
    return point

def neighbor8(binary):
    """
    通过8邻域寻找特征
        参数：
        - `binary`:二值图

        返回值：
            面积 重心 轮廓周长 外接矩形[左上角坐标(y,x),宽,高]
    """
    row,column=binary.shape
    #使用标记数组记录是否遍历过
    sign = np.zeros((row+2,column+2))
    #队列用于广度优先遍历
    q = []
    #周长，面积，重心
    perimeter = 0
    area=1
    gravity_core=0
    #定义四个变量用于记录最外围坐标
    min_i,min_j=row+2,column+2
    max_i,max_j=-1,-1

    #寻找第一个黑色像素
    for i in range(row):
        for j in range(column):
            if binary[i][j]==0:
                #遍历过的像素置1
                sign[i+1,j+1] = 1
                #第一个黑色入队
                q.append((i, j))
                break
        #找到第一个就跳出循环
        if len(q)!=0:
            break
    while(len(q)!=0):
        #获取坐标
        coordinate=q.pop(0)
        #分别用x，y表示
        x = coordinate[0]
        y = coordinate[1]
        #每次循环更新外围坐标
        if max_i < x: max_i = x
        if max_j < y: max_j = y
        if min_i > x: min_i = x
        if min_j > y: min_j = y
        # 遍历8领域
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                # 防止越界并保证未遍历过
                if  x+i < row and x+i >= 0 and y+j >= 0 and y+j < column and sign[x+i+1, y+j+1] == 0 and binary[x+i, y+j] == 0:
                    # 将8领域内值为0的点的坐标加入队列
                    q.append((x+i, y+j))
                    sign[x+i+1, y+j+1] = 1
                    # 求面积(像素点个数就是面积)
                    area += 1
                # 与字符像素相接 不是图像的便是轮廓
                elif sign[x+i+1, y+j+1] == 0:
                    perimeter += 1
                    sign[x+i+1, y+j+1] = 1
    gravity_core=find_Gravity_core(binary[min_i:max_i,min_j:max_j])
    return area, gravity_core , perimeter, [min_j, min_i, max_j - min_j, max_i - min_i]

def exp4(img,binary):
    """
    在彩图上画最小外接矩形，并输出特征信息
        参数：
        - `img`:彩图
        - `binary`:二值图

        返回值：
            NULL
    """
    #获取水平投影
    h_state = hproject(binary)
    h_shape = findChar(h_state)
    #获取垂直投影
    v_state = vproject(binary[h_shape[2]:h_shape[3]])
    v_shape = findChar(v_state)
    #找到字符行
    h_shape=[h_shape[2],h_shape[3]]

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    for i in range(0,len(v_shape), 2):
        area, gravity_core,perimeter, ar = neighbor8(binary[h_shape[0]:h_shape[1],v_shape[i]:v_shape[i+1]])
        #计算重心位置
        g_core=[v_shape[i]+gravity_core[0],v_shape[i]+gravity_core[1]]
        print(r'面积：',area,r'重心：',g_core,r'周长：',perimeter)
        plt.gca().add_patch(patches.Rectangle((v_shape[i]-2, h_shape[0]-2), v_shape[i+1] - v_shape[i] + 4, h_shape[1] - h_shape[0] + 4, linewidth=2, edgecolor='r', facecolor='none'))
    plt.show()

if __name__ == "__main__":
    ### 实验1 编程环境与图像读写
    ##  (1)读图
    img=cv2.imread(r'cv_experiments\test.jpg',1)
    # cv2.imshow('origin',img)

    #  (2)转灰度图
    gray=toGray(img)
    # cv2.imshow('gary',gray)

    ##  (3)获取灰度图像中每一个像素点的值并输出到gray.txt中
    # np.savetxt(r'cv_experiments\gray.txt',gray,fmt="%d",delimiter=",")


    ### 实验2 图像分割
    ##  (1)画出ISBN号图像的灰度直方图
    # hist(gray)

    ##  (2)确定图像分割的阈值
    #通过大津算法获取该图阈值
    threshold=otsu(gray)

    ##  (3)将图像转变成只有字符和背景的二值图像
    Binary=toBinary(gray,threshold)
    # cv2.imshow('binary',Binary)

    ### 实验3 边缘检测
    ##  (1)对图片进行滤波处理(高斯滤波)
    # gaussBlur=GaussBlur(gray)
    # cv2.imshow('gaussBlur',gaussBlur)

    ##  (2)边缘检测(Canny算法)
    #   原理：图形边缘一般是梯度大的地方
    # canny=Canny(gray,90,100)
    # cv2.imshow('canny',canny)
    ### 实验4 图像特征提取
    exp4(img,Binary)


    #延长窗口时间
    cv2.waitKey(0)