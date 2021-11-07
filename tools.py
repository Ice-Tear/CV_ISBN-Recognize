import cv2
import numpy as np
import os
import config
from math import *
from scipy.stats import mode
from skimage import exposure,img_as_float,img_as_ubyte

#图像校正
class ImgCorrect():
    def __init__(self, img):
        self.img = img
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    def img_lines(self):
        ret, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
        binary = cv2.dilate(binary, kernel)  # 膨胀
        edges = cv2.Canny(binary, 50, 200)
        self.lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)
        if self.lines is None:
            return None

        lines1 = self.lines[:, 0, :]  # 提取为二维
        imglines = self.img.copy()
        for x1, y1, x2, y2 in lines1[:]:
            cv2.line(imglines, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return imglines

    def search_lines(self):
        lines = self.lines[:, 0, :]  # 提取为二维
        number_inexistence_k = 0
        sum_positive_k45 = 0
        number_positive_k45 = 0
        sum_positive_k90 = 0
        number_positive_k90 = 0
        sum_negative_k45 = 0
        number_negative_k45 = 0
        sum_negative_k90 = 0
        number_negative_k90 = 0
        number_zero_k = 0
        for x in lines:
            if x[2] == x[0]:
                number_inexistence_k += 1
                continue
            if 0 < degrees(atan((x[3] - x[1]) / (x[2] - x[0]))) < 45:
                number_positive_k45 += 1
                sum_positive_k45 += degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
            if 45 <= degrees(atan((x[3] - x[1]) / (x[2] - x[0]))) < 90:
                number_positive_k90 += 1
                sum_positive_k90 += degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
            if -45 < degrees(atan((x[3] - x[1]) / (x[2] - x[0]))) < 0:
                number_negative_k45 += 1
                sum_negative_k45 += degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
            if -90 < degrees(atan((x[3] - x[1]) / (x[2] - x[0]))) <= -45:
                number_negative_k90 += 1
                sum_negative_k90 += degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
            if x[3] == x[1]:
                number_zero_k += 1

        max_number = max(number_inexistence_k, number_positive_k45, number_positive_k90, number_negative_k45,
                         number_negative_k90, number_zero_k)
        if max_number == number_inexistence_k:
            return 90
        if max_number == number_positive_k45:
            return sum_positive_k45 / number_positive_k45
        if max_number == number_positive_k90:
            return sum_positive_k90 / number_positive_k90
        if max_number == number_negative_k45:
            return sum_negative_k45 / number_negative_k45
        if max_number == number_negative_k90:
            return sum_negative_k90 / number_negative_k90
        if max_number == number_zero_k:
            return 0

    def rotate_image(self, degree):
        """
        正角 逆时针旋转
        :param degree:
        :return:
        """
        if -45 <= degree <= 0:
            degree = degree  # #负角度 顺时针
        if -90 <= degree < -45:
            degree = 90 + degree  # 正角度 逆时针
        if 0 < degree <= 45:
            degree = degree  # 正角度 逆时针
        if 45 < degree <= 90:
            degree = degree - 90  # 负角度 顺时针
        # 获取旋转后4角的填充色
        filled_color = -1
        if filled_color == -1:
            filled_color = mode([self.img[0, 0], self.img[0, -1],
                                 self.img[-1, 0], self.img[-1, -1]]).mode[0]
        if np.array(filled_color).shape[0] == 2:
            if isinstance(filled_color, int):
                filled_color = (filled_color, filled_color, filled_color)
        else:
            filled_color = tuple([int(i) for i in filled_color])

        height, width = self.img.shape[:2]
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))  # 这个公式参考之前内容
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)  # 逆时针旋转 degree

        matRotation[0, 2] += (widthNew - width) / 2  # 因为旋转之后,坐标系原点是新图像的左上角,所以需要根据原图做转化
        matRotation[1, 2] += (heightNew - height) / 2

        imgRotation = cv2.warpAffine(self.img, matRotation, (widthNew, heightNew), borderValue=filled_color)

        return imgRotation

def correctJPG(img):
    imgcorrect = ImgCorrect(img)
    lines_img = imgcorrect.img_lines()
    if lines_img is None:
        return imgcorrect.rotate_image(0)
    else:
        degree = imgcorrect.search_lines()
        return imgcorrect.rotate_image(degree)

def correct(binary):
    """
    基于最小外接矩形的图像校正
        参数：
        - `binary`:二值图
    
        返回值：
        - `binary`:校正后的二值图
    """
    coords = np.column_stack(np.where(binary == 0))
    #获取旋转角度
    angle = cv2.minAreaRect(coords)[-1]
    if abs(angle) > 70:
        height, width = binary.shape
        center = (width // 2, height // 2)
        mat = cv2.getRotationMatrix2D(center, 90 - angle, 1.0)
        binary = cv2.warpAffine(binary, mat, (width, height),borderValue=255)
        #注：仿射变换后不是二值图，故需要把灰度图转化成二值图
        binary=cv2.threshold(binary, 0, 255, cv2.THRESH_OTSU)[1]
    return binary

def correctImage(img):
    """
    图像校正
        参数：
        - `img`:图像
    
        返回值：
        - `image`:校正后的图像
    """
    #若图像是真彩图
    if(len(img.shape)==3):
        return correctJPG(img)
    #若图像不是真彩图
    else:
        return correct(img)

def h_Projection(binary):
    """
    获取水平投影
        参数：
        - `binary`:二值图
    
        返回值：
        - `stats`:水平投影信息(一维np数组),元素值表示对应行黑色像素个数
    """
    #获取行列数
    row, column = binary.shape
    #新建一维np数组记录投影信息
    stats = np.zeros((row,), dtype=int, order='c')
    #循环赋值
    for i in range(row):
        for j in range(column):
            #若该像素为黑，则该行值+1
            if binary[i][j]==0:
                stats[i] += 1

    #测试用
    # h_Binary=binary.copy()
    # for i in range(row):
    #     for j in range(column):
    #         if h_Binary[i][j]==0:
    #             stats[i] += 1
    #             h_Binary[i][j]=255
    # for i in range(row):
    #     for j in range(stats[i]):
    #         h_Binary[i][j]=0
    #显示水平投影
    # cv2.imshow('h_Projection',h_Binary)
    # print(stats)
    return stats

def v_Projection(binary):
    """
    获取垂直投影
        参数：
        - `binary`:二值图
    
        返回值：
        - `stats`:垂直投影信息(一维np数组),元素值表示对应行黑色像素个数
    """
    #获取行列数
    row, column = binary.shape
    #新建一维np数组记录投影信息
    stats = np.zeros((column,), dtype=int, order='c')
    #循环赋值
    for j in range(column):
        for i in range(row):
            #若该像素为黑，则该列值+1
            if binary[i][j]==0:
                stats[j] += 1


    #测试用
    # v_Binary=binary.copy()
    # for j in range(column):
    #     for i in range(row):
    #         if v_Binary[i][j]==0:
    #             stats[j] += 1
    #             v_Binary[i][j]=255
    # for j in range(column):
    #     for i in range(row-stats[j],row):
    #         v_Binary[i][j]=0
    #显示垂直投影
    # cv2.imshow('v_Projection',v_Binary)
    # print(stats)
    return stats

def projectionBlur(stats,length=0,height=0):
    """
    将传入投影数组中的噪点去除
        参数：
        - `stats`:一维投影数组
        - `height`:高度阈值(默认值为0)
        - `length`:长度阈值(默认值为0)
        返回值：
        - `stats`:去噪后的投影(一维np数组)
    """
    #去除过小值(黑色像素少的算成噪点)
    for i in range(stats.size):
        #小于高度阈值
        if stats[i]<height:
            stats[i]=0
    #去除长度短的(如直线之类的)
    start_mark=False
    end_mark=False
    start,end=0,0
    for i in range(stats.size):
        #若无起始点，遇到第一个黑色当做起始点
        if stats[i]>0 and not start_mark:
            start=i
            start_mark=True
        #若有起始点，该点是白色，则结束
        if stats[i]==0 and start_mark:
            end=i
            end_mark=True
        if start_mark==True and end_mark==True:
            #长度小于长度阈值，全部变白
            if end-start<length:
                stats[start:end]=[0]*(end-start)
            start_mark=False
            end_mark=False
    return stats

def ClearBackGround(binary):
    """
    利用四角填充白色的方法清除二值图的黑色背景
        参数：
        - `binary`:二值图
    
        返回值：
        - `binary`:二值图(二维np矩阵)
    """
    #获取图片宽高
    height, width = binary.shape 
    #去除黑色背景，seedPoint代表填充点，进行四次，即对四个角都做一次，可去除最外围的黑边
    binary = cv2.floodFill(binary, mask=None,seedPoint=(0,0),newVal=(255,255,255))[1]
    binary = cv2.floodFill(binary, mask=None, seedPoint=(0, height-1), newVal=(255, 255, 255))[1]
    binary = cv2.floodFill(binary, mask=None, seedPoint=(width-1,height-1), newVal=(255, 255, 255))[1]
    binary = cv2.floodFill(binary, mask=None, seedPoint=(width-1, 0), newVal=(255, 255, 255))[1]
    return binary

def h_Split(binary):
    """
    根据水平投影进行水平方向的切割,获取最上面一块区域的水平切割图像
        参数：
        - `binary`:二值图
    
        返回值：
        - `h_Split`:水平切割后的二值图(二维np矩阵)
    """
    #获取行列数
    row, column = binary.shape
    #获取水平投影信息并去噪
    hstats = projectionBlur(h_Projection(binary),15,15)
    #设置起始标志
    start = False
    #设置起始位置和结束位置，记录起始点和结束点坐标
    h_start, h_end= 0,0
    for i in range(row):
        #若找到第一个黑色像素且还没有起始点，则认为该点是起始点
        if hstats[i] > 0 and not start:
            h_start=i
            start = True
        #从开始点+10个像素到整个图像的一半中的最小值一般是isbn和下面二维码的分界位置，因为整个isbn号基本占20-30个像素
        if hstats[i] == hstats[h_start+10:(row//2)].min() and start:
            #记录结束点坐标
            h_end=i
            #根据其垂直投影去除大面积噪声
            h_Split=binary[h_start:h_end, :].copy()
            v_pro=projectionBlur(v_Projection(h_Split),5)
            v_start,v_end=0,0
            for k in range(0,column):
                if v_pro[k]>0:
                    v_start=k
                    break
            for j in range(column-1,-1,-1):
                if v_pro[j]>0:
                    v_end=j
                    break
            # print(v_start,',',v_end)
            if ((v_end-v_start)*1.7)<column:
                start=False
                continue
            #因为只需要获取一个区域，故直接break跳出
            break
    #截取起始位置到终止位置的图像
    h_Split=binary[h_start:h_end, :].copy()
    return h_Split

def v_Split(binary):
    """
    根据垂直投影进行垂直方向的切割,获取所有垂直切割图像
        参数：
        - `binary`:二值图
    
        返回值：
        - `v_Split`:垂直切割图像集(list)
    """
    width = binary.shape[1]
    # 获取垂直投影信息并去噪
    vstats = projectionBlur(v_Projection(binary),5,4)
    #新建list存储垂直切割后每个图片信息
    v_Split = []
    #设置起始标志和结束标志
    start_mark=False
    end_mark=False
    #设置起始位置和结束位置，记录起始点和结束点坐标
    v_start,v_end=0,0
    for i in range(width):
        #找到起点并记录
        if vstats[i]>0 and not start_mark:
            v_start=i
            start_mark=True
        #若已有起点并且当前为0，则是终点
        if vstats[i]==0 and start_mark:
            v_end=i
            end_mark=True
        #若起点和终点都有，则切割
        if start_mark==True and end_mark==True:
            # print(v_start,',',v_end)
            hcount=0
            #获取水平投影
            h_pro=h_Projection(binary[:,v_start:v_end])
            #计算高度
            for j in range(h_pro.size):
                if h_pro[j]>0:
                    hcount+=1
            #若高度大于15，则判断是字符行
            if hcount>15:
                v_Split.append(binary[:,v_start:v_end])
            start_mark=False
            end_mark=False
    return v_Split

def ishyphen(binary):
    """
    基于最小外接矩形的图像校正(二值图)
        参数：
        - `binary`:二值图
    
        返回值：
        - `bool`:True是横杠，False不是横杠
    """
    #获取水平垂直投影
    hstats = h_Projection(binary)
    vstats = v_Projection(binary)
    #hcount表示图像的高，vcount表示图像的宽
    hcount = 0
    vcount = 0
    for i in hstats:
        if i > 0:
            hcount += 1
    for j in vstats:
        if j > 0:
            vcount += 1
    #宽高比大于2的认为是横杠
    if vcount/(hcount+0.00000001) > 2:
        return True
    else:
        return False

def getdigitnum(isbnname: str):
    """
    通过文件名获取图片数字的个数
        参数：
        - `isbnname`:文件名(string)
    
        返回值：
        - `num`:图片中字数的个数(int)
    """
    num = 0
    for char in isbnname:
        if char.isdigit():
            num += 1
    return num

def preProcess(image):
    '''
    图像预处理
        参数:
        - `image`:真彩图

        返回值:
        - `binary`:处理完成的二值图
    '''
    #旋转变换
    image=correctImage(image)
    #缩放变换
    image=cv2.resize(image,(800,400))
    #直方图均衡化 (去除光照影响)
    b,g,r=cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 4))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image=cv2.merge([b,g,r])
    #拉普拉斯锐化 (会使线条变细)
    kernel_L=np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
    image = cv2.filter2D(image,-1,kernel_L)
    #gamma增强 (会加粗线条)
    sk_image = img_as_float(image)
    sk_image=exposure.adjust_gamma(sk_image, 1)
    image = img_as_ubyte(sk_image)
    #灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #去噪
    gray=cv2.GaussianBlur(gray,(5, 5), 1, 0)
    # # 局部直方图均衡化(去除光照影响)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # gray = clahe.apply(gray)
    #二值化
    binary=cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)[1]
    #清除背景黑边
    binary=ClearBackGround(binary)

    # cv2.imshow('binary',binary)
    # cv2.waitKey(0)
    return binary

def extract_numbers_to_folder(imgSrc,dstfolder):
    '''
    根据图片地址，提取该图片中所有ISBN数字，并将其放至指定文件夹中
        参数:
        - `imgSrc`:图片地址
        - `dstfolder`:输出路径

        返回值:
        - `NULL`
    '''
    #读图
    image=cv2.imread(imgSrc)
    #预处理
    binary=preProcess(image)
    #水平切割，分离出isbn区域
    h_image=h_Split(binary)
    #图像校正
    h_image=correctImage(h_image)
    #获取垂直切割图像组
    v_images=v_Split(h_image)
    #将切割后的图像组输出至指定文件中
    idx = 1
    filename = os.path.split(imgSrc)[1]
    oriname = os.path.splitext(filename)[0].split('_')[0]
    digitnum = getdigitnum(oriname)
    #从数组下标为4开始导出数字，因为ISBN分别占了前4个
    for img in v_images[4:]:
        if not ishyphen(img) and idx <= digitnum:
            #调整分割好的图片大小
            img = cv2.resize(img, config.size)
            basename = os.path.splitext(filename)[0]
            #把分割好的图片写入文件中
            cv2.imwrite(os.path.join(dstfolder, '{}_{}.jpg'.format(basename, idx)), img)
            idx += 1

def extract_numbers_all_to_folder(srcfolder: str, dstfolder:str):
    '''
    根据文件路径，提取该文件中所有图片，并将图片中所有数字放至指定文件夹
        参数:
        - `srcfolder`:文件路径
        - `dstfolder`:输出路径

        返回值:
        - `NULL`
    '''
    count = 0
    for dirpath, dirnames, filenames in os.walk(srcfolder):
        for filename in filenames:
            count += 1
            print(count, ':', filename)
            imgSrc = os.path.join(dirpath, filename)
            #读图
            image=cv2.imread(imgSrc)
            #预处理
            binary=preProcess(image)
            #水平切割，分离出isbn区域
            h_image=h_Split(binary)
            #图像校正
            h_image=correctImage(h_image)
            #获取垂直切割图像组
            v_images=v_Split(h_image)
            #将切割后的图像组输出至指定文件中
            idx = 1
            oriname = os.path.splitext(filename)[0].split('_')[0]
            digitnum = getdigitnum(oriname)
            for img in v_images[4:]:
                if not ishyphen(img) and idx <= digitnum:
                    img = cv2.resize(img, config.size)
                    basename = os.path.splitext(filename)[0]
                    cv2.imwrite(os.path.join(dstfolder, '{}_{}.jpg'.format(basename, idx)), img)
                    idx += 1

def getdigit(basename: str, idx: int):
    """取出basename中第count个数字.

    注意: idx 只考虑数字, 中间的其他字符不予考虑.
    """
    for char in basename:
        if char.isdigit():
            if idx == 1:
                return char
            else:
                idx -= 1

def gencsv(charsfolder: str, csvfile: str):
    """生成csv文件.
    """
    for dirpath, dirnames, filenames in os.walk(charsfolder):
        #按_和.之间的数字排序
        filenames.sort(key = lambda x: (int(x.split('_')[1].split('.')[0])))

        if len(filenames) > 0:
            with open(csvfile, 'w', encoding='utf-8') as fp:
                # 填充标题行, 由于我们生成的字符图片大小为20x30, 所以我们需要填充600个分隔符
                for i in range(600):
                    fp.write(',')
                fp.write('\n')
                for filename in filenames:
                    if os.path.splitext(filename)[1].lower() == '.jpg':

                        imgloc = os.path.join(dirpath, filename)
                        rawimg = cv2.imread(imgloc)
                        grayimg = cv2.cvtColor(rawimg, cv2.COLOR_BGR2GRAY)
                        basename = os.path.splitext(filename)[0]
                        oriname, idx = basename.split(sep='_')
                        digit = getdigit(oriname, int(idx))
                        if digit is None:
                            continue
                        fp.write(digit + ',')
                        grayimg.tofile(fp, sep=',', format='%d')
                        fp.write('\n')

def test(srcImg):
    """
    测试用
    """
    binary=preProcess(srcImg)
    # cv2.imshow('binary',binary)
    #水平切割，分离出isbn区域
    h_image=h_Split(binary)
    # cv2.imshow('h_image',h_image)
    #图像校正
    h_image=correctImage(h_image)
    cv2.imshow('h_image',h_image)

        # #获取垂直切割图像组
    # v=v_Projection(h_image)
    v_images=v_Split(h_image)
    for img in v_images:
        cv2.imshow('list',img)
        cv2.waitKey(0)

# if __name__ == '__main__':

#     srcImg=r'test\ISBN 978-7-208-04232-2.jpg'
#     image=cv2.imread(srcImg)
#     test(image)
#     cv2.waitKey(0)