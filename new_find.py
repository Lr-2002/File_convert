"""
代码重构：
1. 这个文件是干嘛的
    文件用于处理36键好像是
2. 每个函数又都干了些什么
"""
import cv2
import numpy as np


# <editor-fold desc="通过结构体表示一个点">
class Pose:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    '''def __cmp__(self,other):
        return cmp(self.x,other.x)'''

    def __eq__(self, other):
        return self.x == other.x

    def __le__(self, other):
        return self.x < other.x

    def __gt__(self, other):
        return self.x > other.x
# </editor-fold>


# <editor-fold desc="应该是用来去除重复的点">
def dis(x1, y1, x2, y2, r):
    if ((x1 - x2) * (x1 - x2) + (y1 - y2) *
            (y1 - y2) > r * r):  # 只有这个点的距离之和大于r*r 才能够进入
        return True
    else:
        return False
# </editor-fold>



# <editor-fold desc="函数有两个作用：1. 图片预处理，将图片变成容易处理的图片 2. 计算出这张图片上的点的坐标">
def make(a,  x1, x2, y1, y2):
    # <editor-fold desc="图片读取和裁剪">
    # print('path : '+ip)
    # a = cv2.imread(ip)  # black 多了一点 # todo 只是测试
    a = a[y1:y2, x1:x2]
    # </editor-fold>


    # <editor-fold desc="图片预处理，生成容易处理的图片">
    kernel = np.ones((3, 3), np.uint8)
    dic = cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel)
    dst = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)

    '''cv2.imshow('open',dst)
    cv2.imshow('close',dic)
    cv2.waitKey(0)'''

    grayimage = cv2.cvtColor(dic, cv2.COLOR_BGR2GRAY)  # 先将图片转换为灰图，再进行处理
    #cv2.imshow("1", img)
    #cv2.waitKey(0)
    blur = cv2.GaussianBlur(grayimage, (1, 1), 0)  # 用高斯滤波消除噪音
    '''
    直接取消高斯滤波
    '''

    #cv2.imshow("jpo", blur)
    #cv2.waitKey(0)
    ret, th = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY
        or cv2.THRESH_OTSU)  # 由于otsu只能够处理单通道的值，，所以需要将前边的转变为单通道值，，也就是灰图
    cannypic = cv2.Canny(th, 1, 200)
    contours, hierarchy = cv2.findContours(cannypic, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_NONE)
    srcc = cv2.drawContours(dic, contours, -1, (0, 0, 255), 1)
    # </editor-fold>


    # <editor-fold desc="列表初始化">
    r = []
    x = []
    y = []
    # </editor-fold>
    # <editor-fold desc="确定每个琴键标定点的位置">
    for i in range(len(contours)):  # 通过最大最小值确定中心位置
        maxn = 0
        mx = 0
        mi = 10000
        minn = 10000
        for j in range(len(contours[i])):
            maxn = max(contours[i][j][0][1], maxn)  # 由于 contours 这个函数，需要多层拨开
            mx = max(contours[i][j][0][0], mx)
            mi = min(contours[i][j][0][0], mi)
            minn = min(contours[i][j][0][1], minn)
        leng = maxn - minn  # y方向的长度
        lenn = mx - mi  # 为了保证结果的正确性，这里采用x和y同步判断

        if (leng <=35 and lenn <= 35 and leng >= 2 and lenn >= 2):
            if (len(x) == 0):  # 第一个点分出来计算
                x.append(mx / 2 + mi / 2)
                y.append(maxn / 2 + minn / 2)
                rr = max(leng / 2, lenn / 2)
                r.append(rr)
            else:
                flag = False
                for j in range(len(x)):  # 问题是，他妈只能识别一个点
                    if (dis(mx / 2 + mi / 2, maxn / 2 + minn / 2, x[j], y[j],
                            r[j])):
                        flag = True  # 只有当每个都整长是
                    else:
                        flag = False
                if (flag):
                    x.append(mx / 2 + mi / 2)
                    y.append(maxn / 2 + minn / 2)
                    rr = max(leng / 2, lenn / 2)
                    r.append(rr)
    # </editor-fold>
    # <editor-fold desc="对标定点进行去重">
    cnt = []  #  每个宽度的数量
    high = []
    # print(len(high))
    for i in range(len(x)):
        if (len(high)):
            flag = 1
            # print("there")
            for j in range(len(high)):  # j 是 高度的遍历
                if (y[i] > high[j] - 20 and y[i] < high[j] + 20):  # 确定他在一个条带中
                    cnt[j] += 1  # j 对应的边的数量要增加
                    flag = 0  # 在这里，我们把flag改为0 ，然
                    break
            if flag:
                high.append(y[i])  # 这里，我们不能在循环里 只有所有的循环结束了
                cnt.append(1)
        else:
            high.append(y[i])
            cnt.append(1)
            # rs.append(r[i])
    mnum = 0  # 这是最大的对应的high
    mnui = 0  # 这是最大的对应的i值
    for i in range(len(cnt)):  # cnt 是在对应条带里的点的多少
        if (cnt[i] > mnum):
            mnum = cnt[i]
            mnui = i
    res = []
    rex = []
    rey = []
    retu = []
    for i in range(len(x)):
        if (high[mnui] + 20 > y[i] and high[mnui] - 20 < y[i]):
            res.append(i)
            rex.append(x[i])
            rey.append(y[i])
            retu.append(Pose(x[i], y[i]))

    retu = sorted(retu)
    return retu
    # </editor-fold>
# </editor-fold>

# <editor-fold desc="make_img 函数有两个作用：1. 图片预处理，将图片变成容易处理的图片 2. 计算出这张图片上的点的坐标">
def make_img(a,  x1, x2, y1, y2):
    # <editor-fold desc="图片读取和裁剪">
    print(type(x1))

    a = a[y1:y2, x1:x2]
    # </editor-fold>


    # <editor-fold desc="图片预处理，生成容易处理的图片">
    kernel = np.ones((3, 3), np.uint8)
    dic = cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel)
    dst = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)

    '''cv2.imshow('open',dst)
    cv2.imshow('close',dic)
    cv2.waitKey(0)'''

    grayimage = cv2.cvtColor(dic, cv2.COLOR_BGR2GRAY)  # 先将图片转换为灰图，再进行处理
    #cv2.imshow("1", img)
    #cv2.waitKey(0)
    blur = cv2.GaussianBlur(grayimage, (1, 1), 0)  # 用高斯滤波消除噪音
    '''
    直接取消高斯滤波
    '''

    #cv2.imshow("jpo", blur)
    #cv2.waitKey(0)
    ret, th = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY
        or cv2.THRESH_OTSU)  # 由于otsu只能够处理单通道的值，，所以需要将前边的转变为单通道值，，也就是灰图
    cannypic = cv2.Canny(th, 1, 200)
    contours, hierarchy = cv2.findContours(cannypic, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_NONE)
    srcc = cv2.drawContours(dic, contours, -1, (0, 0, 255), 1)
    # </editor-fold>


    # <editor-fold desc="列表初始化">
    r = []
    x = []
    y = []
    # </editor-fold>
    # <editor-fold desc="确定每个琴键标定点的位置">
    for i in range(len(contours)):  # 通过最大最小值确定中心位置
        maxn = 0
        mx = 0
        mi = 10000
        minn = 10000
        for j in range(len(contours[i])):
            maxn = max(contours[i][j][0][1], maxn)  # 由于 contours 这个函数，需要多层拨开
            mx = max(contours[i][j][0][0], mx)
            mi = min(contours[i][j][0][0], mi)
            minn = min(contours[i][j][0][1], minn)
        leng = maxn - minn  # y方向的长度
        lenn = mx - mi  # 为了保证结果的正确性，这里采用x和y同步判断

        if (leng <=35 and lenn <= 35 and leng >= 2 and lenn >= 2):
            if (len(x) == 0):  # 第一个点分出来计算
                x.append(mx / 2 + mi / 2)
                y.append(maxn / 2 + minn / 2)
                rr = max(leng / 2, lenn / 2)
                r.append(rr)
            else:
                flag = False
                for j in range(len(x)):  # 问题是，他妈只能识别一个点
                    if (dis(mx / 2 + mi / 2, maxn / 2 + minn / 2, x[j], y[j],
                            r[j])):
                        flag = True  # 只有当每个都整长是
                    else:
                        flag = False
                if (flag):
                    x.append(mx / 2 + mi / 2)
                    y.append(maxn / 2 + minn / 2)
                    rr = max(leng / 2, lenn / 2)
                    r.append(rr)
    # </editor-fold>
    # <editor-fold desc="对标定点进行去重">
    cnt = []  #  每个宽度的数量
    high = []
    # print(len(high))
    for i in range(len(x)):
        if (len(high)):
            flag = 1
            # print("there")
            for j in range(len(high)):  # j 是 高度的遍历
                if (y[i] > high[j] - 20 and y[i] < high[j] + 20):  # 确定他在一个条带中
                    cnt[j] += 1  # j 对应的边的数量要增加
                    flag = 0  # 在这里，我们把flag改为0 ，然
                    break
            if flag:
                high.append(y[i])  # 这里，我们不能在循环里 只有所有的循环结束了
                cnt.append(1)
        else:
            high.append(y[i])
            cnt.append(1)
            # rs.append(r[i])
    mnum = 0  # 这是最大的对应的high
    mnui = 0  # 这是最大的对应的i值
    for i in range(len(cnt)):  # cnt 是在对应条带里的点的多少
        if (cnt[i] > mnum):
            mnum = cnt[i]
            mnui = i
    res = []
    rex = []
    rey = []
    retu = []
    for i in range(len(x)):
        if (high[mnui] + 20 > y[i] and high[mnui] - 20 < y[i]):
            res.append(i)
            rex.append(x[i])
            rey.append(y[i])
            retu.append(Pose(x[i], y[i]))

    retu = sorted(retu)
    return retu
    # </editor-fold>
# </editor-fold>

def get_point(p1,p2):
    return p1[0],p1[1],p2[0],p2[1]


# <editor-fold desc="关键点是给出两个截图的坐标">
if __name__=='__main__':
    location = []

    x1,y1,x2,y2 =get_point((229,463),(1571,604))
    test = make('./test_pic/0399.jpg',x1,x2,y1,y2)
    print(len(test))
#     a = make('./test_img.jpg', 7, x1, x2, y1, y2)
#     print('a: ',len(a))
#     cv2.waitKey(0)
#     for i in range(len(a)):
#         lis3 = []
#         lis3.append(int(a[i].x + x1))
#         lis3.append(int(a[i].y + y1))
#         location.append(lis3)
#     # location 是键盘
#     print(location)
#     print(len(location))
#     lis1 = []
#     '''
#     当前遇到的问题：
#     点基本上都能够识别
#     但是在比较的过程中
#     需要更加准确的优化计算
#
#     目标的结果，应该是42个点
#     '''
#     #   26
#     for i in range(200, 400):
#         '''if i==170 or i==305 or i==243 :
#             continue'''
#         if i == 170:
#             continue
#         a = make('./test_img.jpg', i, x1, x2, y1, y2)
#         if (len(a) == 88):
#             lis2 = []
#             for j in range(len(a)):
#                 for k in range(26,61):
#                     if a[j].x < location[k][0] - x1 + 2 and a[
#                         j].x > location[k][0] - x1 - 2:
#                         if len(lis1):
#                             if a[j].y - location[k][
#                                 1] + y1 <= -1:  # 这里试图将不同的图像分开，但其实，只需要等到点回到原来的位置就行了
#                                 if i - lis1[len(lis1) - 1][-1] <= 2 and j == lis1[
#                                     len(lis1) - 1][-2]:
#                                     del lis1[len(lis1) - 1]
#                                     lis2.append(j)
#                                     print(j)
#                                 else:
#                                     lis2.append(j)
#                                     print(j)
#                         else:
#                             if a[j].y - location[k][1] + y1 <= -1:
#                                 lis2.append(j)
#             if (len(lis2)):
#                 lis2.append(i)
#                 lis1.append(lis2)
#     # lis1 说明！！！！！
#     # 每个的lis1的-1标识的是帧数！！！
#     '''
#
#     最后再于谱子进行比较可能会好些
#
#     '''
#     # print(location)
#     # print(lis1)
#     # cnt = 0  # cnt 用来表示这个数字积累的多少
#     # num = 0  # num用来表示这个数字是谁
#     # cnt_lis = []
#     '''
#     接下来是一个通过谱子对识别结果进行修正的过程
#
#     '''
# # </editor-fold>
