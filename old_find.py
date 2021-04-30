'''
2020.10.31 wang
'''
import cv2


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


def dis(x1, y1, x2, y2, r):
    if ((x1 - x2) * (x1 - x2) + (y1 - y2) *
        (y1 - y2) > r * r):  # 只有这个点的距离之和大于r*r 才能够进入
        return True
    else:
        return False


def make(ip, tips, x1, x2, y1, y2):
    tips += 1
    picc = cv2.imread(ip)
    picc = picc[y1:y2, x1:x2]
    GrayImage = cv2.cvtColor(picc, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(GrayImage, 50, 255, cv2.THRESH_OTSU)
    length = image.shape[0]
    depth = image.shape[1]
    shrinkedpic = image

    median = cv2.medianBlur(image, 5)
    cannypic = cv2.Canny(median, 10, 200)
    contours, hierarchy = cv2.findContours(cannypic, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_NONE)
    r = []
    x = []
    y = []
    for i in range(len(contours)):  # 通过最大最小值确定中心位置
        maxn = 0
        mx = 0
        mi = 10000
        minn = 10000
        # print(contours[i])
        for j in range(len(contours[i])):
            maxn = max(contours[i][j][0][1], maxn)  # 由于 contours 这个函数，需要多层拨开
            mx = max(contours[i][j][0][0], mx)
            mi = min(contours[i][j][0][0], mi)
            minn = min(contours[i][j][0][1], minn)
        leng = maxn - minn  # y方向的长度
        lenn = mx - mi  # 为了保证结果的正确性，这里采用x和y同步判断

        if (leng <= 10 and lenn <= 10):
            if (i == 0):  # 第一个点分出来计算
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
    cnt = []  #  每个宽度的数量
    high = []
    # print(len(high))
    for i in range(len(x)):
        if (len(high)):
            flag = 1
            # print("there")
            for j in range(len(high)):  # j 是 高度的遍历
                if (y[i] > high[j] - 50 and y[i] < high[j] + 50):  # 确定他在一个条带中
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
        if (high[mnui] + 50 > y[i] and high[mnui] - 50 < y[i]):
            res.append(i)
            rex.append(x[i])
            rey.append(y[i])
            retu.append(Pose(x[i], y[i]))

    retu = sorted(retu)
    return retu

# ================================================
def piano_location(path, output):
    lis=[] # 读取照片阵列
    videoFile = path
    outputFile = output
    vc = cv2.VideoCapture(videoFile)
    c = 1
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        print('openerror!')
        rval = False

    timeF = 4  #视频帧计数间隔次数
    while rval:
        rval, frame = vc.read()
        if c % timeF == 0:
            cv2.imwrite(outputFile + str(int(c / timeF)) + '.jpg', frame)
            lis.append(outputFile+str(int(c/timeF))+'.jpg')
            print(outputFile + str(int(c / timeF)) + '.jpg')
        c += 1

        cv2.waitKey(1)
    vc.release()
    lis1=[]
    x1=767
    x2=950
    y1=838
    y2=885
    for i in range(7,len(lis)-14):
        a=make(outputFile+str(i)+'.jpg',i,x1,x2,y1,y2)
        b=make(outputFile+str(i+1)+'.jpg',i+1,x1,x2,y1,y2)
        if(len(a)==len(b)):
            lis2=[]
            for j in range(len(a)):
                if(a[j].y-b[j].y>1):
                    lis2.append(j)
            if(len(lis2)):
                lis2.append(i+1)
                lis1.append(lis2)
    # lis1 说明！！！！！
    # 每个的lis1的-1标识的是帧数！！！
    location=[]
    a=make(outputFile+str(7)+'.jpg',7,x1,x2,y1,y2)
    for i in range(len(a)):
        lis3=[]
        lis3.append(int(a[i].x+x1))
        lis3.append(int(a[i].y+y1))
        location.append(lis3)
    print('location is :\n', location)
    print('list is :\n', lis1)
    # location 是键盘
    return location, lis1

