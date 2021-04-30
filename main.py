from hand_image import hand_image_estimation
from old_find import piano_location
import cv2
import math
# 1. 用Windows自带的camera录制视频
# 2. cutvideo函数将视频切成图片
# 2. 经过find.py 返回：
#       1) 钢琴各个琴键坐标（x, y）, piano
#       2) 钢琴被按下的琴键圆中心坐标（x_1, y_1）
#       3) 钢琴此时帧数frame_number
# 4. 经过hand_image_estimation() 返回各个手指坐标 points[(x, y)]与时间time（视频中时间）
# 5. 经过caculate_points() 计算每个手指的三个点与(, y_1)的距离y_distance, 对y_distance取average
#       比较5个y_distance，最小distance代表那根手指按了这个键，返回手指名称finger
# 6. 最终显示：time：1s finger：little finger
path = 'piano-data/fix-camera-hand/2020_10_31_10.avi'
outputFile = 'output/'
location, piano = piano_location(path, outputFile)
points = []
frame_number = {}
frame_max = len(piano)  # 多少次钢琴键被按下
for i in range(0, frame_max):
    frame_number[i] = piano[i][-1]
    print('frame number is :', frame_number[i])


camera = cv2.VideoCapture(path)
camera.set(3, 1980)
camera.set(4, 1080)
frame_num = 0
timeF = 4
i = 0
x0 = 707
y0 = 489
x1 = 1029
y1 = 890
fps = camera.get(cv2.CAP_PROP_FPS)
frame_all = camera.get(cv2.CAP_PROP_FRAME_COUNT)
frame_all = int(frame_all/4)   # 隔4帧取1帧

for frame_num in range(0, frame_all):

    if frame_num == frame_number[i]:  # 视频帧与需要手部动捕的帧一一对应
        points = []
        path = outputFile + str(frame_num) + '.jpg'
        img1 = cv2.imread(path)
        cut_img = img1[y0:y1, x0:x1]  # 裁剪图像，[y0:y1, x0:x1] x0,y0 左上  x1, y1 右下
        cv2.imwrite(path, cut_img)
        points = hand_image_estimation(path=outputFile + str(frame_num) + '.jpg')
        print('WIn!')
        thumb_x = {}
        index_x = {}
        middle_x = {}
        ring_x = {}
        little_x = {}
        j = 0
        for j in range(1, 5):  #  这里1到5是因为有22总点，取里面20个
            # points[2-4]: 拇指指关节， points[6-8]:食指指关节， points[10-12]: 中指指关节
            # points[14-16]: 无名指指关节， points[18-20]: 小拇指指关节
            x_0, y_0 = points[j]
            x_1, y_1 = points[j+4]
            x_2, y_2 = points[j+8]
            x_3, y_3 = points[j+12]
            x_4, y_4 = points[j+16]
            thumb_x[j] = x_0
            index_x[j] = x_1
            middle_x[j] = x_2
            ring_x[j] = x_3
            little_x[j] = x_4
        finger_average = []
        # 各个指关节权重 weight
        # 注意points[4]，[8]，[12]，[16]，[20] 分别代表拇指指尖，食指指尖，中指指尖，无名指指尖，小拇指指尖
        thumb_average, index_average, middle_average, ring_average, little_average = 0,0,0,0,0
        # (权重*指关节x3)/3.0 + x0(还原成原图坐标)
        s_weight = 1
        m_weight = 1
        thumb_average = (s_weight * thumb_x[1] + m_weight * thumb_x[2] + thumb_x[3])/3.0+x0; finger_average.append(thumb_average)
        index_average = (s_weight * index_x[1] + m_weight * index_x[2] + index_x[3])/3.0+x0; finger_average.append(index_average)
        middle_average = (s_weight * middle_x[1] + m_weight * middle_x[2] + middle_x[3])/3.0+x0; finger_average.append(middle_average)
        ring_average = (s_weight * ring_x[1] + m_weight * ring_x[2] + ring_x[3])/3.0+x0; finger_average.append(ring_average)
        little_average = (s_weight * little_x[1] + m_weight * little_x[2] + little_x[3])/3.0+x0; finger_average.append(little_average)
        #用数组存放平均值
        res = []
        minn = 10000  # 首先，得到了平均值，其次，我们有当前的照片，之后，我们需要放回原图，并进行比较，我现在需要读取，然后就可以读取他的坐标
        for j in range(len(piano[i])-1):  # 这里减一的原因是一次可能会弹多个键，去掉最后一位的视频帧
            finger=0
            for k in range(len(finger_average)):  # 只识别一只手，len(fig_average_list) = 5
                    if(abs(finger_average[k]-location[piano[i][j]][0])<minn):
                        minn=abs(finger_average[k]-location[piano[i][j]][0])
                        finger=k
                    else:
                        continue
            print(finger, frame_num)
            res.append(finger)





        print('thumb is :', thumb_x)
        print('index is :', index_x)
        print('middle is :', middle_x)
        print('ring is :', ring_x)
        print('little is :', little_x)


        if i == frame_max-1:
            break
        i = i+1

print('END')
