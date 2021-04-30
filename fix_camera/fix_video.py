import cv2
import numpy as np

# 导入前面fix.py 计算的镜头内外参数
u = np.load('parameter/u.npy')
v = np.load('parameter/v.npy')
mtx = np.load('parameter/mtx.npy')
dist = np.load('parameter/dist.npy')

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))

# 需要的参数：mtx，dist，u，v

# 打开待处理视频
camera = cv2.VideoCapture('../video_test_418.mp4')
camera.set(3, 1980)
camera.set(4, 1080)
frame_num = 1
fps = camera.get(cv2.CAP_PROP_FPS)
frame_all = camera.get(cv2.CAP_PROP_FRAME_COUNT)
print("[INFO] 视频FPS: {}".format(fps))
print("[INFO] 视频总帧数: {}".format(frame_all))
print("[INFO] 视频时长: {}s".format(frame_all/fps))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('../video_test_418_fixed.mp4',
                               fourcc, 24.0, (1920, 1080), True)  # 24.0 代表帧率
frame_all = int(frame_all)
for frame_num in range(0, frame_all):
    (grabbed, frame) = camera.read()
    h1, w1 = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))
    if frame_num >= 0 and frame_num<frame_all:
        # 纠正畸变
        dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        #dst2 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        mapx, mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w1,h1),5)
        dst2 = cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)
        output_video.write(dst2)
        print('your fucking computer is still working: ', frame_num)
    if cv2.waitKey(1) == 27:  #Esc quit
        print('exiting.....')
        break

    frame_num = frame_num + 1 # 逐帧处理，若要换把1换成别的就可以



# while camera.isOpened():
#     (grabbed, frame) = camera.read()
#     h1, w1 = frame.shape[:2]
#     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))
#     if frame_num >= begin and frame_num<frame_all-2:
#         # 纠正畸变
#         dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
#         #dst2 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
#         mapx, mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w1,h1),5)
#         dst2 = cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)
#         output_video.write(dst2)
#     if cv2.waitKey(1) == 27:  #Esc quit
#         print('exiting.....')
#         break
#     print('your fucking computer is still working: ', frame_num)
#     frame_num = frame_num + 1 # 逐帧处理，若要换把1换成别的就可以



    # 裁剪图像，输出纠正畸变以后的图片
    # x, y, w1, h1 = roi
    # dst1 = dst1[y:y + h1, x:x + w1]

    # cv2.imshow('frame',dst2)
    # cv2.imshow('dst1',dst1)
    # cv2.imshow('dst2', dst2)
    i = 0

    '''
    问题在于机子太卡，丢失很多帧
    if grabbed is True:

        cv2.imshow('frame', dst2)
        output_video.write(dst2)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
    '''



    '''
    同样的问题，机子太卡
        while True:
        if cv2.waitKey(1) & 0xFF == ord('j'):  # 按j保存一张图片
            i += 1
            u = str(i)
            firename = str('D:/PycharmProjects/hand-piano/piano-data/fix-camera-hand/fix-hand' + u + '.jpg')
            cv2.imwrite(firename, dst1)
            print('写入：', firename)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('正在退出.....')
            break
    if cv2.waitKey(1) & 0xFF == ord('a'):
        print('正在退出.....')
        break
    '''
    '''
    存储图片
    if cv2.waitKey(1) & 0xFF == ord('j'):
        filename = str('D:/PycharmProjects/hand-piano/piano-data/fix-camera-hand/fix-hand1.jpg')
        cv2.imwrite(filename, dst2)
        print('Writting...', filename)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Exit...')
        break
    '''

print('OVER !')
camera.release()
output_video.release()
cv2.destroyAllWindows()


