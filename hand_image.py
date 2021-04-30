import cv2
import time
import numpy as np


def hand_image_estimation(frame, framenumber):
    protoFile = "./pose_deploy.prototext"
    weightFile = "./pose_iter_102000.caffemodel"
    nPoints = 22
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4],
              [0, 5], [5, 6], [6, 7], [7, 8],
              [0, 9], [9, 10], [10, 11], [11, 12],
              [0, 13], [13, 14], [14, 15], [15, 16],
              [0, 17], [17, 18], [18, 19], [19, 20]]
    net = cv2.dnn.readNetFromCaffe(protoFile, weightFile)

    # frame = cv2.imread(path)
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth/frameHeight

    threshold = 0.1

    # print("Input Image Frame Height:", frameHeight)
    # print("input Image Frame Width: ", frameWidth)

    t = time.time()
    # input image dimensions for the network
    inHeight = 368
    inWidth = int(((aspect_ratio*inHeight)*8)//8)
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    # print("time taken by network : {:.3f}".format(time.time() - t))

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part. probMap 'Probability Map'
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold:
            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            # 各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frame, points[partA], 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
    # 参数说明
    # img：输入的图片data
    # center：圆心位置
    # radius：圆的半径
    # color：圆的颜色
    # thickness：圆形轮廓的粗细（如果为正）。负厚度表示要绘制实心圆。
    # lineType： 圆边界的类型。
    # shift：中心坐标和半径值中的小数位数

    # cv2.imwrite( 'output/' + str(framenumber) + '_keypoints.jpg', frameCopy)
    # cv2.imwrite( 'output/' + str(framenumber) + '_skeleton.jpg', frame)
    # # 单独存在一个文件夹里面方便检查
    # cv2.imwrite( 'output/hand/' + str(framenumber) + '_keypoints.jpg', frameCopy)
    # cv2.imwrite( 'output/hand/' + str(framenumber) + '_skeleton.jpg', frame)
    #
    # # 这里返回的 name = 'outputFile + str(frame_num)'
    # # 应该是 name = 'output/1
    #
    # cv2.imwrite('output/Output-Keypoints.jpg', frameCopy)
    # cv2.imwrite('output/Output-Skeleton.jpg', frame)


    print("Total time taken : {:.3f}".format(time.time() - t))
    # print('the frame is\n', frame)
    # print('the framecopy is:\n', frameCopy)
    # print('the points are:\n', points)
    # print('points number:\n', len(points))
    return points,frame



'''
========================================================================================================================
'''

