import cv2

path = 'piano-data/fix-camera-hand/2020_10_31_10.avi'
camera = cv2.VideoCapture(path)
camera.set(3, 1980)
camera.set(4, 1080)
hasFrame, frame = camera.read()
fps = camera.get(cv2.CAP_PROP_FPS)
frame_all = camera.get(cv2.CAP_PROP_FRAME_COUNT)
frame_all = int(frame_all)
x0 = 707
y0 = 489
x1 = 1029
y1 = 890
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('D:/PycharmProjects/hand-piano/piano-data/fix-camera-hand/2020_11_13_01.mp4',
                               fourcc, 24.0, (x1-x0, y1-y0), True)  # 24.0 代表帧率

for frame_num in range(0, frame_all-1):
    hasFrame, frame = camera.read()
    cut_img = frame[y0:y1, x0:x1]
    output_video.write(cut_img)
    print('your fucking computer is still working: ', frame_num)
    frame_num = frame_num + 1

print('OVER !')
camera.release()
output_video.release()
cv2.destroyAllWindows()
