import cv2
def get_constant_length_num(num,length=4):
    return str(num).zfill(length)
def makeImage(video_path,output_path):
    cap = cv2.VideoCapture(video_path)
    num = 1
    pic_formate='.jpg'
    while True:
        ret , frame =cap.read()
        if ret :
            midstr =str(get_constant_length_num(num))
            print(midstr)
            cv2.imwrite(output_path+midstr+pic_formate,frame)
        else :
            break
        num +=1



if __name__ == '__main__':
    makeImage('./video_test_418_fixed.mp4','test_pic/')