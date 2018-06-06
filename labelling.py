import cv2
import sys
import csv
import os.path
import numpy as np

def player(video_path):
    #動画のパス
    root, video_name = os.path.split(video_path)

    #csvのパス
    video_root, ext = os.path.splitext(video_name)
    csv_path = '{0}.csv'.format(video_root)
    

    #動画読み込み
    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #labels初期化(csvがあれば読み込み)
    if os.path.exists(csv_path):      
        f = open(csv_path, 'r')
        reader = csv.reader(f)
        labels = []
        for row in reader:
            for i in row:
                labels.append(int(float(i)))
    else:
        labels = [0 for i in range(total)]



    frame_num = 0
    block_num = 0

    while(block_num * 100 < total):
        #フレームを100枚ずつ配列に格納
        frames = []
        while(cap.isOpened()):

            ret, frame = cap.read()

            if frame is None:
                break

            frames.append(frame)

            sys.stdout.write("\r%d" % len(frames))
            sys.stdout.flush()

            if len(frames) >= 100:
                break

        #100枚について繰り返し
        while frame_num - block_num * 100 < len(frames):
            frame = frames[frame_num - block_num * 100]
  
            frame = cv2.resize(frame,(1280, 720))

            cv2.putText(frame,str(frame_num + 1) + '/' + str(total),(15,45),cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255),2,cv2.LINE_AA)

            cv2.imshow("Frame", frame)

            key = cv2.waitKey()

            if key == 121:    #y
                labels[frame_num] = 1
                frame_num += 1

            elif key == 110:     #n
                labels[frame_num] = 0
                frame_num += 1

            elif key == 98:      #b
                if frame_num - block_num * 100 > 0:
                    frame_num -= 1                

            elif key == 115:     #s
                frame_num = block_num * 100 + len(frames)

            elif key == 32:      #space
                cap.release()
                cv2.destroyAllWindows()
                save(labels,csv_path)
                sys.exit()

            #else:
                #labels[frame_num] = 0
                #frame_num += 1        
 
            key = 0

        save(labels,csv_path)

        block_num += 1


    cap.release()
    cv2.destroyAllWindows()

    save(labels,csv_path)

    sys.exit()
    

def save(labels,csv_path):
    np.savetxt(csv_path,labels,delimiter=',')
    


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('not enough parms')
        exit()

    video_path = sys.argv[1]

    player(video_path)
