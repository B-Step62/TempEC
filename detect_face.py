import cv2
import csv
import sys
import os.path
import numpy as np
import dlib
import copy

eyes_detector = dlib.get_frontal_face_detector()
eyes_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

img_rows, img_cols = 36, 60 #画像の縦横サイズ

#GCNはdata_utilで行う

def preprocessing_img(img):
        resized = cv2.resize(img, (img_cols, img_rows))
        #gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
        return resized #gray

def clip_eye(img, parts):
    parts_x = []
    parts_y = []
    for part in parts:
        parts_x.append(part.x)
        parts_y.append(part.y)

    top = min(parts_y)
    bottom = max(parts_y)
    left = min(parts_x)
    right = max(parts_x)

    width = right - left
    height = bottom - top

    #目領域の識別誤差に対応するためのマージン
    margin_w = width * 0.4
    margin_h = height * 0.4

    x = np.random.uniform(-margin_w,margin_w)
    y = np.random.uniform(-margin_h,margin_h)

    top    = top    - margin_h + y * 0.1
    bottom = bottom + margin_h + y * 0.1
    left   = left   - margin_w + x * 0.1
    right  = right  + margin_w + x * 0.1

    #width = right - left
    #height = bottom - top

    #60:36くらいにする
    if height < width * 0.6:     #横長の場合
        top = (top + bottom) / 2 - width * 0.3
        bottm = (top + bottom) /2 + width * 0.3
    else:     #縦長の場合
        left = (left + right) / 2 - height * 0.3
        right = (left + right) /2 + height * 0.3


    return img[int(top + 0.5):int(bottom + 0.5),int(left + 0.5):int(right + 0.5)]

def detect_shape(run_type,img):
    dets = eyes_detector(img, 1)

    lefts = []
    rights = []
    flag = 0
    
    for k,d in enumerate(dets):
        flag = 1

        shape = eyes_predictor(img, d)
        
        for shape_point_count in range(shape.num_parts):
            shape_point = shape.part(shape_point_count)

            #cv2.putText(marked_img, '.',(int(shape_point.x), int(shape_point.y)),cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

            if shape_point_count == 36: #左目尻
                left_position = shape_point
            if shape_point_count == 45: #右目尻
                right_position = shape_point

            if shape_point_count > 35 and shape_point_count < 42:
                lefts.append(shape_point)
            elif shape_point_count > 41 and shape_point_count < 48:
                rights.append(shape_point)

    if run_type == 'detect_face':              
        if lefts: 
            return clip_eye(img,lefts), clip_eye(img,rights), left_position, right_position #, marked_img
        else:
            return None, None, None, None#, marked_img
    elif run_type == 'estimate_headpose':
        if flag == 0:
            return None
        else: 
            return shape
    

def detect_face(video_path,width,height):
    #動画読み込み
    cap = cv2.VideoCapture(video_path)

    dir_path ,video = os.path.split(video_path)
    video_name, ext = os.path.splitext(video)


    #mkdir
    left_dir = 'dataset/images_color/{0}/left'.format(video_name)
    right_dir = 'dataset/images_color/{0}/right'.format(video_name)
    if not os.path.exists('dataset/images_color/{0}'.format(video_name)):
        os.mkdir('dataset/images_color/{0}'.format(video_name))
    if not os.path.exists('dataset/images_color/{0}/left'.format(video_name)):
        os.mkdir('dataset/images_color/{0}/left'.format(video_name))
    if not os.path.exists('dataset/images_color/{0}/right'.format(video_name)):
        os.mkdir('dataset/images_color/{0}/right'.format(video_name))
    if not os.path.exists('dataset/images_color/{0}/original'.format(video_name)):
        os.mkdir('dataset/images_color/{0}/original'.format(video_name))


    face_positions = []

    frame_num = 0    
    prev_left_width = width

    while(cap.isOpened()):
        #フレームを取得
        ret, frame = cap.read()

        if frame is None:
            break


        sys.stdout.write("\r%d" % frame_num)
        sys.stdout.flush()

        cv2.imwrite(os.path.join('dataset/images_color/{0}/original'.format(video_name),'{0}.png'.format(frame_num)), frame)

        left_eye_img, right_eye_img, left_position, right_position = detect_shape('detect_face',frame)

        if left_position == None:
            face_positions.append([0.5, 0.5, 0.5, 0.5])
        else:
            face_positions.append([left_position.x / width, left_position.y / height, right_position.x / width, right_position.y / height])

        if left_eye_img is None or right_eye_img is None:
            prev_left_width = width #いったん初期化
            frame_num += 1
            continue
        else:
            left_height, left_width, left_channels = left_eye_img.shape  
            right_height, right_width, right_channels = right_eye_img.shape
            if left_height > 0 and right_height > 0 and left_width > 0 and right_width > 0 and left_width < prev_left_width * 3:
                processed_left_eye_img = preprocessing_img(left_eye_img)
                processed_right_eye_img = preprocessing_img(right_eye_img)                           
                cv2.imwrite(os.path.join(left_dir,'{0}.png'.format(frame_num)), processed_left_eye_img)
                cv2.imwrite(os.path.join(right_dir,'{0}.png'.format(frame_num)), processed_right_eye_img)

                prev_left_width = left_width

        frame_num += 1
        

    #np.savetxt('dataset/face_positions/{0}.csv'.format(video_name), face_positions,delimiter=',')

    cap.release()


def detect_face_single(img_path):

    head, file_name = os.path.split(img_path) 

    img = cv2.imread(img_path)

    shape = detect_shape('estimate_headpose',img)

    for shape_point_count in range(shape.num_parts):
        shape_point = shape.part(shape_point_count)

        cv2.circle(img, (int(shape_point.x), int(shape_point.y)), 2, (0,0,255), -1)


        if shape_point_count == 0:
            left = int(shape_point.x)
        elif shape_point_count == 16:
            right = int(shape_point.x)
        elif shape_point_count == 24:
            top = int(shape_point.y)
        elif shape_point_count == 8:
            bottom = int(shape_point.y)
    left = left - (right - left) // 5
    right = right + (right - left) // 5
    top = top - (bottom - top) // 3
    bottom = bottom + (bottom - top) // 10
    img = img[top:bottom,left:right]
    cv2.imwrite(file_name, img)


def estimate_headpose(video_path):

    #動画読み込み
    cap = cv2.VideoCapture(video_path)

    dir ,video = os.path.split(video_path)
    video_name, ext = os.path.splitext(video)

    dir_path = 'dataset/images/{0}'.format(video_name)

    if not os.path.exists('dataset/images_color/{0}'.format(video_name)):
        os.mkdir('dataset/images_color/{0}'.format(video_name))
    if not os.path.exists('dataset/images_color/{0}/headpose'.format(video_name)):
        os.mkdir('dataset/images_color/{0}/headpose'.format(video_name))

    headposes = []

    frame_num = 0    
    while(cap.isOpened()):
        #フレームを取得
        ret, frame = cap.read()

        if frame is None:
            break

        sys.stdout.write("\r%d" % frame_num)
        sys.stdout.flush()

        shape = detect_shape('estimate_headpose',frame)

        if shape is None:
            headposes.append([0, 0, 0])
            frame_num += 1
            continue

        #position of face parts 
        image_points = np.array([
                            (shape.part(33).x, shape.part(33).y),     # Nose tip
                            (shape.part(8).x, shape.part(8).y),     # Chin
                            (shape.part(36).x, shape.part(36).y),     # Left eye left corner
                            (shape.part(45).x, shape.part(45).y),    # Right eye right corne
                            (shape.part(48).x, shape.part(48).y),    # Left Mouth corner
                            (shape.part(54).x, shape.part(54).y)      # Right mouth corner
                        ], dtype="double")

        # 3D model points.
        model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                         
                        ])

        width = frame.shape[1]
        height = frame.shape[0]

        focal_length = width
        center = (width, height) 
        camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)


        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        #rotation_matrix, el = cv2.Rodrigues(rotation_vector)

        #projMat = np.array([[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], 0],
        #                [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], 0],
        #                [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], 0]])

        #matA, matB, matC, matD, matE, matF, eulerAngles = cv2.decomposeProjectionMatrix(projMat, camera_matrix, rotation_matrix, translation_vector)

        #yaw   = eulerAngles[1] 
        #pitch = eulerAngles[0]
        #roll  = eulerAngles[2]

        #headposes.append([yaw, pitch, roll])

        normalized_rotation_vector = rotation_vector / np.linalg.norm(rotation_vector)

        normalized_rotation_vector_list = [normalized_rotation_vector[0][0],normalized_rotation_vector[1][0],normalized_rotation_vector[2][0]] 

        headposes.append(normalized_rotation_vector_list)


        #np.savetxt('dataset/head_pose_vec/{0}.csv'.format(video_name), headposes,delimiter=',')


        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
 
 
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
 
        cv2.line(frame, p1, p2, (255,0,0), 4)

        cv2.imwrite('dataset/images_color/{0}/headpose/{1}.png'.format(video_name,frame_num), frame)

        frame_num += 1

    cap.release()

def crop(img_path):

    head, file_name = os.path.split(img_path) 

    img = cv2.imread(img_path)

    shape = detect_shape('estimate_headpose',img)

    for shape_point_count in range(shape.num_parts):
        shape_point = shape.part(shape_point_count)

        if shape_point_count == 0:
            left = int(shape_point.x)
        elif shape_point_count == 16:
            right = int(shape_point.x)
        elif shape_point_count == 24:
            top = int(shape_point.y)
        elif shape_point_count == 8:
            bottom = int(shape_point.y)
    left = left - (right - left) // 4
    right = right + (right - left) // 4
    top = top - (bottom - top) // 3
    bottom = bottom + (bottom - top) // 8
    img = img[top:bottom,left:right]
    cv2.imwrite('{0}_crop.png'.format(file_name), img)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('not enough parms')
        exit()

    run_type = sys.argv[1]
    video_path = sys.argv[2]
    video_paths = ['dataset/movies/Aziz.mp4','dataset/movies/Derek.mp4','dataset/movies/Elle.mp4','dataset/movies/Emma.mp4','dataset/movies/Hiyane.mp4','dataset/movies/Imaizumi.mp4','dataset/movies/James.mp4','dataset/movies/Kendall.mp4','dataset/movies/Kitazumi.mp4','dataset/movies/Liza.mp4','dataset/movies/Neil.mp4','dataset/movies/Ogawa.mp4','dataset/movies/Selena.mp4','dataset/movies/Shiraishi.mp4','dataset/movies/Taylor.mp4']


    if run_type == 'detect_face':
        width = int(sys.argv[3])
        height = int(sys.argv[4])
        detect_face(video_path,width,height)
    elif run_type == 'estimate_headpose':
        for video_path in video_paths:
            estimate_headpose(video_path)
    elif run_type == 'detect_face_single':
        img_path = video_path
        detect_face_single(img_path)
    elif run_type == 'crop':
        img_path = video_path
        crop(img_path)

