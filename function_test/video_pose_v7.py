import cv2
import mediapipe as mp
import glob
import os
import numpy as np


class color:
    # Color difine
    purple = (245,66,230)
    blue = (245,117,66)
    red = (0, 0, 255)
    green = (0, 255, 0)


def camera_info(cap):
    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'w: {w}, h: {h}')

    while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                cv2.imshow('Raw Video Feed', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    print('Done.')
    cap.release()
    cv2.destroyAllWindows()


def mediapipe_detections(cap, out_video=None):
    ''' Make mediapipe detections with a video (or use camera). 
    Parameter 'out_video' which default is None, it means not save output video.
    If want save result video, we can specific out_video=['result_out.mp4].
    '''
    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps, frame_count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_w, cap_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'Frames Per Second: {input_fps}')
        print(f'frame count: {frame_count}')
        print(f'w: {cap_w}, h:{cap_h}')

    if out_video == None:
        pass
    else:
        output_fps = input_fps - 1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 輸出附檔名為 mp4
        out = cv2.VideoWriter(out_video, fourcc, output_fps, (cap_w, cap_h))

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers.
    mp_holistic = mp.solutions.holistic     # Mediapipe Solutions.

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        

                # Make Detections
                results = holistic.process(image)
                # print(results.face_landmarks)
                
                # Angle use example
                try:
                    landmarks = results.pose_landmarks.landmark
                    # print(f'nose_x: {landmarks[0].x}')
                    # print(f'nose_y: {landmarks[0].y}')
                    
                    l_elbow_x, l_elbow_y = landmarks[13].x * cap_w, landmarks[13].y * cap_h
                    l_shoulder_x, l_shoulder_y = landmarks[11].x * cap_w, landmarks[11].y * cap_h
                    l_hip_x, l_hip_y = landmarks[23].x * cap_w, landmarks[23].y * cap_h

                    elbow = [l_elbow_x, l_elbow_y]
                    shoulder = [l_shoulder_x, l_shoulder_y]
                    hip = [l_hip_x, l_hip_y]
                    angle = calculate_angle(elbow, shoulder, hip)
                    print(f'angle: {angle}')
                except:
                    print('Cannnot get the landmarks, plz check again!')

                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Pose Detections
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=color.blue, thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=color.purple, thickness=2, circle_radius=2)
                )

                if out_video == None:
                    pass
                else:
                    out.write(image)

                cv2.imshow('Raw Video Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
            
    print('Done.')
    cap.release()
    cv2.destroyAllWindows()


def extract_images(cap, str_class):
    '''Need a floder ./resource/extract_images/[str_class] 
        to save extract images from a video.'''

    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')

    count = 0
    while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imwrite('./resource/extract_images/'+str_class+'/image_{0:0>3}.jpg'.format(count), frame) # Save frame as JPEG file.
                print(f'save frame: {count}')
                count += 1
                cv2.imshow('extract_video', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break

    print('Extract images done!')
    cap.release()
    cv2.destroyAllWindows()


def extract_mediapipe_images(cap, str_class):
    # Color difine
    color_pose1 = (245,117,66)
    color_pose2 = (245,66,230)

    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers.
    mp_holistic = mp.solutions.holistic     # Mediapipe Solutions.

    count = 0
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        

                # Make Detections
                results = holistic.process(image)
                # print(results.face_landmarks)
                # landmarks = results.pose_landmarks.landmark
                # print(f'nose_x: {landmarks[0].x}')
                # print(f'nose_y: {landmarks[0].y}')

                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Pose Detections
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=color_pose1, thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=color_pose2, thickness=2, circle_radius=2)
                )

                cv2.imwrite('./resource/extract_images/'+str_class+'/image_{0:0>3}.jpg'.format(count), image) # Save frame as JPEG file.
                print(f'save frame: {count}')
                count += 1
                cv2.imshow('Raw Video Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
            
    print('Done.')
    cap.release()
    cv2.destroyAllWindows()  


def img_to_video(input_imgs_floder, output_video):
    ''''path = './*.jpg', output_video = file name.'''
    img_array = []

    for filename in glob.glob(input_imgs_floder):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    print('create done!')


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


if __name__ == '__main__':

    video_path = [
        'functional_reach_8s.mp4',
        'FR_t1.mp4',
    ]

    # directory = 'half_squat'

    # # Parent Directory path
    # parent_dir = './extract_images/'
    # path = os.path.join(parent_dir, directory)
    # os.mkdir(path)  # Create the directory in path.
    # extract_images(cap=cv2.VideoCapture(video_path[5]), str_class=directory)
    
    functional_reach = 'FR_t1_mp'
    # extract_mediapipe_images(cap=cv2.VideoCapture(video_path[1]), str_class=functional_reach)
    # extract_images(cap=cv2.VideoCapture(video_path[1]), str_class=functional_reach)

    # mediapipe_detections(cap=cv2.VideoCapture(video_path[1]), out_video='test.mp4')
    mediapipe_detections(cap=cv2.VideoCapture(video_path[1]))
