import cv2
import os
import csv
import numpy as np
import mediapipe as mp
import pandas as pd 


def mediapipe_detections(cap):
    '''# 1. Make Some Detections with a video # '''
    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')
    
    # Color difine
    color_face1 = (80,110,10)
    color_face2 = (80,256,121)
    color_r_hand1 = (80,22,10)
    color_r_hand2 = (80,44,121)
    color_l_hand1 = (121,22,76)
    color_l_hand2 = (121,44,250)
    color_pose1 = (245,117,66)
    color_pose2 = (245,66,230)

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

                # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                '''# 1. Draw face landmarks
                mp_drawing.draw_landmarks(
                    image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=color_face1, thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=color_face2, thickness=1, circle_radius=1)
                )

                # 2. Right hand
                mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=color_r_hand1, thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=color_r_hand2, thickness=2, circle_radius=2)
                )

                # 3. Left Hand
                mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=color_l_hand1, thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=color_l_hand2, thickness=2, circle_radius=2)
                )'''

                # 4. Pose Detections
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=color_pose1, thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=color_pose2, thickness=2, circle_radius=2)
                )

                cv2.imshow('Raw Video Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
        
    print('Done.')
    cap.release()
    cv2.destroyAllWindows()


def create_pose_csv(cap, create_csv):
    ''' Create pose detections csv  with a video.'''
    if (cap.isOpened() == False):
        print("\nError opening the video file.")
        return
    else:
        pass
        # input_fps = cap.get(cv2.CAP_PROP_FPS)
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(f'Frames per second: {input_fps}')
        # print(f'Frame count: {frame_count}')

    # Color difine
    color_face1 = (80,110,10)
    color_face2 = (80,256,121)
    color_r_hand1 = (80,22,10)
    color_r_hand2 = (80,44,121)
    color_l_hand1 = (121,22,76)
    color_l_hand2 = (121,44,250)
    color_pose1 = (245,117,66)
    color_pose2 = (245,66,230)

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

                # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                '''# 1. Draw face landmarks
                mp_drawing.draw_landmarks(
                    image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=color_face1, thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=color_face2, thickness=1, circle_radius=1)
                )

                # 2. Right hand
                mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=color_r_hand1, thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=color_r_hand2, thickness=2, circle_radius=2)
                )

                # 3. Left Hand
                mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=color_l_hand1, thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=color_l_hand2, thickness=2, circle_radius=2)
                )'''

                # 4. Pose Detections
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=color_pose1, thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=color_pose2, thickness=2, circle_radius=2)
                )

                try:
                    num_coords = len(results.pose_landmarks.landmark) # num_coords: 33

                    landmarks = ['class'] # Create first rows data.
                    for val in range(1, num_coords+1):
                        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
                    
                    # E.g., (pose+face)2005=1+501*4, (pose+r_hand)217=1+54*4, 133=1+33*4
                    # print(f'len(landmarks): {len(landmarks)}')

                    # Define first class rows in csv file.
                    with open(create_csv, mode='w', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(landmarks)
                except:
                    pass

                cv2.imshow('Raw Video Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break

    print(f'\nCreate {dataset_csv_file} done! \n\nNow you can run again.')
    cap.release()
    cv2.destroyAllWindows()


def add_record_coordinates(cap, class_name, export_csv):
    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')

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

                # Recolor image back to BGR for rendering
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                '''# 1. Draw face landmarks
                mp_drawing.draw_landmarks(
                    image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                )

                # 2. Right hand
                mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                )

                # 3. Left Hand
                mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                )'''

                # 4. Pose Detections
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    # Extract Face landmarks
                    # face = results.face_landmarks.landmark
                    # face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                    
                    # Concate rows
                    # row = pose_row+face_row
                    row = pose_row

                    # Append class name.
                    row.insert(0, class_name)

                    # Export to CSV
                    with open(export_csv, mode='a', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(row) 

                except:
                    pass

                cv2.imshow('Raw Webcam Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
            
    print('Add done!\n -------------------')
    cap.release()
    cv2.destroyAllWindows()
    # check_csv_contents(file=export_csv)


def check_csv_contents(file):
    df = pd.read_csv(file)
    print(f'Top5 datas: \n{df.head()}')
    print(f'Last5 datas: \n{df.tail()}')


if __name__ == '__main__':
    
    n = 0

    # 0: cat_camel, 1: bridge_exercise, 2: heel_raise
    category = [0, 1, 2]
    video_file_name = 'cat_camel' + '1'
    
    # Add n categories of pose.
    add_class = category[n]

    # Can create train dataset or test dataset.
    dataset_csv_file = './dataset/numerical_coords_dataset_test'+ str(n) +'.csv'

    video_path = "./video/" + video_file_name +".mp4"
    output_video = video_file_name + "_out.mp4"

    cap = cv2.VideoCapture(video_path)

    if os.path.isfile(dataset_csv_file):
        print (f'{dataset_csv_file}: Exist.')
        print(f'Add class: {add_class} \n-----------------')

        add_record_coordinates(cap=cap, class_name=add_class, export_csv=dataset_csv_file)
    else:
        print (f'{dataset_csv_file}: Not exist.')
        print('\nInitiate creating a csv file....\n')

        create_pose_csv(cap, create_csv=dataset_csv_file)