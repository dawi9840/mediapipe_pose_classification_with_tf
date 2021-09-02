from IPython.core.events import post_execute
import tensorflow as tf 
from tensorflow import keras
import cv2 
import mediapipe as mp
import pandas as pd
from model_train_pose import point
import numpy as np


def display_tflite_classify_pose(cap, model):
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

                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    # Concate rows
                    row = pose_row
                    # print(f'type: {type(row)}, \n{row}')

                    # point[11] - point[32] input to tflite model.
                    coords = point()
                    specify_float = 8

                    dict_p12_to_p33 = {
                        # x12 to x33
                        coords[0][0]:round(row[44], specify_float),
                        coords[0][1]:round(row[48], specify_float),
                        coords[0][2]:round(row[52], specify_float),
                        coords[0][3]:round(row[56], specify_float),
                        coords[0][4]:round(row[60], specify_float),
                        coords[0][5]:round(row[64], specify_float),
                        coords[0][6]:round(row[68], specify_float),
                        coords[0][7]:round(row[72], specify_float),
                        coords[0][8]:round(row[76], specify_float),
                        coords[0][9]:round(row[80], specify_float),
                        coords[0][10]:round(row[84], specify_float),
                        coords[0][11]:round(row[88], specify_float),
                        coords[0][12]:round(row[92], specify_float),
                        coords[0][13]:round(row[96], specify_float),
                        coords[0][14]:round(row[100], specify_float),
                        coords[0][15]:round(row[104], specify_float),
                        coords[0][16]:round(row[108], specify_float),
                        coords[0][17]:round(row[112], specify_float),
                        coords[0][18]:round(row[116], specify_float),
                        coords[0][19]:round(row[120], specify_float),
                        coords[0][20]:round(row[124], specify_float),
                        coords[0][21]:round(row[128], specify_float),

                        # y12 to y33
                        coords[1][0]:round(row[45], specify_float),
                        coords[1][1]:round(row[49], specify_float),
                        coords[1][2]:round(row[53], specify_float),
                        coords[1][3]:round(row[57], specify_float),
                        coords[1][4]:round(row[61], specify_float),
                        coords[1][5]:round(row[65], specify_float),
                        coords[1][6]:round(row[69], specify_float),
                        coords[1][7]:round(row[73], specify_float),
                        coords[1][8]:round(row[77], specify_float),
                        coords[1][9]:round(row[81], specify_float),
                        coords[1][10]:round(row[85], specify_float),
                        coords[1][11]:round(row[89], specify_float),
                        coords[1][12]:round(row[93], specify_float),
                        coords[1][13]:round(row[97], specify_float),
                        coords[1][14]:round(row[101], specify_float),
                        coords[1][15]:round(row[105], specify_float),
                        coords[1][16]:round(row[109], specify_float),
                        coords[1][17]:round(row[113], specify_float),
                        coords[1][18]:round(row[117], specify_float),
                        coords[1][19]:round(row[121], specify_float),
                        coords[1][20]:round(row[125], specify_float),
                        coords[1][21]:round(row[129], specify_float),

                        # z12 to z33
                        coords[2][0]:round(row[46], specify_float),
                        coords[2][1]:round(row[50], specify_float),
                        coords[2][2]:round(row[54], specify_float),
                        coords[2][3]:round(row[58], specify_float),
                        coords[2][4]:round(row[62], specify_float),
                        coords[2][5]:round(row[66], specify_float),
                        coords[2][6]:round(row[70], specify_float),
                        coords[2][7]:round(row[74], specify_float),
                        coords[2][8]:round(row[78], specify_float),
                        coords[2][9]:round(row[82], specify_float),
                        coords[2][10]:round(row[86], specify_float),
                        coords[2][11]:round(row[90], specify_float),
                        coords[2][12]:round(row[94], specify_float),
                        coords[2][13]:round(row[98], specify_float),
                        coords[2][14]:round(row[102], specify_float),
                        coords[2][15]:round(row[106], specify_float),
                        coords[2][16]:round(row[110], specify_float),
                        coords[2][17]:round(row[114], specify_float),
                        coords[2][18]:round(row[118], specify_float),
                        coords[2][19]:round(row[122], specify_float),
                        coords[2][20]:round(row[126], specify_float),
                        coords[2][21]:round(row[130], specify_float),

                        # v12 to v33
                        coords[3][0]:round(row[47], specify_float),
                        coords[3][1]:round(row[51], specify_float),
                        coords[3][2]:round(row[55], specify_float),
                        coords[3][3]:round(row[59], specify_float),
                        coords[3][4]:round(row[63], specify_float),
                        coords[3][5]:round(row[67], specify_float),
                        coords[3][6]:round(row[71], specify_float),
                        coords[3][7]:round(row[75], specify_float),
                        coords[3][8]:round(row[79], specify_float),
                        coords[3][9]:round(row[83], specify_float),
                        coords[3][10]:round(row[87], specify_float),
                        coords[3][11]:round(row[91], specify_float),
                        coords[3][12]:round(row[95], specify_float),
                        coords[3][13]:round(row[99], specify_float),
                        coords[3][14]:round(row[103], specify_float),
                        coords[3][15]:round(row[107], specify_float),
                        coords[3][16]:round(row[111], specify_float),
                        coords[3][17]:round(row[115], specify_float),
                        coords[3][18]:round(row[119], specify_float),
                        coords[3][19]:round(row[123], specify_float),
                        coords[3][20]:round(row[127], specify_float),
                        coords[3][21]:round(row[131], specify_float),
                    }
                    input_dict = {name: np.expand_dims(np.array(value, dtype=np.float32), axis=0) for name, value in dict_p12_to_p33.items()}
                    
                    # Make Detections
                    # 0: cat_camel, 1: bridge_exercise, 2: heel_raise.
                    result = tflite_inference(input=input_dict, model=model)
                    body_language_class = np.argmax(result)
                    # body_language_prob = round(result[np.argmax(result)], 2)*100
                    body_language_prob = result[np.argmax(result)]

                    if str(body_language_class) == '0':
                        pose_class = 'Cat camel' 
                    elif str(body_language_class) == '1':
                        pose_class = 'Bridge exercise'
                    else:
                        pose_class = 'Heel raise'
                    
                    # print(f'calss: {body_language_class}, prob: {body_language_prob}')

                    # Show pose category near the ear.
                    coords = tuple(np.multiply(
                        np.array(
                            (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                        [1280,480]
                    ).astype(int))

                    # cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度).
                    # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類).
                    cv2.rectangle(image, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+200, coords[1]-30), 
                                (245, 117, 16), -1)
                    cv2.putText(image, pose_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Get status box
                    cv2.rectangle(image, (10,0), (310, 55), (0, 0, 0), -1)

                    # Display Class
                    cv2.putText(
                        image, 
                        'CLASS: ', (15, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (255, 255, 0), 1, cv2.LINE_AA
                    )
                    cv2.putText(
                        image, 
                        pose_class, (120, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA
                    )

                    # Display Probability
                    cv2.putText(
                        image, 
                        'PROB: ', (15, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (255, 255, 0), 1, cv2.LINE_AA
                    )
                    cv2.putText(
                        image, 
                        str(body_language_prob), (120, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA
                    )

                except:
                    pass

                cv2.imshow('Raw Webcam Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            
            else:
                break

    print('Done!')
    cap.release()
    cv2.destroyAllWindows()


def save_tflite_classify_pose(cap, model, result_out):
    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_fps = input_fps - 1
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'video_w: {w}, video_h: {h}')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 輸出附檔名為 mp4. 
    out = cv2.VideoWriter(result_out, fourcc, output_fps, (w, h))

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

                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    # Concate rows
                    row = pose_row
                    # print(f'type: {type(row)}, \n{row}')

                    # point[11] - point[32] input to tflite model.
                    coords = point()
                    specify_float = 8

                    dict_p12_to_p33 = {
                        # x12 to x33
                        coords[0][0]:round(row[44], specify_float),
                        coords[0][1]:round(row[48], specify_float),
                        coords[0][2]:round(row[52], specify_float),
                        coords[0][3]:round(row[56], specify_float),
                        coords[0][4]:round(row[60], specify_float),
                        coords[0][5]:round(row[64], specify_float),
                        coords[0][6]:round(row[68], specify_float),
                        coords[0][7]:round(row[72], specify_float),
                        coords[0][8]:round(row[76], specify_float),
                        coords[0][9]:round(row[80], specify_float),
                        coords[0][10]:round(row[84], specify_float),
                        coords[0][11]:round(row[88], specify_float),
                        coords[0][12]:round(row[92], specify_float),
                        coords[0][13]:round(row[96], specify_float),
                        coords[0][14]:round(row[100], specify_float),
                        coords[0][15]:round(row[104], specify_float),
                        coords[0][16]:round(row[108], specify_float),
                        coords[0][17]:round(row[112], specify_float),
                        coords[0][18]:round(row[116], specify_float),
                        coords[0][19]:round(row[120], specify_float),
                        coords[0][20]:round(row[124], specify_float),
                        coords[0][21]:round(row[128], specify_float),

                        # y12 to y33
                        coords[1][0]:round(row[45], specify_float),
                        coords[1][1]:round(row[49], specify_float),
                        coords[1][2]:round(row[53], specify_float),
                        coords[1][3]:round(row[57], specify_float),
                        coords[1][4]:round(row[61], specify_float),
                        coords[1][5]:round(row[65], specify_float),
                        coords[1][6]:round(row[69], specify_float),
                        coords[1][7]:round(row[73], specify_float),
                        coords[1][8]:round(row[77], specify_float),
                        coords[1][9]:round(row[81], specify_float),
                        coords[1][10]:round(row[85], specify_float),
                        coords[1][11]:round(row[89], specify_float),
                        coords[1][12]:round(row[93], specify_float),
                        coords[1][13]:round(row[97], specify_float),
                        coords[1][14]:round(row[101], specify_float),
                        coords[1][15]:round(row[105], specify_float),
                        coords[1][16]:round(row[109], specify_float),
                        coords[1][17]:round(row[113], specify_float),
                        coords[1][18]:round(row[117], specify_float),
                        coords[1][19]:round(row[121], specify_float),
                        coords[1][20]:round(row[125], specify_float),
                        coords[1][21]:round(row[129], specify_float),

                        # z12 to z33
                        coords[2][0]:round(row[46], specify_float),
                        coords[2][1]:round(row[50], specify_float),
                        coords[2][2]:round(row[54], specify_float),
                        coords[2][3]:round(row[58], specify_float),
                        coords[2][4]:round(row[62], specify_float),
                        coords[2][5]:round(row[66], specify_float),
                        coords[2][6]:round(row[70], specify_float),
                        coords[2][7]:round(row[74], specify_float),
                        coords[2][8]:round(row[78], specify_float),
                        coords[2][9]:round(row[82], specify_float),
                        coords[2][10]:round(row[86], specify_float),
                        coords[2][11]:round(row[90], specify_float),
                        coords[2][12]:round(row[94], specify_float),
                        coords[2][13]:round(row[98], specify_float),
                        coords[2][14]:round(row[102], specify_float),
                        coords[2][15]:round(row[106], specify_float),
                        coords[2][16]:round(row[110], specify_float),
                        coords[2][17]:round(row[114], specify_float),
                        coords[2][18]:round(row[118], specify_float),
                        coords[2][19]:round(row[122], specify_float),
                        coords[2][20]:round(row[126], specify_float),
                        coords[2][21]:round(row[130], specify_float),

                        # v12 to v33
                        coords[3][0]:round(row[47], specify_float),
                        coords[3][1]:round(row[51], specify_float),
                        coords[3][2]:round(row[55], specify_float),
                        coords[3][3]:round(row[59], specify_float),
                        coords[3][4]:round(row[63], specify_float),
                        coords[3][5]:round(row[67], specify_float),
                        coords[3][6]:round(row[71], specify_float),
                        coords[3][7]:round(row[75], specify_float),
                        coords[3][8]:round(row[79], specify_float),
                        coords[3][9]:round(row[83], specify_float),
                        coords[3][10]:round(row[87], specify_float),
                        coords[3][11]:round(row[91], specify_float),
                        coords[3][12]:round(row[95], specify_float),
                        coords[3][13]:round(row[99], specify_float),
                        coords[3][14]:round(row[103], specify_float),
                        coords[3][15]:round(row[107], specify_float),
                        coords[3][16]:round(row[111], specify_float),
                        coords[3][17]:round(row[115], specify_float),
                        coords[3][18]:round(row[119], specify_float),
                        coords[3][19]:round(row[123], specify_float),
                        coords[3][20]:round(row[127], specify_float),
                        coords[3][21]:round(row[131], specify_float),
                    }
                    input_dict = {name: np.expand_dims(np.array(value, dtype=np.float32), axis=0) for name, value in dict_p12_to_p33.items()}
                    
                    # Make Detections
                    # 0: cat_camel, 1: bridge_exercise, 2: heel_raise.
                    result = tflite_inference(input=input_dict, model=model)
                    body_language_class = np.argmax(result)
                    # body_language_prob = round(result[np.argmax(result)], 2)*100
                    body_language_prob = result[np.argmax(result)]

                    if str(body_language_class) == '0':
                        pose_class = 'Cat camel' 
                    elif str(body_language_class) == '1':
                        pose_class = 'Bridge exercise'
                    else:
                        pose_class = 'Heel raise'
                    
                    # print(f'calss: {body_language_class}, prob: {body_language_prob}')

                    # Show pose category near the ear.
                    coords = tuple(np.multiply(
                        np.array(
                            (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                        [1280,480]
                    ).astype(int))

                    # cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度).
                    # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類).
                    cv2.rectangle(image, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+200, coords[1]-30), 
                                (245, 117, 16), -1)
                    cv2.putText(image, pose_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Get status box
                    cv2.rectangle(image, (10,0), (310, 55), (0, 0, 0), -1)

                    # Display Class
                    cv2.putText(
                        image, 
                        'CLASS: ', (15, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (255, 255, 0), 1, cv2.LINE_AA
                    )
                    cv2.putText(
                        image, 
                        pose_class, (120, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA
                    )

                    # Display Probability
                    cv2.putText(
                        image, 
                        'PROB: ', (15, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (255, 255, 0), 1, cv2.LINE_AA
                    )
                    cv2.putText(
                        image, 
                        str(body_language_prob), (120, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA
                    )

                except:
                    pass

                out.write(image)
                cv2.imshow('Pose classification result', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            
            else:
                break

    print('Save done!')
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def sample1_xyz(file):
    df = test_data_xyz(file)
    # print(f'df: \n{df}')
    
    coords = point()
    specify_float = 8
    print(round(df.iat[0, 66], specify_float))
    # bb=df['z33']
    # print(f'z33: {bb}')
    
    sample = {
        # x12 to x33
        coords[0][0]:round(df.iat[0, 1], specify_float),
        coords[0][1]:round(df.iat[0, 4], specify_float),
        coords[0][2]:round(df.iat[0, 7], specify_float),
        coords[0][3]:round(df.iat[0, 10], specify_float),
        coords[0][4]:round(df.iat[0, 13], specify_float),
        coords[0][5]:round(df.iat[0, 16], specify_float),
        coords[0][6]:round(df.iat[0, 19], specify_float),
        coords[0][7]:round(df.iat[0, 22], specify_float),
        coords[0][8]:round(df.iat[0, 25], specify_float),
        coords[0][9]:round(df.iat[0, 28], specify_float),
        coords[0][10]:round(df.iat[0, 31], specify_float),
        coords[0][11]:round(df.iat[0, 34], specify_float),
        coords[0][12]:round(df.iat[0, 37], specify_float),
        coords[0][13]:round(df.iat[0, 40], specify_float),
        coords[0][14]:round(df.iat[0, 43], specify_float),
        coords[0][15]:round(df.iat[0, 46], specify_float),
        coords[0][16]:round(df.iat[0, 49], specify_float),
        coords[0][17]:round(df.iat[0, 52], specify_float),
        coords[0][18]:round(df.iat[0, 55], specify_float),
        coords[0][19]:round(df.iat[0, 58], specify_float),
        coords[0][20]:round(df.iat[0, 61], specify_float),
        coords[0][21]:round(df.iat[0, 64], specify_float),

        # y12 to y33
        coords[1][0]:round(df.iat[0, 2], specify_float),
        coords[1][1]:round(df.iat[0, 5], specify_float),
        coords[1][2]:round(df.iat[0, 8], specify_float),
        coords[1][3]:round(df.iat[0, 11], specify_float),
        coords[1][4]:round(df.iat[0, 14], specify_float),
        coords[1][5]:round(df.iat[0, 17], specify_float),
        coords[1][6]:round(df.iat[0, 20], specify_float),
        coords[1][7]:round(df.iat[0, 23], specify_float),
        coords[1][8]:round(df.iat[0, 26], specify_float),
        coords[1][9]:round(df.iat[0, 39], specify_float),
        coords[1][10]:round(df.iat[0, 32], specify_float),
        coords[1][11]:round(df.iat[0, 35], specify_float),
        coords[1][12]:round(df.iat[0, 38], specify_float),
        coords[1][13]:round(df.iat[0, 41], specify_float),
        coords[1][14]:round(df.iat[0, 44], specify_float),
        coords[1][15]:round(df.iat[0, 47], specify_float),
        coords[1][16]:round(df.iat[0, 50], specify_float),
        coords[1][17]:round(df.iat[0, 53], specify_float),
        coords[1][18]:round(df.iat[0, 56], specify_float),
        coords[1][19]:round(df.iat[0, 59], specify_float),
        coords[1][20]:round(df.iat[0, 62], specify_float),
        coords[1][21]:round(df.iat[0, 65], specify_float),

        # z12 to z33
        coords[2][0]:round(df.iat[0, 3], specify_float),
        coords[2][1]:round(df.iat[0, 6], specify_float),
        coords[2][2]:round(df.iat[0, 9], specify_float),
        coords[2][3]:round(df.iat[0, 12], specify_float),
        coords[2][4]:round(df.iat[0, 15], specify_float),
        coords[2][5]:round(df.iat[0, 18], specify_float),
        coords[2][6]:round(df.iat[0, 21], specify_float),
        coords[2][7]:round(df.iat[0, 24], specify_float),
        coords[2][8]:round(df.iat[0, 27], specify_float),
        coords[2][9]:round(df.iat[0, 30], specify_float),
        coords[2][10]:round(df.iat[0, 33], specify_float),
        coords[2][11]:round(df.iat[0, 36], specify_float),
        coords[2][12]:round(df.iat[0, 39], specify_float),
        coords[2][13]:round(df.iat[0, 42], specify_float),
        coords[2][14]:round(df.iat[0, 45], specify_float),
        coords[2][15]:round(df.iat[0, 48], specify_float),
        coords[2][16]:round(df.iat[0, 51], specify_float),
        coords[2][17]:round(df.iat[0, 54], specify_float),
        coords[2][18]:round(df.iat[0, 57], specify_float),
        coords[2][19]:round(df.iat[0, 60], specify_float),
        coords[2][20]:round(df.iat[0, 63], specify_float),
        coords[2][21]:round(df.iat[0, 66], specify_float),
    }

    return sample


def sample2_xyzv(file):
    df = test_data_xyzv(file)
    # print(f'df: \n{df}')
    
    coords = point()
    specify_float = 8
    
    sample = {
        # x12 to x33
        coords[0][0]:round(df.iat[0, 1], specify_float),
        coords[0][1]:round(df.iat[0, 5], specify_float),
        coords[0][2]:round(df.iat[0, 9], specify_float),
        coords[0][3]:round(df.iat[0, 13], specify_float),
        coords[0][4]:round(df.iat[0, 17], specify_float),
        coords[0][5]:round(df.iat[0, 21], specify_float),
        coords[0][6]:round(df.iat[0, 25], specify_float),
        coords[0][7]:round(df.iat[0, 29], specify_float),
        coords[0][8]:round(df.iat[0, 33], specify_float),
        coords[0][9]:round(df.iat[0, 37], specify_float),
        coords[0][10]:round(df.iat[0, 41], specify_float),
        coords[0][11]:round(df.iat[0, 45], specify_float),
        coords[0][12]:round(df.iat[0, 49], specify_float),
        coords[0][13]:round(df.iat[0, 53], specify_float),
        coords[0][14]:round(df.iat[0, 57], specify_float),
        coords[0][15]:round(df.iat[0, 61], specify_float),
        coords[0][16]:round(df.iat[0, 65], specify_float),
        coords[0][17]:round(df.iat[0, 69], specify_float),
        coords[0][18]:round(df.iat[0, 73], specify_float),
        coords[0][19]:round(df.iat[0, 77], specify_float),
        coords[0][20]:round(df.iat[0, 81], specify_float),
        coords[0][21]:round(df.iat[0, 85], specify_float),

        # y12 to y33
        coords[1][0]:round(df.iat[0, 2], specify_float),
        coords[1][1]:round(df.iat[0, 6], specify_float),
        coords[1][2]:round(df.iat[0, 10], specify_float),
        coords[1][3]:round(df.iat[0, 14], specify_float),
        coords[1][4]:round(df.iat[0, 18], specify_float),
        coords[1][5]:round(df.iat[0, 22], specify_float),
        coords[1][6]:round(df.iat[0, 26], specify_float),
        coords[1][7]:round(df.iat[0, 30], specify_float),
        coords[1][8]:round(df.iat[0, 34], specify_float),
        coords[1][9]:round(df.iat[0, 38], specify_float),
        coords[1][10]:round(df.iat[0, 42], specify_float),
        coords[1][11]:round(df.iat[0, 46], specify_float),
        coords[1][12]:round(df.iat[0, 50], specify_float),
        coords[1][13]:round(df.iat[0, 51], specify_float),
        coords[1][14]:round(df.iat[0, 58], specify_float),
        coords[1][15]:round(df.iat[0, 62], specify_float),
        coords[1][16]:round(df.iat[0, 66], specify_float),
        coords[1][17]:round(df.iat[0, 70], specify_float),
        coords[1][18]:round(df.iat[0, 74], specify_float),
        coords[1][19]:round(df.iat[0, 78], specify_float),
        coords[1][20]:round(df.iat[0, 82], specify_float),
        coords[1][21]:round(df.iat[0, 86], specify_float),

        # z12 to z33
        coords[2][0]:round(df.iat[0, 3], specify_float),
        coords[2][1]:round(df.iat[0, 7], specify_float),
        coords[2][2]:round(df.iat[0, 11], specify_float),
        coords[2][3]:round(df.iat[0, 15], specify_float),
        coords[2][4]:round(df.iat[0, 19], specify_float),
        coords[2][5]:round(df.iat[0, 23], specify_float),
        coords[2][6]:round(df.iat[0, 27], specify_float),
        coords[2][7]:round(df.iat[0, 31], specify_float),
        coords[2][8]:round(df.iat[0, 35], specify_float),
        coords[2][9]:round(df.iat[0, 39], specify_float),
        coords[2][10]:round(df.iat[0, 43], specify_float),
        coords[2][11]:round(df.iat[0, 47], specify_float),
        coords[2][12]:round(df.iat[0, 51], specify_float),
        coords[2][13]:round(df.iat[0, 55], specify_float),
        coords[2][14]:round(df.iat[0, 59], specify_float),
        coords[2][15]:round(df.iat[0, 63], specify_float),
        coords[2][16]:round(df.iat[0, 67], specify_float),
        coords[2][17]:round(df.iat[0, 71], specify_float),
        coords[2][18]:round(df.iat[0, 75], specify_float),
        coords[2][19]:round(df.iat[0, 79], specify_float),
        coords[2][20]:round(df.iat[0, 83], specify_float),
        coords[2][21]:round(df.iat[0, 87], specify_float),

        # v12 to v33
        coords[3][0]:round(df.iat[0, 4], specify_float),
        coords[3][1]:round(df.iat[0, 8], specify_float),
        coords[3][2]:round(df.iat[0, 12], specify_float),
        coords[3][3]:round(df.iat[0, 16], specify_float),
        coords[3][4]:round(df.iat[0, 20], specify_float),
        coords[3][5]:round(df.iat[0, 24], specify_float),
        coords[3][6]:round(df.iat[0, 28], specify_float),
        coords[3][7]:round(df.iat[0, 32], specify_float),
        coords[3][8]:round(df.iat[0, 36], specify_float),
        coords[3][9]:round(df.iat[0, 40], specify_float),
        coords[3][10]:round(df.iat[0, 44], specify_float),
        coords[3][11]:round(df.iat[0, 48], specify_float),
        coords[3][12]:round(df.iat[0, 52], specify_float),
        coords[3][13]:round(df.iat[0, 56], specify_float),
        coords[3][14]:round(df.iat[0, 60], specify_float),
        coords[3][15]:round(df.iat[0, 64], specify_float),
        coords[3][16]:round(df.iat[0, 68], specify_float),
        coords[3][17]:round(df.iat[0, 72], specify_float),
        coords[3][18]:round(df.iat[0, 76], specify_float),
        coords[3][19]:round(df.iat[0, 80], specify_float),
        coords[3][20]:round(df.iat[0, 84], specify_float),
        coords[3][21]:round(df.iat[0, 88], specify_float),
    }

    return sample


def test_data_xyz(file):
    df = pd.read_csv(file)
    df2 = df.copy()

    columns_removed = [
        'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11',
        'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11',
        'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11',
        'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11',

        'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21',
        'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31',
        'v32', 'v33',
    ]

    df2 = df2.drop(columns_removed, axis = 'columns')

    # Get A row from A to B.
    get_a_row_value = df2.iloc[4:5]
    return get_a_row_value


def test_data_xyzv(file):
    df = pd.read_csv(file)
    df2 = df.copy()

    columns_removed = [
        'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11',
        'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11',
        'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11',
        'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11',
    ]

    df2 = df2.drop(columns_removed, axis = 'columns')

    # Get row from 4 to 5.
    get_a_row_value = df2.iloc[4:5]
    return get_a_row_value


def desktop_model_inference(input, model):
    '''# Loads the model and training weights for desktop model and test inference.'''

    model = keras.models.load_model(model)
    # print(model.summary())

    # print(f'input: \n{type(input)}')
    outputs = model.predict(input)

    # print('*'*30) 
    # print(f'tatal: {outputs[0][0] + outputs[0][1] + outputs[0][2]}')
    # print(f'calss: {np.argmax(outputs[0])}, prob: {outputs[0][np.argmax(outputs[0])]}')
    # print(f'calss: {np.argmax(outputs[0])}, prob: {round(outputs[0][np.argmax(outputs[0])]*100, 5)}%')
    return outputs


def tflite_inference(input, model):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    ### Get input and output tensors. ###
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # tf.print('input_details[0]:\n', input_details[0])
    # tf.print('output_details:\n', output_details)
    # print('-'*30)
    # print(f'input_index: {input_details[0]["index"]}')
    # print(f'output_index: {output_details[0]["index"]}')
    # print(f'input_shape: {input_details[0]["shape"]}')
    # print('-'*30)

    ### Verify the TensorFlow Lite model. ###
    for i, (name, value) in enumerate(input.items()):

        input_value = np.expand_dims(value, axis=1)
        # print(f'index: {i}, type: {type(input_value)}, shape:{input_value.shape}')
        # print(input_value)
        interpreter.set_tensor(input_details[i]['index'], input_value)
        interpreter.invoke()

    output = interpreter.tensor(output_details[0]['index'])()[0]

    # print(f'prob: {output}, type: {type(output)}')
    # print(f'calss: {np.argmax(output)}, prob: {output[np.argmax(output)]}')
    return output


def test_model_inference(input_csv, pc_model, tflite_model):
    '''Input a row data(point12 to point33 features) to desktop model and tflite model from test csv file for inference.'''
    test_input_xyzv = sample2_xyzv(file=input_csv)
    input_dict = {name: np.expand_dims(np.array(value, dtype=np.float32), axis=0) for name, value in test_input_xyzv.items()}

    result_pc_model = desktop_model_inference(input=input_dict, model=pc_model)
    result_tflite = tflite_inference(input=input_dict, model=tflite_model)

    print('-'*30)
    print(f'[Desktop model inference]\nprob: {result_pc_model}, type: {type(result_pc_model)}')
    print(f'calss: {np.argmax(result_pc_model[0])}, prob: {result_pc_model[0][np.argmax(result_pc_model[0])]}')
    print('-'*30)
    print(f'[TFLite model inference]\nprob: {result_tflite}, type: {type(result_tflite)}')
    print(f'calss: {np.argmax(result_tflite)}, prob: {result_tflite[np.argmax(result_tflite)]}')
    print('-'*30)


if __name__ == '__main__':

    # 0: cat_camel, 1: bridge_exercise, 2: heel_raise
    video_file_name = 'cat_camel' + '2'
    output_video = './' + video_file_name + '_out.mp4'
    video_path = './video/'+ video_file_name + '.mp4'
    
    test_file = './datasets/numerical_coords_dataset_test2.csv'
    all_model = './model_weights/all_model/08.31_xyzv/3_categories_pose'
    tflite_model = './tflite_model/model.tflite'

    cap = cv2.VideoCapture(video_path)

    # test_model_inference(input_csv=test_file, pc_model=all_model, tflite_model=tflite_model)

    display_tflite_classify_pose(cap, model=tflite_model)
    # save_tflite_classify_pose(cap, model=tflite_model, result_out=output_video)