# mediapipe_pose_classification_with_tf

Use mediapipe holistic model to get 33 landmarks of body and input to a simple custom model with tensorflow 2.X for training pose classification.
The idea is from this repo: [classify_pose_with_mediapipe](https://github.com/dawi9840/classify_pose_with_mediapipe.git).


# File description  

**dataset_create.py**- To create training datsset and testing dataset with mediapipe. With mediapipe holistic model to record the landmarks position (x, y, z, visibility) into a csv file.  

**model_train_pose.py**- Using input features which read from the training set csv file and input to Keras model for training pose classification.
With training done which can save the model and weights. Finallly, convert to TFlite model.  

**model_inference_demo.py**-  Demo the result about using TFLite model inference with mediapipe for output.




```bash

conda create --name [env_name]  python=3.8
conda activate [env_name]
pip install pandas==1.1.3
pip install numpy
pip install tensorflow-gpu==2.6.0
conda install cudnn==8.2.0.53
pip install pydot
pip install mediapipe==0.8.6.2
```
