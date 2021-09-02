import tensorflow as tf 
from tensorflow import keras
import pandas as pd
import numpy as np


def point():
    x = [
        'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22',
        'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'x31', 'x32', 'x33',
    ]
    y = [
        'y12', 'y13', 'y14', 'y15', 'y16', 'y17', 'y18', 'y19', 'y20', 'y21', 'y22',
        'y23', 'y24', 'y25', 'y26', 'y27', 'y28', 'y29', 'y30', 'y31', 'y32', 'y33',
    ]
    z = [
        'z12', 'z13', 'z14', 'z15', 'z16', 'z17', 'z18', 'z19', 'z20', 'z21', 'z22',
        'z23', 'z24', 'z25', 'z26', 'z27', 'z28', 'z29', 'z30', 'z31', 'z32', 'z33',
    ]
    v = [
        'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22',
        'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'v29', 'v30', 'v31', 'v32', 'v33',
    ]
    coords = [x, y, z, v]
    return coords


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


if __name__ == '__main__':

    # 0: cat_camel, 1: bridge_exercise, 2: heel_raise.

    test_file = '../datasets/numerical_coords_dataset_test2.csv'
    all_model = '../model_weights/all_model/08.31_xyzv/3_categories_pose'

    # Loads the model and training weights.
    model = keras.models.load_model(all_model)
    # print(model.summary())

    test_input_xyzv = sample2_xyzv(file=test_file)

    # print(test_input_xyzv)

    input_dict = {name: tf.convert_to_tensor([value]) for name, value in test_input_xyzv.items()}
    print(f'input_dict: \n{type(input_dict)}')

    outputs = model.predict(input_dict)

    print('-'*30)
    print(f'tatal: {outputs[0][0] + outputs[0][1] + outputs[0][2]}, {outputs[0]}')
    print(f'calss: {np.argmax(outputs[0])}, prob: {outputs[0][np.argmax(outputs[0])]}')
    # print(f'calss: {np.argmax(outputs[0])}, prob: {round(outputs[0][np.argmax(outputs[0])]*100, 4)}%')