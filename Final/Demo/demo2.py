import threading
import string
import language_tool_python
import cv2
import numpy as np
# from tensorflow.keras.models import load_model
# import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, GlobalAveragePooling2D, TimeDistributed, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def create_model_wlasl20c(frames, width, height, channels, output):
    """
    Create the keras model.

    :param frames: frame number of the sequence.
    :param width: width of the image.
    :param height: height of the image.
    :param channels: 3 for RGB, 1 for B/W images.
    :param output: number of neurons for classification.
    :return: the keras model.
    """
    model = Sequential([
        # ConvNet
        TimeDistributed(
            MobileNetV2(weights='imagenet', include_top=False,
                        input_shape=[height, width, channels]),
            input_shape=[frames, height, width, channels]
        ),
        TimeDistributed(GlobalAveragePooling2D()),

        # GRUs
        GRU(256, return_sequences=True),
        BatchNormalization(),
        GRU(256),

        # Feedforward
        Dense(units=64, activation='relu'),
        Dropout(0.65),
        Dense(units=32, activation='relu'),
        Dropout(0.65),
        Dense(units=output, activation='softmax')
    ])

    return model

def create_model_wlasl100(frames, width, height, channels, output):
    """
    Create the keras model.

    :param frames: frame number of the sequence.
    :param width: width of the image.
    :param height: height of the image.
    :param channels: 3 for RGB, 1 for B/W images.
    :param output: number of neurons for classification.
    :return: the keras model.
    """
    model = Sequential([
        # ConvNet
        TimeDistributed(
            MobileNetV2(weights='imagenet', include_top=False,
                        input_shape=[height, width, channels]),
            input_shape=[frames, height, width, channels]
        ),
        TimeDistributed(GlobalAveragePooling2D()),

        # GRUs
        GRU(256, return_sequences=True),
        BatchNormalization(),
        GRU(256),

        # Feedforward
        Dense(units=200, activation='relu'),
        Dropout(0.66),
        Dense(units=150, activation='relu'),
        Dropout(0.66),
        Dense(units=output, activation='softmax')
    ])

    return model

# global variables
gloss_show = 'Word: none'
pred = ''
def make_prediction(frame_buffer, model, threshold):
    global gloss_show
    global pred
    frame_buffer_resh = frame_buffer.reshape(1, *frame_buffer.shape)
    # model prediction
    predictions = model.predict(frame_buffer_resh)[0]
    # get the best prediction
    best_pred_idx = np.argmax(predictions)
    acc_best_pred = predictions[best_pred_idx]
    # check mislabeling
    if acc_best_pred > threshold:
        gloss = labels[best_pred_idx]
        pred = gloss
        gloss_show = "Word: {: <3}  {:.2f}% ".format(
            gloss,
            acc_best_pred * 100)
        print(gloss_show)
    else:
        gloss_show = 'Word: none'
        pred = ''

labels = {
    0: 'book',
    1: 'chair',
    2: 'clothes',
    3: 'computer',
    4: 'drink',
    5: 'drum',
    6: 'family',
    7: 'football',
    8: 'go',
    9: 'hat',
    10: 'hello',
    11: 'kiss',
    12: 'like',
    13: 'play',
    14: 'school',
    15: 'street',
    16: 'table',
    17: 'university',
    18: 'violin',
    19: 'wall'
}
def main():
    height = 224
    width = 224
    dim = (height, width)
    batch_size = 8
    frames = 10
    channels = 3
    output = 20
    # model_path = './model/model_weights_300_v1.weights.h5'
    threshold = .50

    print("ASL Real-time Recognition\n")
    print("[INFO] initializing ...")
    # define empty buffer
    frame_buffer = np.empty((0, *dim, channels))

    print("[INFO] loading ASL detection model ...")
    # load model
    # model = load_model(model_path)
    model = create_model_wlasl20c(frames, width, height, channels, output)
    model.summary()
    adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                amsgrad=False, name="Adam")
    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.load_weights('./model/model_weights_W20_300epoch_v1.weights.h5')

    # Create an instance of the grammar correction tool
    tool = language_tool_python.LanguageToolPublicAPI('en-UK')

    # Initialize the lists
    sentence, grammar_result = [], []
    actions = None
    # Access the camera and check if the camera is opened successfully
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera.")
        exit()
    x = threading.Thread()
    while cap.isOpened():
        # Read a frame from the camera
        ret, frame = cap.read()
        if ret:
            # process the frame
            frame_res = cv2.resize(frame, dim)
            frame_res = frame_res / 255.0
            # append the frame to buffer
            frame_resh = np.reshape(frame_res, (1, *frame_res.shape))
            frame_buffer = np.append(frame_buffer, frame_resh, axis=0)
            # start sign recognition only if the buffer is full
            if frame_buffer.shape[0] == frames:
                # make the prediction
                if not x.is_alive():
                    x = threading.Thread(target=make_prediction, args=(
                        frame_buffer, model, threshold))
                    x.start()
                else:
                    pass
                # left-shift of the buffer
                frame_buffer = frame_buffer[1:frames]
                # show label
                if(actions == None):
                    actions = pred
                    sentence.append(pred)
                else:
                    if actions != pred:
                        actions = pred
                        sentence.append(pred)
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(frame, gloss_show, (20, 450), font, 1, (0, 255, 0),
                #             2, cv2.LINE_AA)
                # cv2.imshow('frame', frame)

            # Check if the list is not empty
            if sentence:
                # Capitalize the first word of the sentence
                sentence[0] = sentence[0].capitalize()

            if len(sentence) >= 2:
                # Check if the last element of the sentence belongs to the alphabet (lower or upper cases)
                if sentence[-1] in string.ascii_lowercase or sentence[-1] in string.ascii_uppercase:
                    # Check if the second last element of sentence belongs to the alphabet or is a new word
                    if sentence[-2] in string.ascii_lowercase or sentence[-2] in string.ascii_uppercase or (sentence[-2] not in actions and sentence[-2] not in list(x.capitalize() for x in actions)):
                        # Combine last two elements
                        sentence[-1] = sentence[-2] + sentence[-1]
                        sentence.pop(len(sentence) - 2)
                        sentence[-1] = sentence[-1].capitalize()


            if len(sentence) > 3:
                text = ' '.join(sentence)
                # Apply grammar correction tool and extract the corrected result
                grammar_result = tool.correct(text)
                textsize = cv2.getTextSize(grammar_result, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_X_coord = (frame.shape[1] - textsize[0]) // 2

                # Draw the sentence on the image
                cv2.putText(frame, grammar_result, (text_X_coord, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                sentence = []

            if len(grammar_result) > 0:
                cv2.putText(frame, grammar_result, (text_X_coord, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


            cv2.imshow('Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Shut off the server
    tool.close()

if __name__ == '__main__':
    main()