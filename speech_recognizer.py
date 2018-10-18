from scipy.io import wavfile as wav
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import scipy
from sklearn.model_selection import train_test_split
import os
from tensorflow import keras
from tensorflow.python.client import device_lib



def process_data(data, rate):

    frame_size = 0.025  # 25 milliseconds
    frame_step = 0.01   # 10 milliseconds
    frame_size = int(round(frame_size * rate))  # in samples/frame
    frame_step = int(round(frame_step * rate))
    signal_length = len(data)

    num_frames = int(np.ceil(float(np.abs(signal_length - frame_size)) / frame_step))

    pad_length = num_frames * frame_step + frame_size

    zeros = np.zeros(pad_length - signal_length)

    data = np.append(data, zeros)

    part_1 = np.tile(np.arange(0, frame_size), (num_frames, 1))

    part_2 = np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_size, 1)).T

    indices = part_1 + part_2

    frames = data[indices.astype(np.int32, copy=False)]

    frames *= np.hamming(frame_size)

    n_fft = 512

    frames = ((np.absolute(np.fft.rfft(frames, n_fft)))**2)/n_fft

    return frames


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def main():

    print("Training EMail Scam Classifier")
    print("Devices available for training:", get_available_gpus())
    print("Tensorflow version:", tf.__version__)
    print("Keras version:", keras.__version__)

    return

    # traverse file structure

    speech_files = []
    noise_files = []
    rate = 0

    rel = 'data-speech_commands'
    for dirName, subdirList, fileList in os.walk(rel):
        if dirName == 'data-speech_commands':
            continue
        elif dirName == 'data-speech_commands/_background_noise_':
            for file in fileList:
                if file != 'README.md':
                    rate, raw_data = wav.read(dirName + '/' + file)
                    noise_files.append(raw_data)
        else:
            for file in fileList:
                rate, raw_data = wav.read(dirName + '/' + file)
                speech_files.append(raw_data)


    # process files

    for x in range(len(speech_files)):
        speech_files[x] = process_data(speech_files[x], rate)

    for x in range(len(noise_files)):
        noise_files[x] = process_data(noise_files[x], rate)

    # train classifier

    frame_data = []
    frame_label = []

    for example in speech_files:
        for frame in example:
            frame_data.append(frame)
            frame_label.append(1)

    for example in noise_files:
        for frame in example:
            frame_data.append(frame)
            frame_label.append(0)

    print("Processing to numpy arrays")
    # frame_data = np.array(frame_data)
    # frame_label = np.array(frame_label)

    print("Splitting Data")
    train_data, test_data, train_label, test_label = train_test_split(frame_data, frame_label, test_size=0.2)

    print(train_data[0])
    print(train_label[0])
    print("Create Classifier")
    classifier = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    print("Compile Classifier")
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Training Classifier")
    classifier.fit(train_data, train_label)

    print("Testing Classifier")
    results = classifier.evaluate(test_data, test_label)

    print(results)

    return

    ###########################
    # Process Non-Scam E-Mails
    ###########################

    print("Processing Non-Scam EMails")
    non_scam_emails = []

    rel = 'dataset/spam_assasin_ham'
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, rel)
    for file in os.listdir(filename):
        with open(rel + '/' + file, encoding="ISO-8859-1") as text:
            non_scam_emails.append({'email_body_text': text.read(), 'label': 0})


if __name__ == "__main__":
    main()