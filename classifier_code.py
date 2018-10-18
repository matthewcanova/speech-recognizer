from matplotlib import pyplot as plt

from scipy.io import wavfile as wav
from scipy.fftpack import fft
from scipy.fftpack import dct

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import random
import math
import os


mask_values = {}

# n is a key identifying the hash function, x is the value to be hashed
# references: http://stackoverflow.com/questions/2255604/hash-functions-family-generator-in-python
def hash_function(n, x):
    mask = mask_values.get(n)
    if mask is None:
        random.seed(n)
        mask = mask_values[n] = random.getrandbits(64)
    hashes = []
    m = pow(2, 7)
    for value in map(hash, x):
        hashes.append((value ^ mask) % m)

    return hashes


# t is the number of hash functions used to compute the min hash
def min_hash(t, set_one, set_two):

    min_hash_1 = []
    min_hash_2 = []

    for i in range(t):
        min_hash_1.append(math.inf)
        min_hash_2.append(math.inf)

    for i in range(t):
        hash1 = hash_function(i, set_one)
        min_hash_1[i] = min(hash1)

        hash2 = hash_function(i, set_two)
        min_hash_2[i] = min(hash2)

    js = 0

    for i in range(t):
        if min_hash_1[i] == min_hash_2[i]:
            js += 1

    return js / t


def k_gram(data, k):
    grams = []

    end = len(data)
    i = k
    while i < end:
        gram = []
        for j in range(k):
            for l in range(len(data[i-k+j])):
                gram.append(data[i-k+j][l])
        grams.append(tuple(gram))
        i += k

    return grams


def process_file(file):

    rate, raw_data = wav.read(file)

    raw_data = raw_data.T[0]

    data = raw_data[0:60000]

    # emphasize signal
    emphasis = 0.97
    data = np.append(data[0], data[1:] - emphasis * data[:-1])

    frame_size = 0.025  # 25 milliseconds
    frame_stride = 0.01  # 10 milliseconds
    frame_length = frame_size * rate  # in samples/frame
    frame_step = frame_stride * rate  # in samples/frame

    signal_length = len(data)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    # Make sure that we have at least 1 frame
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # Pad Signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    pad_signal = np.append(data, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    frames *= np.hamming(frame_length)

    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = (1.0 / NFFT) * (mag_frames ** 2)  # Power Spectrum

    nfilt = 40

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13

    (nframes, ncoeff) = mfcc.shape
    cep_lifter = 22
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  # *

    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)

    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    return [tuple(x.tolist()) for x in mfcc]

def process_file_raw(file):

    rate, raw_data = wav.read(file)

    raw_data = raw_data.T[0]

    data = raw_data[0:60000]

    # emphasize signal
    emphasis = 0.97
    data = np.append(data[0], data[1:] - emphasis * data[:-1])

    frame_size = 0.025  # 25 milliseconds
    frame_stride = 0.01  # 10 milliseconds
    frame_length = frame_size * rate  # in samples/frame
    frame_step = frame_stride * rate  # in samples/frame

    signal_length = len(data)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    # Make sure that we have at least 1 frame
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # Pad Signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    pad_signal = np.append(data, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    frames *= np.hamming(frame_length)

    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = (1.0 / NFFT) * (mag_frames ** 2)  # Power Spectrum

    return pow_frames


def create_wav(rate, data, name):

    data = np.asarray(data)

    wav.write(name, rate, data)


def main():

    ###############################
    # REPORTING/LOGGING PARAMETERS
    ###############################

    num_samples = 0
    log_fails = True

    ############################################################
    # GENERATE CLASSIFIER FOR HUMAN AUDITION/NOT HUMAN AUDITION
    ############################################################

    with open("data-speech_commands/nigerian_prince_emails.txt", encoding="ISO-8859-1") as f:
        examples = f.read().splitlines()[1:]

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

    ##################################
    # CREATE TRAIN AND TEST DATA SETS
    ##################################

    all_emails = []
    random.shuffle(all_emails)

    split = int(len(all_emails) * 0.8)

    train_data = []
    test_data = []

    for i in range(split):
        train_data.append(all_emails[i])

    for i in range(split, len(all_emails)):
        test_data.append(all_emails[i])

    ###################
    # TRAIN CLASSIFIER
    ###################

    vector_data_text = []
    label_data = []

    for em in train_data:
        vector_data_text.append(em['email_body_processed'])
        label_data.append(em['label'])

    classifier = linear_model.SGDClassifier(max_iter=1000, tol=0.0001)

    vectorizer = CountVectorizer(analyzer=str.split)

    vector_data = vectorizer.fit_transform(vector_data_text)

    classifier.fit(vector_data, label_data)

    ###################
    # STORE CLASSIFIER
    ###################

    filename = 'bow_classifier.joblib.pkl'

    _ = joblib.dump(classifier, filename, compress=3)

    ##################
    # LOAD CLASSIFIER
    ##################

    classifier = joblib.load(filename)

    #######################
    # ANALYZE TOP FEATURES
    #######################

    print("Analyzing Top Features... \n")
    features = vectorizer.get_feature_names()
    weights = classifier.coef_[0]
    pairs = {}

    for feature in range(len(features)):
            pairs[features[feature]] = weights[feature]

    num_features = 20
    print("\nTop " + str(num_features) + " Features for identifying a scam email: \n")
    for x in range(num_features):
        max_feature = max(pairs, key=pairs.get)
        print(max_feature)
        print(pairs[max_feature])
        pairs.pop(max_feature)

    print("\nTop " + str(num_features) + " Features for identifying a non-scam email: \n")
    for x in range(num_features):
        min_feature = min(pairs, key=pairs.get)
        print(min_feature)
        print(pairs[min_feature])
        pairs.pop(min_feature)

    ####################################
    # TEST CLASSIFIER AGAINST TEST DATA
    ####################################

    print("\nTesting Classifier and Printing Failed Predictions: \n")
    num_correct = 0
    num_guesses = 0
    num_scams = 0
    for test in test_data:
        vector_data = vectorizer.transform([test['email_body_processed']])
        prediction = classifier.predict(vector_data)[0]
        test['prediction'] = prediction
        label = test['label']
        if prediction == 1:
            num_guesses += 1
        if label == 1:
            num_scams += 1
        if prediction == label and label == 1:
            num_correct += 1
        elif log_fails and prediction != label:
            print("EMail Not Labeled Correctly:")

    precision = num_correct/num_guesses
    recall = num_correct/num_scams

    print("After Training with " + str(len(train_data)) + " emails")
    print("and Testing " + str(len(test_data)) + " emails")
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F-Score: " + str((2 * precision * recall) / (precision + recall)))
    print("\nPrinting samples from the test run: \n")


if __name__ == "__main__":
    main()

