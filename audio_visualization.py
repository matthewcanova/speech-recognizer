import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from scipy.io import wavfile as wav
import pywt
import math
import os
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pickle

def create_wav(rate, data, name):

    data = np.asarray(data)

    wav.write(name, rate, data)


class VizWindow(tk.Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.pack(side="top", fill="both", expand=True)

        print('Loading Data')
        self.raw_data = np.load('raw.data.npy')
        self.frame_size = 1000

        #############
        # RIGHT SIDE
        #############

        # Initialize matplotlib figure for graphing purposes
        # self.fig = Figure()
        # self.plot1 = self.fig.add_subplot(211)
        # self.plot2 = self.fig.add_subplot(212)

        self.fig, (self.plot1, self.plot2, self.plot3) = plt.subplots(nrows=3, ncols=1)
        print('Plotting Graphs')
        self.generate_raw_data()
        self.generate_fft_data()
        self.generate_wavelet_data()
        plt.tight_layout()

        # Special type of "canvas" to allow for matplotlib graphing
        print('Drawing Canvas')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.plot_widget = self.canvas.get_tk_widget()

        # Add the plot to the tkinter widget
        self.plot_widget.grid(row=0, column=1)

        ############
        # LEFT SIDE
        ############

        self.left_panel = tk.Frame(self)
        self.left_panel.grid(row=0, column=0, sticky='NW', pady=64)

        self.file_label = tk.Label(self.left_panel, text="Enter File Name:")
        self.file_label.pack(side='top')

        self.file_entry = tk.Entry(self.left_panel)
        self.file_entry.pack(side='top')

        self.file_load = tk.Button(self.left_panel, text="Load File", command=self.load_file)
        self.file_load.pack(side='top', fill='both')

        self.frame_label = tk.Label(self.left_panel, text="Set FFT Frame Size (# of samples):")
        self.frame_label.pack(side='top')

        self.frame_entry = tk.Entry(self.left_panel)
        self.frame_entry.pack(side='top')

        # Button for loading file for input
        self.button_load = tk.Button(self.left_panel, text="Update FFT", command=self.update_frames)
        self.button_load.pack(side='top', fill='both')

        self.canvas.show()

    def load_file(self):

        file_base_m = './Audio_Samples/Subject_Meghan/samples/'
        rate, raw_data = wav.read(file_base_m + self.file_entry.get())
        self.raw_data = raw_data.T[0][20000:50000]
        self.generate_raw_data()
        self.generate_fft_data()
        self.canvas.draw()

    def update_frames(self):
        self.frame_size = int(self.frame_entry.get())
        self.generate_fft_data()
        self.canvas.draw()

    def generate_raw_data(self):

        x_axis = range(len(self.raw_data))

        print('Plotting Bar Chart')
        self.plot1.bar(x_axis, self.raw_data, alpha=0.5)
        print('Done Charting')

        self.plot1.set_title("Raw Audio Data")
        self.plot1.set_xlabel("Time")
        self.plot1.set_ylabel("Amplitude")

    def generate_fft_data(self):
        raw_data = self.raw_data

        frame_sample_size = self.frame_size
        num_frames = len(raw_data) // frame_sample_size

        y_axis = []

        for x in range(num_frames):
            sample = raw_data[(x * frame_sample_size):((x + 1) * frame_sample_size)]
            fft = np.abs(np.fft.rfft(sample))
            fft = fft[:len(fft)//2]
            y_axis.append(fft)

        y_axis = np.array(y_axis)
        self.plot2.imshow(y_axis.T, norm=colors.Normalize(), aspect='auto')
        self.plot3.imshow(y_axis.T, norm=colors.Normalize(), aspect='auto')

    def generate_wavelet_data(self):
        wavelets = pywt.dwt(self.raw_data, 'db1')
        print(wavelets)
        print(wavelets[0])
        print(wavelets[0][0])

def main():

    # Main Window
    root = tk.Tk()
    root.wm_title("Audio Data Visualization")
    root.geometry("900x600")

    win = VizWindow(master=root)

    win.mainloop()


if __name__ == '__main__':
    main()
