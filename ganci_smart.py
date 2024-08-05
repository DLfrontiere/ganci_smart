# -*- coding: utf-8 -*-
"""ganci_smart.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1haC9mlQCA6I-VpFS60Wy6GilfHgs_4A6
"""

import os

import numpy as np

#@title setup paths

current_path = os.getcwd()
data_path = os.path.join(current_path,"data")

labeling_csv_folder_path = os.path.join(data_path,"2024_6_25_labeling")
sensors_csv_folder_path = os.path.join(data_path,"2024_6_25_sensors")
video_mp4_folder_path = os.path.join(data_path,"2024_6_25_videos")
association_data_gopro_json_path = os.path.join(data_path,"association_data_GOPRO.json")
github_folder = os.path.join(current_path,"gs")

import sys
sys.path.append(github_folder)



from src.label import label
from src.transformations.dataset import read_csv_ganci, csv_shift, timestamp, split_sensors
from src.transformations.features import magnitudo, add_hours
from src.transformations.filters import kalman_filter, lowess_filter

import pandas as pd
import json

# Function to read all CSV files in a folder

labeling_paths = [os.path.join(labeling_csv_folder_path, f) for f in os.listdir(labeling_csv_folder_path) if f.endswith('.csv')]
sensors_paths = [os.path.join(sensors_csv_folder_path, f) for f in os.listdir(sensors_csv_folder_path) if f.endswith('.csv')]

def read_csv_folder(folder_path,is_csv_data = True):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if is_csv_data:
        dataframes = {csv_file: pd.read_csv(os.path.join(folder_path, csv_file),header=6,sep=';') for csv_file in csv_files}
    else:
        dataframes = {csv_file: pd.read_csv(os.path.join(folder_path, csv_file)) for csv_file in csv_files}
    return dataframes

# Read CSV files from both folders
sensors_data = read_csv_folder(sensors_csv_folder_path)
labeling_data = read_csv_folder(labeling_csv_folder_path,is_csv_data = False)

# Read JSON file
with open(association_data_gopro_json_path, 'r') as json_file:
    association_data_gopro = json.load(json_file)




def clean_data_from_path(csv_path):
  df = read_csv_ganci(csv_path)
  dfs = df\
        .pipe(csv_shift)\
        .pipe(timestamp)\
        .pipe(split_sensors)
  return dfs

def clean_data(df):
  dfs = df\
        .pipe(csv_shift)\
        .pipe(timestamp)\
        .pipe(split_sensors)
  return dfs

def apply_kalman_to_sensors(dfs):
  df = dfs
  for sensor in df.keys():
    df[sensor] = df[sensor].pipe(kalman_filter, cols=['Ax', 'Ay', 'Az'])\
                              .pipe(magnitudo, cols_xyz=['Ax', 'Ay', 'Az'], names_magnitudo='MAGNITUDO')\
                              .pipe(add_hours, hours=2)
  return df



clean_and_split_dfs = clean_data(sensors_data['data_20240625_104649.csv'])
clean_and_split_dfs = clean_data_from_path(sensors_paths[0])


filtered_dfs = apply_kalman_to_sensors(clean_and_split_dfs)

for sensor in filtered_dfs.keys():
  print(sensor)
  print(filtered_dfs[sensor].head(3))

# Create a dictionary with video names as keys
video_dict = {}


for key, value in association_data_gopro.items():

    print(key,value)
    video_name = value['video']
    if not video_name:
      continue

    sens_data = sensors_data.get(f'{key}.csv', None)
    lab_data = labeling_data.get(f'{video_name}.csv', None)
    video_path = os.path.join(video_mp4_folder_path, f'{video_name}.mp4')

    video_data = {
        'video_name': video_name,
        'start_video': value['start_video'],
        'sensors_data': sens_data,
        'labeling_data': lab_data,#TYPE_EVENT	 START_OFFSET_SECOND	END_OFFSET_SECOND
        'video_path': video_path,
        'cleaned_filtered_data': apply_kalman_to_sensors(clean_data(sens_data)) #a dict with 5 keys,one for each of 5 sensors,for each key a dataframe with 8 columns,ax ay ax gx gy gx timestamp magnitudo
    }

    video_dict[video_name] = video_data

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

"""
For each sensor:
    plot magnitudo vs time
    plot labels vs time
"""

def plot_magnitudo_vs_time(ax, sensor_data):
    for sensor, df in sensor_data.items():
        ax.plot(df['TIMESTAMP'], df['MAGNITUDO'], label=f'{sensor} magnitudo')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Magnitudo')
    ax.legend()

def plot_labels_vs_time(ax, label_data, start_video):
    event_colors = {
        'sx_aggancio': 'red',
        'sx_sgancio': 'blue',
        'dx_aggancio': 'green',
        'dx_sgancio': 'purple'
    }

    start_video_time = datetime.strptime(start_video, '%H:%M:%S')

    for index, row in label_data.iterrows():
        event_type = row['TYPE_EVENT']
        color = event_colors.get(event_type, 'black')  # Default to black if event type not found
        start_time = start_video_time + timedelta(seconds=row['START_OFFSET_SECOND'])
        end_time = start_video_time + timedelta(seconds=row['END_OFFSET_SECOND'])
        ax.plot([start_time, end_time], [index, index], color=color, label=event_type)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel('Time')
    ax.set_ylabel('Event Index')

# Plotting the data for each video in video_dict
for video_name, data in video_dict.items():
    if data['cleaned_filtered_data'] is not None and data['labeling_data'] is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        plot_magnitudo_vs_time(ax1, data['cleaned_filtered_data'])
        plot_labels_vs_time(ax2, data['labeling_data'], data['start_video'])

        ax1.set_title(f'{video_name} - Magnitude vs Time')
        ax2.set_title(f'{video_name} - Labels vs Time')

        plt.tight_layout()
        #plt.show()


"""
For each video:
    plot video frame by frame
    plot magnitudo vs time
    plot labels vs time
    vertical line on all plots for current values
"""

import tkinter as tk
from tkinter import Label, Button
from datetime import datetime, timedelta
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import pandas as pd
import matplotlib.dates as mdates

# Global variables
playing = False
fps_step = 1

def display_frame(frame, label, time_label):
    frame_resized = cv2.resize(frame, (720, 480))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    # Calculate and display the current time of the frame
    current_time = start_video_time + timedelta(seconds=frame_idx / fps)
    time_label.config(text=current_time.strftime('%H:%M:%S'))

    # Update plots with the current frame time
    update_plots(current_time)

def next_frame(seconds=1):
    global frame_idx, video_cap, total_frames, fps
    step = int(fps * seconds)
    frame_idx += step
    if frame_idx >= total_frames:
        frame_idx = total_frames - 1
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = video_cap.read()
    if ret:
        display_frame(frame, label, time_label)
    print(f"Next frame: {frame_idx}")

def prev_frame(seconds=1):
    global frame_idx, video_cap, fps
    step = int(fps * seconds)
    frame_idx -= step
    if frame_idx < 0:
        frame_idx = 0
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = video_cap.read()
    if ret:
        display_frame(frame, label, time_label)
    print(f"Previous frame: {frame_idx}")



def plot_magnitudo_vs_time(ax, sensor_data):
    ax.clear()
    
    for sensor, df in sensor_data.items():
        # Ensure TIMESTAMP is in datetime format and drop NaN values
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
        
        # Print df timestamp type and value, and the same for the first element
        print(f"TIMESTAMP type for {sensor}: {type(df['TIMESTAMP'])}")
        print(f"TIMESTAMP values for {sensor}: {df['TIMESTAMP'].head()}")
        print(f"First TIMESTAMP value for {sensor}: {df['TIMESTAMP'].iloc[0]} type: {type(df['TIMESTAMP'].iloc[0])}")
        
        # Plot using full TIMESTAMP but format x-axis to show only time
        ax.plot(df['TIMESTAMP'], df['MAGNITUDO'], label=f'{sensor} magnitudo')
    
    # Format the x-axis to show only the time portion
    ax.set_xlabel('Time')
    ax.set_ylabel('Magnitudo')
    ax.legend()
    ax.figure.autofmt_xdate()  # Auto format date labels


def plot_labels_vs_time(ax, label_data, start_video):
    ax.clear()
    event_colors = {
        'sx_aggancio': 'red',
        'sx_sgancio': 'blue',
        'dx_aggancio': 'green',
        'dx_sgancio': 'purple'
    }

    start_video_time = datetime.strptime(start_video, '%H:%M:%S')

    for index, row in label_data.iterrows():
        event_type = row['TYPE_EVENT']
        color = event_colors.get(event_type, 'black')  # Default to black if event type not found
        start_time = start_video_time + timedelta(seconds=row['START_OFFSET_SECOND'])
        end_time = start_video_time + timedelta(seconds=row['END_OFFSET_SECOND'])
        ax.plot([start_time, end_time], [index, index], color=color, label=event_type)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel('Time')
    ax.set_ylabel('Event Index')

def update_plots(current_time):
    global fig, ax1, ax2
    plot_magnitudo_vs_time(ax1, sensor_data)
    plot_labels_vs_time(ax2, labeling_data, start_video_time.strftime('%H:%M:%S'))
    ax1.axvline(current_time, color='r', linestyle='--')
    ax2.axvline(current_time, color='r', linestyle='--')
    canvas.draw()

def play_video():
    global playing
    playing = True
    while playing:
        next_frame(fps_step)
        root.update()
        root.after(int(1000 / fps))

def stop_video():
    global playing
    playing = False

def set_fps_step():
    global fps_step
    fps_step = float(step_entry.get())
    next_button.config(text=f"Next ({fps_step}s)")
    prev_button.config(text=f"Previous ({fps_step}s)")
    print(f"Set FPS step to: {fps_step}")

for video_name, data in video_dict.items():
    video_path = data['video_path']
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        continue
    
    start_video_time = datetime.strptime(data['start_video'], '%H:%M:%S')
    sensor_data = data['cleaned_filtered_data']
    labeling_data = data['labeling_data']
    
    #print("sd",sensor_data['TR2'].head(5),"ld",labeling_data.head(5))

    print("Opening video")
    video_cap = cv2.VideoCapture(video_path)
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    if total_frames == 0:
        print(f"No frames extracted from video {video_path}.")
        continue

    frame_idx = 0

    # Create tkinter window
    root = tk.Tk()
    root.title(f"Displaying video: {video_name}")

    # Create label to display image
    label = Label(root)
    label.grid(row=0, column=0, rowspan=6)

    # Create time label to display the current time of the frame
    time_label = Label(root, text="")
    time_label.grid(row=6, column=0)

    # Create buttons for next and previous with 1-second step
    button_frame = tk.Frame(root)
    button_frame.grid(row=7, column=0)

    prev_button = Button(button_frame, text=f"Previous ({fps_step}s)", command=lambda: prev_frame(fps_step))
    prev_button.pack(side="left")
    next_button = Button(button_frame, text=f"Next ({fps_step}s)", command=lambda: next_frame(fps_step))
    next_button.pack(side="right")

    stop_button = Button(button_frame, text="Stop", command=stop_video)
    stop_button.pack(side="left")
    play_button = Button(button_frame, text="Play", command=play_video)
    play_button.pack(side="right")

    step_label = Label(button_frame, text="Step (s):")
    step_label.pack(side="left")
    step_entry = tk.Entry(button_frame)
    step_entry.pack(side="left")
    set_step_button = Button(button_frame, text="Set Step", command=set_fps_step)
    set_step_button.pack(side="left")

    min_step_label = Label(root, text=f"Min Step: {1/fps:.2f} s")
    min_step_label.grid(row=8, column=0)

    # Create matplotlib figure and axes
    
    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(8, 8))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=0, column=1, rowspan=12)

    # Display the first frame
    ret, frame = video_cap.read()
    if ret:
        display_frame(frame, label, time_label)

    # Start the tkinter main loop
    root.mainloop()

    video_cap.release()
    break

