# -*- coding: utf-8 -*-
"""ganci_smart.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1haC9mlQCA6I-VpFS60Wy6GilfHgs_4A6
"""


# Commented out IPython magic to ensure Python compatibility.
#@title setup paths

labeling_csv_folder_path = "/content/drive/MyDrive/Projects/Ganci_Smart/data/2024_6_25_labeling"
sensors_csv_folder_path = "/content/drive/MyDrive/Projects/Ganci_Smart/data/2024_6_25_sensors"
video_mp4_folder_path = "/content/drive/MyDrive/Projects/Ganci_Smart/data/2024_6_25_video"
association_data_gopro_json_path = "/content/drive/MyDrive/Projects/Ganci_Smart/data/association_data_GOPRO.json"
github_folder = "/content/drive/MyDrive/Projects/Ganci_Smart/ganci_smart"

# %cd {github_folder}

!pip install -r requirements.txt

#@title read files

import os
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
labeling_data = read_csv_folder(labeling_csv_folder_path,is_csv_data = False)
sensors_data = read_csv_folder(sensors_csv_folder_path)

print(sensors_data.keys())
# Read JSON file
with open(association_data_gopro_json_path, 'r') as json_file:
    association_data_gopro = json.load(json_file)

#@title pre-processing from ganci_smart/notebooks/syncronization_csv_video.ipynb

from src.label import label
from src.transformations.dataset import read_csv_ganci, csv_shift, timestamp, split_sensors
from src.transformations.features import magnitudo, add_hours
from src.transformations.filters import kalman_filter, lowess_filter


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


print(sensors_data['data_20240625_104649.csv'])
clean_and_split_dfs = clean_data(sensors_data['data_20240625_104649.csv'])
clean_and_split_dfs = clean_data_from_path(sensors_paths[0])

for sensor in clean_and_split_dfs.keys():
  print(sensor)
  print(clean_and_split_dfs[sensor].head())

filtered_dfs = apply_kalman_to_sensors(clean_and_split_dfs)

for sensor in filtered_dfs.keys():
  print(sensor)
  print(filtered_dfs[sensor].head())

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

# Example usage:
print(video_dict)

#IDEA 1 documento
"""
Visualizzare a schermo il video frame by frame,
label di quel frame e dati utili del csv in quel frame(magnitudo?), confrontare visivamente plot e video.
Aggiungere meccanismo di shift avanti e indietro del csv.
"""

#Fatta da chat gpt,da provare

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

def visualize_video_with_labels(video_dict):
    for video_name, video_data in video_dict.items():
        cap = cv2.VideoCapture(video_data['video_path'])
        labeling_data = video_data['labeling_data']

        if labeling_data is not None:
            labeling_data = pd.DataFrame(labeling_data)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current time in seconds
            if labeling_data is not None:
                current_labels = labeling_data[(labeling_data['START_OFFSET_SECOND'] <= current_time) &
                                               (labeling_data['END_OFFSET_SECOND'] >= current_time)]

            if len(current_labels) > 0:
                label_text = ' '.join(current_labels['TYPE_EVENT'].values)
            else:
                label_text = 'No Label'

            # Convert frame to RGB for Matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.imshow(frame_rgb)
            plt.title(f'{video_name} - Time: {current_time:.2f}s - Label: {label_text}')
            plt.axis('off')
            plt.show()

        cap.release()

# Example usage:
visualize_video_with_labels(video_dict)

#IDEA 2 Plot magnitudo and labels

"""
Plottare magnitudo ( o altre info utili)  vs tempo lungo tutto il csv
Plottare le label vs tempo lungo tutto il video
Confrontare visivamente i due plot
"""

import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_magnitudo_vs_time(ax, sensor_data):
    for sensor, df in sensor_data.items():
        ax.plot(df['TIMESTAMP'], df['MAGNITUDO'], label=f'{sensor} magnitudo')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Magnitudo')
    ax.legend()

def plot_labels_vs_time(ax, label_data):
    event_colors = {
        'sx_aggancio': 'red',
        'sx_sgancio': 'blue',
        'dx_aggancio': 'green',
        'dx_sgancio': 'purple'
    }

    for index, row in label_data.iterrows():
        event_type = row['TYPE_EVENT']
        color = event_colors.get(event_type, 'black')  # Default to black if event type not found
        ax.plot([row['START_OFFSET_SECOND'], row['END_OFFSET_SECOND']], [index, index], color=color, label=event_type)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Event Index')

# Plotting the data for each video in video_dict
for video_name, data in video_dict.items():
    if data['cleaned_filtered_data'] is not None and data['labeling_data'] is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        plot_magnitudo_vs_time(ax1, data['cleaned_filtered_data'])
        plot_labels_vs_time(ax2, data['labeling_data'])

        ax1.set_title(f'{video_name} - Magnitudo vs Time')
        ax2.set_title(f'{video_name} - Labels vs Time')

        plt.tight_layout()
        plt.show()

#IDEA 2 Plot magnitudo and labels with labels times in hh:mm:ss

"""
Plottare magnitudo ( o altre info utili)  vs tempo lungo tutto il csv
Plottare le label vs tempo lungo tutto il video
Confrontare visivamente i due plot
"""


import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

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

        ax1.set_title(f'{video_name} - Magnitudo vs Time')
        ax2.set_title(f'{video_name} - Labels vs Time')

        plt.tight_layout()
        plt.show()

#IDEA 2 Plot magnitudo and labels with labels times in hh:mm:ss and the intersection plot over time

"""
Plottare magnitudo ( o altre info utili)  vs tempo lungo tutto il csv
Plottare le label vs tempo lungo tutto il video
Confrontare visivamente i due plot
"""


import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

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

def plot_combined(ax, sensor_data, label_data, start_video):
    event_colors = {
        'sx_aggancio': 'red',
        'sx_sgancio': 'blue',
        'dx_aggancio': 'green',
        'dx_sgancio': 'purple'
    }

    start_video_time = datetime.strptime(start_video, '%H:%M:%S')

    # Calculate the overlapping time range
    min_label_time = start_video_time + timedelta(seconds=label_data['START_OFFSET_SECOND'].min())
    max_label_time = start_video_time + timedelta(seconds=label_data['END_OFFSET_SECOND'].max())

    min_sensor_time = max(df['TIMESTAMP'].min() for df in sensor_data.values())
    max_sensor_time = min(df['TIMESTAMP'].max() for df in sensor_data.values())

    overlap_start = max(min_label_time, min_sensor_time)
    overlap_end = min(max_label_time, max_sensor_time)

    if overlap_start >= overlap_end:
        print(f"No overlapping time range between labels and sensor data for {start_video}")
        return

    # Plot magnitudo values within the overlapping time range
    for sensor, df in sensor_data.items():
        filtered_df = df[(df['TIMESTAMP'] >= overlap_start) & (df['TIMESTAMP'] <= overlap_end)]
        ax.plot(filtered_df['TIMESTAMP'], filtered_df['MAGNITUDO'], label=f'{sensor} magnitudo')

    # Plot labels within the overlapping time range
    for index, row in label_data.iterrows():
        event_type = row['TYPE_EVENT']
        color = event_colors.get(event_type, 'black')
        start_time = start_video_time + timedelta(seconds=row['START_OFFSET_SECOND'])
        end_time = start_video_time + timedelta(seconds=row['END_OFFSET_SECOND'])
        if start_time < overlap_start:
            start_time = overlap_start
        if end_time > overlap_end:
            end_time = overlap_end
        ax.plot([start_time, end_time], [0, 0], color=color, linewidth=5, label=event_type)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel('Time')
    ax.set_ylabel('Magnitudo and Labels')

# Plotting the data for each video in video_dict
for video_name, data in video_dict.items():
    if data['cleaned_filtered_data'] is not None and data['labeling_data'] is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        plot_magnitudo_vs_time(ax1, data['cleaned_filtered_data'])
        plot_labels_vs_time(ax2, data['labeling_data'], data['start_video'])
        plot_combined(ax3, data['cleaned_filtered_data'], data['labeling_data'], data['start_video'])

        ax1.set_title(f'{video_name} - Magnitudo vs Time')
        ax2.set_title(f'{video_name} - Labels vs Time')
        ax3.set_title(f'{video_name} - Combined Magnitudo and Labels vs Time')

        plt.tight_layout()
        plt.show()

import cv2
import os

def play_video_frame_by_frame(video_path):

  video = cv2.VideoCapture(video_path)
    # window name and size
  cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)

  while video.isOpened():
      # Read video capture
      ret, frame = video.read()
      # Display each frame
      cv2.imshow("video", frame)
      # show one frame at a time
      key = cv2.waitKey(0)
      while key not in [ord('q'), ord('k')]:
          key = cv2.waitKey(0)
      # Quit when 'q' is pressed
      if key == ord('q'):
          break

# Example usage with video_dict
for video_name, data in video_dict.items():
    video_path = data['video_path']
    print(f"Playing video: {video_name}")
    play_video_frame_by_frame(video_path)