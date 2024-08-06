import numpy as np
import os
from scipy.signal import find_peaks

def az_el(df):
    # azimuth = np.arctan(df['CyclopeanEyeDirection.y'], df['CyclopeanEyeDirection.x'])
    azimuth = np.arctan(df['CyclopeanEyeDirection.x'], df['CyclopeanEyeDirection.z'])
    # elevation = np.arctan(df['CyclopeanEyeDirection.y'], np.sqrt(df['CyclopeanEyeDirection.x']**2 + df['CyclopeanEyeDirection.z']**2))
    elevation = np.arctan(df['CyclopeanEyeDirection.y'], np.sqrt(df['CyclopeanEyeDirection.x']**2 + df['CyclopeanEyeDirection.z']**2))
    df['CyclopeanEyeDirection.az'] = np.degrees(azimuth)
    df['CyclopeanEyeDirection.el'] = np.degrees(elevation)
    #mean offset?
    df['CyclopeanEyeDirection.az'] = df['CyclopeanEyeDirection.az'] - df['CyclopeanEyeDirection.az'].mean()
    df['CyclopeanEyeDirection.el'] = df['CyclopeanEyeDirection.el'] - df['CyclopeanEyeDirection.el'].mean()
    return df

def az_el_dot_world(df):
    azimuth = np.arctan(df['WorldDotPostion.x'], df['WorldDotPostion.z'])
    elevation = np.arctan(df['WorldDotPostion.y'], np.sqrt(df['WorldDotPostion.x']**2 + df['WorldDotPostion.z']**2))
    df['WorldDotPostion.az'] = np.degrees(azimuth)
    df['WorldDotPostion.el'] = np.degrees(elevation)
    #mean offset?
    df['WorldDotPostion.az'] = df['WorldDotPostion.az'] - df['WorldDotPostion.az'].mean()
    df['WorldDotPostion.el'] = df['WorldDotPostion.el'] - df['WorldDotPostion.el'].mean()
    return df

def az_el_dot_local(df):
    azimuth = np.arctan2(df['LocalDotPostion.x'], df['LocalDotPostion.z'])
    elevation = np.arctan2(df['LocalDotPostion.y'], np.sqrt(df['LocalDotPostion.x']**2 + df['LocalDotPostion.z']**2))
    df['LocalDotPostion.az'] = np.degrees(azimuth)
    df['LocalDotPostion.el'] = np.degrees(elevation)
    #mean offset?
    df['LocalDotPostion.az'] = df['LocalDotPostion.az'] - df['LocalDotPostion.az'].mean()
    df['LocalDotPostion.el'] = df['LocalDotPostion.el'] - df['LocalDotPostion.el'].mean()
    return df

def find_files(root_dir, substring):
    file_paths_dict = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if substring in filename:
                folder_name = os.path.basename(dirpath)
                if folder_name not in file_paths_dict:
                    file_paths_dict[folder_name] = []
                file_paths_dict[folder_name].append(os.path.join(dirpath, filename))
    return file_paths_dict

def detect_square_wave_periods(df, column_name, min_length=30):

    low, high = np.percentile(df[column_name].dropna(),[25,52])
    
    # Detect large changes which are typical of square wave transitions
    peaks, _ = find_peaks(df[column_name].dropna().abs(),distance=10)#height=high,

    # Group peaks into periods of consistent large changes
    peaks = list(peaks)  # Convert array to list for easy manipulation
    i = 0
    while i < len(peaks) - 1:
        if peaks[i+1] - peaks[i] < min_length:
            peaks.pop(i)
        else:
            i += 1

    return peaks

def extract_con_and_control():
    main_directory = '2023_2024'

    src = []
    control = []

    for root, dirs, files in os.walk(main_directory):
        for dir_name in dirs:
            if 'CON1' in dir_name or 'C1' in dir_name:
                src.append(dir_name)
            elif not any(substring in dir_name for substring in ['CON', 'PC', 'C1', 'SF']):
                control.append(dir_name)

    # print("Concussions:")
    # print(src)

    # print("\nControls:")
    # print(control)

    return src, control