import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks, square, savgol_filter
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
from colorama import Fore, Style
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageSequence
from sys import argv
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation

"""
Talk to murray about smooth pursuit. all of the movement is in azimuth.
why is the world dot position an H, but the local dot position is not in smooth pursuits
the eye signal follows the local dot position not the world dot position in th smooth pursuits
"""
plt.rcParams.update({
    'font.size': 14,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'legend.fontsize': 14,
    'legend.title_fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'lines.linewidth': 2
})

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

def az_el(df):
    azimuth = np.arctan(df['CyclopeanEyeDirection.y'], df['CyclopeanEyeDirection.x'])
    elevation = np.arctan(df['CyclopeanEyeDirection.z'], np.sqrt(df['CyclopeanEyeDirection.x']**2 + df['CyclopeanEyeDirection.y']**2))
    df['CyclopeanEyeDirection.az'] = np.degrees(azimuth)
    df['CyclopeanEyeDirection.el'] = np.degrees(elevation)
    #mean offset?
    df['CyclopeanEyeDirection.az'] = df['CyclopeanEyeDirection.az'] - df['CyclopeanEyeDirection.az'].mean()
    df['CyclopeanEyeDirection.el'] = df['CyclopeanEyeDirection.el'] - df['CyclopeanEyeDirection.el'].mean()
    return df

def az_el_dot(df):
    azimuth = np.arctan(df['WorldDotPostion.y'], df['WorldDotPostion.x'])
    elevation = np.arctan(df['WorldDotPostion.z'], np.sqrt(df['WorldDotPostion.x']**2 + df['WorldDotPostion.y']**2))
    df['WorldDotPostion.az'] = np.degrees(azimuth)
    df['WorldDotPostion.el'] = np.degrees(elevation)
    #mean offset?
    df['WorldDotPostion.az'] = df['WorldDotPostion.az'] - df['WorldDotPostion.az'].mean()
    df['WorldDotPostion.el'] = df['WorldDotPostion.el'] - df['WorldDotPostion.el'].mean()
    return df

class vrVoms():
    def __init__(self) -> None:
        print('vrVoms class')
        self.all_files_dict = find_files(os.path.join(os.getcwd(),'2023_2024'), "experiment_data_pID")
        self.good_sac_trials_path = '/home/brianszekely/Desktop/ProjectsResearch/murray_lab/vr_eye/good_saccade_trials.txt'
        self.good_sp_trials_path = '/home/brianszekely/Desktop/ProjectsResearch/murray_lab/vr_eye/good_smooth_pursuit_trials.txt'


    def parse_condition_per(self,file):
            data = pd.read_csv(file)

            # if data['Experiment'].iloc[0] == "SACCADES":
            #     self.exp_df = data
            #     self.saccades(file)
            # if data['Experiment'].iloc[0] == "SMOOTH_PURSUIT":
            #     self.exp_df = data
            #     self.smooth_pursuit(file)
            if data['Experiment'].iloc[0] == "VOR":
                self.exp_df = data
                self.VOR()
            # if data['Experiment'].iloc[0] == "VMS":
            #     self.exp_df = data
    
    def VOR(self):
        self.exp_df = az_el(self.exp_df)

        self.exp_df['HeadOrientation.x'] = self.exp_df['HeadOrientation.x'].apply(lambda x: x - 360 if x > 100 else x)
        self.exp_df['HeadOrientation.y'] = self.exp_df['HeadOrientation.y'].apply(lambda x: x - 360 if x > 100 else x)
        self.exp_df['HeadOrientation.z'] = self.exp_df['HeadOrientation.z'].apply(lambda x: x - 360 if x > 100 else x)
        
        self.exp_df['HeadOrientation.x'] = self.exp_df['HeadOrientation.x'] - self.exp_df['HeadOrientation.x'].mean()
        self.exp_df['HeadOrientation.y'] = self.exp_df['HeadOrientation.y'] - self.exp_df['HeadOrientation.y'].mean()
        self.exp_df['HeadOrientation.z'] = self.exp_df['HeadOrientation.z'] - self.exp_df['HeadOrientation.z'].mean()

        if 'CyclopeanEyeDirection.az_filter' not in self.exp_df.columns:
            self.exp_df['CyclopeanEyeDirection.az_filter'] = savgol_filter(self.exp_df['CyclopeanEyeDirection.az'],
                                                                    window_length=45,polyorder=2)
            self.exp_df['CyclopeanEyeDirection.el_filter'] = savgol_filter(self.exp_df['CyclopeanEyeDirection.el'],
                                                                window_length=45,polyorder=2)
        
        az_vel = self.exp_df['CyclopeanEyeDirection.az_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp']))
        az_vel = np.nan_to_num(az_vel, nan=0.0)
        el_vel = self.exp_df['CyclopeanEyeDirection.el_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp']))
        el_vel = np.nan_to_num(el_vel, nan=0.0)

        pitch_vel = self.exp_df['HeadOrientation.x'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp']))
        pitch_vel = np.nan_to_num(pitch_vel, nan=0.0)
        yaw_vel = self.exp_df['HeadOrientation.y'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp']))
        yaw_vel = np.nan_to_num(yaw_vel, nan=0.0)

        what = np.nanmean(el_vel / pitch_vel)
        what2 = np.nanmean(az_vel / yaw_vel)
        print(what2)
        print(what)
        input()
        # plt.plot(az_vel)
        # plt.plot(el_vel)
        # plt.plot(pitch_vel)
        # plt.plot(yaw_vel)

        # plt.plot(self.exp_df['HeadOrientation.x'])
        # plt.plot(self.exp_df['HeadOrientation.y'])
        # plt.plot(self.exp_df['CyclopeanEyeDirection.az_filter'])
        # plt.plot(self.exp_df['CyclopeanEyeDirection.el_filter'])
        plt.show()

    def sp_heuristics(self):
        self.exp_df = az_el_dot(self.exp_df)

        az_vel = np.abs(self.exp_df['CyclopeanEyeDirection.az_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        az_vel = np.nan_to_num(az_vel, nan=0.0) #fill any nan with 0

        el_vel = np.abs(self.exp_df['CyclopeanEyeDirection.el_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        el_vel = np.nan_to_num(el_vel, nan=0.0) #fill any nan with 0

        az_vel_dot = np.abs(self.exp_df['WorldDotPostion.az'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        az_vel_dot = np.nan_to_num(az_vel_dot, nan=0.0) #fill any nan with 0

        el_vel_dot = np.abs(self.exp_df['WorldDotPostion.az'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        el_vel_dot = np.nan_to_num(el_vel_dot, nan=0.0) #fill any nan with 0

        #use variance to determine horizontal versus vertical smooth pursuit
        if np.var(self.exp_df['CyclopeanEyeDirection.az_filter']) > np.var(self.exp_df['CyclopeanEyeDirection.el_filter']):
            sp_time_series = self.exp_df['CyclopeanEyeDirection.az_filter']
            dot_pos = self.exp_df['WorldDotPostion.az']
            sp_vel = az_vel
            sp_vel_dot = az_vel_dot
        else:
            sp_time_series = self.exp_df['CyclopeanEyeDirection.el_filter']
            dot_pos = self.exp_df['WorldDotPostion.el']
            sp_vel = el_vel
            sp_vel_dot = el_vel_dot

        # norm_sp = np.linalg.norm(df['col1'] - df['col2'])
        # norm_dot
        plt.figure()
        # plt.plot(self.exp_df['CyclopeanEyeDirection.el_filter'],self.exp_df['CyclopeanEyeDirection.az_filter'])
        plt.plot(self.exp_df['CyclopeanEyeDirection.el_filter'])
        plt.plot(self.exp_df['CyclopeanEyeDirection.az_filter'])
        plt.show()
        #smooth out dot position
        sp_vel_dot = savgol_filter(sp_vel_dot,window_length=30,polyorder=2)

        #detect saccades in sp
        saccades_dict = self.detect_saccades_in_sp()

        indices_above = np.where(sp_vel_dot > 0.5)[0]
        sp_vel_dot_update = sp_vel_dot[indices_above]
        sp_vel_update = sp_vel[indices_above]

        #heurstic replace saccadic values with nan
        sp_vel_update = np.where(sp_vel_update > 30, np.nan, sp_vel_update)
        gain = sp_vel_update / sp_vel_dot_update
        gain = np.mean(np.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0))

        #sp variability
        velocity_diff = sp_vel_update - sp_vel_dot_update
        variability = np.nanstd(velocity_diff)

        # Find saccade indices
        saccade_indices = []
        for key, (start, end) in saccades_dict.items():
            saccade_indices.extend(self.exp_df[(self.exp_df['TimeStamp'] >= start) & (self.exp_df['TimeStamp'] <= end)].index.tolist())
        print(saccades_dict)
        input()
        if argv[2] == 'methods_plot':
            # Prepare data for animation
            timestamps = self.exp_df['TimeStamp'].values
            sp_series = sp_time_series
            dot_series = dot_pos

            fig, ax = plt.subplots(figsize=(15, 8))
            line_sp, = ax.plot([], [], label='Smooth pursuit signal')
            line_dot, = ax.plot([], [], label='Dot position')
            ax.legend()
            plt.title(f'Gain: {gain}, Variability: {variability}')
            plt.ylabel('Degrees',fontweight='bold')
            plt.xlabel('Time (s)',fontweight='bold')

            def init():
                ax.set_xlim(timestamps[0], timestamps[-1])
                ax.set_ylim(min(np.min(sp_series), np.min(dot_series)), max(np.max(sp_series), np.max(dot_series)))
                line_sp.set_data([], [])
                line_dot.set_data([], [])
                return line_sp, line_dot

            def update(frame):
                line_sp.set_data(timestamps[:frame], sp_series[:frame])
                line_dot.set_data(timestamps[:frame], dot_series[:frame])

                # Color the smooth pursuit signal
                colors = np.where(np.isin(range(frame), saccade_indices[:frame]), 'red', 'blue')
                ax.collections.clear()
                ax.scatter(timestamps[:frame], sp_series[:frame], c=colors[:frame], s=20)

                return line_sp, line_dot

            ani = animation.FuncAnimation(fig, update, frames=len(timestamps), init_func=init, blit=True, repeat=True)

            # Save the animation using Pillow
            ani.save('smooth_pursuit.gif', writer='pillow', fps=30)

            plt.show()

    def detect_saccades_in_sp(self):
        az_vel = np.abs(self.exp_df['CyclopeanEyeDirection.az_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        az_vel = np.nan_to_num(az_vel, nan=0.0) #fill any nan with 0

        el_vel = np.abs(self.exp_df['CyclopeanEyeDirection.el_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        el_vel = np.nan_to_num(el_vel, nan=0.0) #fill any nan with 0
        saccade_found = False
        saccades_dict = {}
        saccade_number = 1

        #check to see if this is horizonal or vertical saccade task
        if np.var(self.exp_df['CyclopeanEyeDirection.az_filter']) > np.var(self.exp_df['CyclopeanEyeDirection.el_filter']):
            saccade_time_series = self.exp_df['CyclopeanEyeDirection.az_filter']
            sacc_vel = az_vel
        else:
            saccade_time_series = self.exp_df['CyclopeanEyeDirection.el_filter']
            sacc_vel = el_vel
        for i in range(len(saccade_time_series)):
            if sacc_vel[i] > 30 and saccade_found == False: #velocity threshold for saccade
                saccade_start = self.exp_df['TimeStamp'].iloc[i]
                saccade_found = True
            if sacc_vel[i] < 30 and saccade_found == True:
                saccade_end = self.exp_df['TimeStamp'].iloc[i]
                saccade_found = False
                sub_df = self.exp_df[(self.exp_df['TimeStamp'] >= saccade_start) & 
                                     (self.exp_df['TimeStamp'] <= saccade_end)]
                # saccade_disp = abs(sub_df['CyclopeanEyeDirection.az_filter'].iloc[-1]) - abs(sub_df['CyclopeanEyeDirection.az_filter'].iloc[0])
                saccade_dur = (sub_df['TimeStamp'].iloc[-1] - sub_df['TimeStamp'].iloc[0]) * 1000
                # print(saccade_disp, saccade_dur*1000)
                if saccade_dur > 20:
                    saccades_dict[saccade_number] = [saccade_start,saccade_end]
                    saccade_number += 1

        return saccades_dict
    
    def smooth_pursuit(self,file):
        self.exp_df = az_el(self.exp_df)
        if 'CyclopeanEyeDirection.az_filter' not in self.exp_df.columns:
            self.exp_df['CyclopeanEyeDirection.az_filter'] = savgol_filter(self.exp_df['CyclopeanEyeDirection.az'],
                                                                    window_length=45,polyorder=2)
            self.exp_df['CyclopeanEyeDirection.el_filter'] = savgol_filter(self.exp_df['CyclopeanEyeDirection.el'],
                                                                window_length=45,polyorder=2)
        self.sp_heuristics()
        if argv[2] == "check_sp_trials":
            fig, ax1 = plt.subplots(figsize=(15, 8),nrows=1,ncols=1)

            ax1.plot(self.exp_df['TimeStamp'].values, self.exp_df['CyclopeanEyeDirection.az_filter'].to_numpy(), label='azimuth')
            ax1.plot(self.exp_df['TimeStamp'].values, self.exp_df['CyclopeanEyeDirection.el_filter'].to_numpy(), label='elevation')
            ax1.legend()
            plt.title(file)
            def check_and_append(filename, textfile):
                if not os.path.exists(textfile):
                    with open(textfile, "w") as f:
                        pass 

                # Read lines from the file
                with open(textfile, "r") as f:
                    lines = f.readlines()

                # Check and append the filename if not already present
                if filename + "\n" not in lines:
                    with open(textfile, "a") as f:
                        f.write(filename + "\n")

            def on_key(event):
                if event.key == 'y':
                    check_and_append(file, "good_smooth_pursuit_trials.txt")
                    plt.close(fig)
                elif event.key == 'n':
                    check_and_append(file, "bad_smooth_pursuit_trials.txt")
                    plt.close(fig)
                else:
                    print("Invalid response. Please press 'y' or 'n'.")

            fig.canvas.mpl_connect('key_press_event', on_key)
            plt.show()
        

    def saccades(self,file):
        self.exp_df = az_el(self.exp_df)
        if 'CyclopeanEyeDirection.az_filter' not in self.exp_df.columns:
            self.exp_df['CyclopeanEyeDirection.az_filter'] = savgol_filter(self.exp_df['CyclopeanEyeDirection.az'],
                                                                    window_length=45,polyorder=2)
            self.exp_df['CyclopeanEyeDirection.el_filter'] = savgol_filter(self.exp_df['CyclopeanEyeDirection.el'],
                                                                window_length=45,polyorder=2)

        az_periods = detect_square_wave_periods(self.exp_df,'CyclopeanEyeDirection.az')
        saccades_dict = self.saccade_heurstics()
        all_saccades = {}
        sacc_velocity_az, sacc_duration, sacc_velocity_el = [], [], []
        for keys, items in saccades_dict.items():
            filtered_df = self.exp_df[(self.exp_df['TimeStamp'] >= items[0]) & (self.exp_df['TimeStamp'] <= items[1])]
            sacc_duration.append((items[1] - items[0]) * 1000)
            sacc_velocity_az.append(np.nanmean(filtered_df['CyclopeanEyeDirection.az_filter'].diff().abs()
                                                     / filtered_df["TimeStamp"].diff()))
            sacc_velocity_el.append(np.nanmean(filtered_df['CyclopeanEyeDirection.el_filter'].diff().abs()
                                                     / filtered_df["TimeStamp"].diff()))
        all_saccades['num_saccades'] = len(saccades_dict)
        all_saccades['sacc_duration'] = np.mean(sacc_duration)
        all_saccades['sacc_velocity_az'] = np.mean([x for x in sacc_velocity_az if np.isfinite(x)])
        all_saccades['sacc_velocity_el'] = np.mean([x for x in sacc_velocity_el if np.isfinite(x)])
        all_saccades['experiment'] = self.exp_df['Experiment'].iloc[0]
        all_saccades['ID'] = self.exp_df['ParticipantID'].iloc[0]

        if argv[2] == "check_saccade_trials":
            fig, ax1 = plt.subplots(figsize=(15, 8),nrows=1,ncols=1)

            ax1.plot(self.exp_df['TimeStamp'].values, self.exp_df['CyclopeanEyeDirection.az'].to_numpy(), label='azimuth')
            ax1.plot(self.exp_df['TimeStamp'].values, self.exp_df['CyclopeanEyeDirection.el'].to_numpy(), label='elevation')

            ax1.scatter(self.exp_df['TimeStamp'].values[az_periods], self.exp_df['CyclopeanEyeDirection.az'].to_numpy()[az_periods], label='eyex')

            ax1.set_xlabel('Time')
            ax1.set_ylabel('Eye Position', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')        

            ax1.legend()
            plt.title(file)

            def check_and_append(filename, textfile):
                with open(textfile, "r") as f:
                    lines = f.readlines()
                if filename + "\n" not in lines:
                    with open(textfile, "a") as f:
                        f.write(filename + "\n")

            def on_key(event):
                if event.key == 'y':
                    check_and_append(file, "good_saccade_trials.txt")
                    plt.close(fig)
                elif event.key == 'n':
                    check_and_append(file, "bad_saccade_trials.txt")
                    plt.close(fig)
                else:
                    print("Invalid response. Please press 'y' or 'n'.")

            fig.canvas.mpl_connect('key_press_event', on_key)
            plt.show()

        return all_saccades

    def saccade_heurstics(self):
        az_vel = np.abs(self.exp_df['CyclopeanEyeDirection.az_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        az_vel = np.nan_to_num(az_vel, nan=0.0) #fill any nan with 0

        el_vel = np.abs(self.exp_df['CyclopeanEyeDirection.el_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        el_vel = np.nan_to_num(el_vel, nan=0.0) #fill any nan with 0
        saccade_found = False
        saccades_dict = {}
        saccade_number = 1

        #check to see if this is horizonal or vertical saccade task
        if np.var(self.exp_df['CyclopeanEyeDirection.az_filter']) > np.var(self.exp_df['CyclopeanEyeDirection.el_filter']):
            saccade_time_series = self.exp_df['CyclopeanEyeDirection.az_filter']
            sacc_vel = az_vel
        else:
            saccade_time_series = self.exp_df['CyclopeanEyeDirection.el_filter']
            sacc_vel = el_vel
        for i in range(len(saccade_time_series)):
            if sacc_vel[i] > 90 and saccade_found == False: #velocity threshold for saccade
                saccade_start = self.exp_df['TimeStamp'].iloc[i]
                saccade_found = True
            if sacc_vel[i] < 90 and saccade_found == True:
                saccade_end = self.exp_df['TimeStamp'].iloc[i]
                saccade_found = False
                sub_df = self.exp_df[(self.exp_df['TimeStamp'] >= saccade_start) & 
                                     (self.exp_df['TimeStamp'] <= saccade_end)]
                # saccade_disp = abs(sub_df['CyclopeanEyeDirection.az_filter'].iloc[-1]) - abs(sub_df['CyclopeanEyeDirection.az_filter'].iloc[0])
                saccade_dur = (sub_df['TimeStamp'].iloc[-1] - sub_df['TimeStamp'].iloc[0]) * 1000
                # print(saccade_disp, saccade_dur*1000)
                if saccade_dur > 20:
                    saccades_dict[saccade_number] = [saccade_start,saccade_end]
                    saccade_number += 1

        # Plotting and animation setup
        if argv[2] == 'methods_plot':
            fig, ax = plt.subplots()
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Azimuth (degrees)')
            ax.set_title('Saccade Detection')
            ax.legend(['Azimuth'])

            # Plot initial data
            line, = ax.plot(self.exp_df['TimeStamp'].values, self.exp_df['CyclopeanEyeDirection.az'].to_numpy(), color='blue')
            saccade_line, = ax.plot([], [], color='green', linewidth=4)

            def update(frame):
                if frame < len(saccades_dict):
                    saccade_start = saccades_dict[frame + 1][0]
                    saccade_end = saccades_dict[frame + 1][1]

                    # Highlight the saccade region in green
                    mask = (self.exp_df['TimeStamp'] >= saccade_start) & (self.exp_df['TimeStamp'] <= saccade_end)
                    saccade_line.set_data(self.exp_df['TimeStamp'][mask], self.exp_df['CyclopeanEyeDirection.az'][mask])

                return saccade_line,

            ani = FuncAnimation(fig, update, frames=len(saccades_dict), blit=True)

            # Save animation as a GIF
            ani.save('saccade_animation.gif', writer='pillow', fps=1)
        
        return saccades_dict

    def run_analysis(self):
        if argv[1] == 'check_trials':
            for key in tqdm(self.all_files_dict.keys()):
                all_files_per_part = self.all_files_dict[key]
                for file in all_files_per_part:
                    try:
                        self.parse_condition_per(file)
                    except Exception as e:
                        print(f'{Fore.RED}file is not included in analysis: {e}{Style.RESET_ALL}')

        # iterate over good smooth pursuit trials
        with open(self.good_sp_trials_path, 'r') as f:
            files = f.readlines()
        
        i = 0
        for file in files:
            file = file.strip()  # Remove any leading/trailing whitespace
            if os.path.isfile(file):  # Check if the file exists
                # if i == 2:
                self.parse_condition_per(file)
            else:
                print(f'{Fore.RED} File not found: {file}{Style.RESET_ALL}')
            i += 1

        #iterate over good saccades
        with open(self.good_sac_trials_path, 'r') as f:
            files = f.readlines()

        for file in files:
            file = file.strip()  # Remove any leading/trailing whitespace
            if os.path.isfile(file):  # Check if the file exists
                self.parse_condition_per(file)
            else:
                print(f'{Fore.RED} File not found: {file}{Style.RESET_ALL}')
        
        exit()
        

def main():
    vrVoms().run_analysis()

if __name__ == "__main__":
    main()
