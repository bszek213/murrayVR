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
from vr_voms_utils import find_files, az_el, az_el_dot_world, az_el_dot_local, detect_square_wave_periods, extract_con_and_control
import argparse
from scipy.spatial.distance import cdist

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

"""
-if I scale the elevation by /2, then the dot and eye match up very well.
I do not need to tocuh the azimuth
"""

def rmse(y_true, y_pred):
    """
    RMSE with handling of nans due to y_true having saccades removed
    """
    squared_diffs = (y_true - y_pred) ** 2
    
    #ignore nans in both arrays
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)

    mean_squared_diff = np.nanmean(squared_diffs[valid_indices])

    return np.sqrt(mean_squared_diff)

class vrVoms():
    def __init__(self,check_trials=False, methods_plot=False) -> None:
        print('vrVoms Smooth Pursuit Class')
        self.all_files_dict = find_files(os.path.join(os.getcwd(),'2023_2024'), "experiment_data_pID")
        self.good_sp_trials_path_src = '/home/brianszekely/Desktop/ProjectsResearch/murray_lab/vr_eye/good_sp_trials_src.txt'
        self.good_sp_trials_path_con = '/home/brianszekely/Desktop/ProjectsResearch/murray_lab/vr_eye/good_sp_trials_control.txt'
        self.check_trials = check_trials
        self.methods_plot = methods_plot
        self.src, self.control = extract_con_and_control()
        prefix = '/home/brianszekely/Desktop/ProjectsResearch/murray_lab/vr_eye/2023_2024/'
        self.src = [prefix + file_name for file_name in self.src]
        self.control = [prefix + file_name for file_name in self.control]

    def parse_condition_per(self,file):
            data = pd.read_csv(file)
            if data['Experiment'].iloc[0] == "SMOOTH_PURSUIT":
                self.exp_df = data
                all_sp = self.smooth_pursuit(file)
                # plt.figure()
                # plt.plot(self.exp_df['CyclopeanEyeDirection.x'],label='hor')
                # plt.plot(self.exp_df['CyclopeanEyeDirection.y'],label='vert')
                # plt.plot(self.exp_df['CyclopeanEyeDirection.z'],label='z')
                # plt.legend()
                # plt.show()
                # plt.close()
                return all_sp
            else:
                return None
            
    def smooth_pursuit(self,file):
        self.exp_df = az_el(self.exp_df)
        if 'CyclopeanEyeDirection.az_filter' not in self.exp_df.columns:
            self.exp_df['CyclopeanEyeDirection.az_filter'] = savgol_filter(self.exp_df['CyclopeanEyeDirection.az'],
                                                                    window_length=23,polyorder=2)
            self.exp_df['CyclopeanEyeDirection.el_filter'] = savgol_filter(self.exp_df['CyclopeanEyeDirection.el'],
                                                                window_length=23,polyorder=2)
        self.exp_df['CyclopeanEyeDirection.el_filter'] = self.exp_df['CyclopeanEyeDirection.el_filter'] * 0.5
        sp_dict = self.sp_heuristics()
        # if self.check_trials:
        #     fig, ax1 = plt.subplots(figsize=(15, 8),nrows=1,ncols=1)

        #     # ax1.plot(self.exp_df['TimeStamp'].values, self.exp_df['CyclopeanEyeDirection.az_filter'].to_numpy(), label='azimuth')
        #     # ax1.plot(self.exp_df['TimeStamp'].values, self.exp_df['CyclopeanEyeDirection.el_filter'].to_numpy(), label='elevation')
        #     # ax1.plot(self.exp_df['TimeStamp'].values, -self.exp_df['WorldDotPostion.az'],label='world dot AZ')
        #     # ax1.plot(self.exp_df['TimeStamp'].values, self.exp_df['WorldDotPostion.el'],label='world dot EL')
        #     ax1.plot(self.exp_df['CyclopeanEyeDirection.az_filter'].to_numpy(),
        #              self.exp_df['CyclopeanEyeDirection.el_filter'].to_numpy(),label='eye')
        #     ax1.plot(self.exp_df['WorldDotPostion.az'].to_numpy(),
        #              self.exp_df['WorldDotPostion.el'].to_numpy(),label='dot')
        #     # ax1.plot(self.exp_df['TimeStamp'].values, self.exp_df['LocalDotPostion.az'],label='local dot AZ')
        #     # ax1.plot(self.exp_df['TimeStamp'].values, self.exp_df['LocalDotPostion.el'],label='local dot EL')
        #     ax1.legend()
        #     plt.title(file)
        #     def check_and_append(filename, textfile):
        #         if not os.path.exists(textfile):
        #             with open(textfile, "w") as f:
        #                 pass 

        #         # Read lines from the file
        #         with open(textfile, "r") as f:
        #             lines = f.readlines()

        #         # Check and append the filename if not already present
        #         if filename + "\n" not in lines:
        #             with open(textfile, "a") as f:
        #                 f.write(filename + "\n")

        #     def on_key(event):
        #         if event.key == 'y':
        #             if 'CON1' in file or 'C1' in file:
        #                 check_and_append(file, "good_sp_trials_src.txt")
        #             elif not any(substring in file for substring in ['CON', 'PC', 'C1', 'SF']):
        #                 check_and_append(file, "good_sp_trials_control.txt")
        #             plt.close(fig)
        #         elif event.key == 'n':
        #             if 'CON1' in file or 'C1' in file:
        #                 check_and_append(file, "bad_sp_trials_src.txt")
        #             elif not any(substring in file for substring in ['CON', 'PC', 'C1', 'SF']):
        #                 check_and_append(file, "bad_sp_trials_control.txt")
        #             plt.close(fig)
        #         else:
        #             print("Invalid response. Please press 'y' or 'n'.")

        #     fig.canvas.mpl_connect('key_press_event', on_key)
        #     plt.show()
        return sp_dict

    
    def sp_heuristics(self):
        sp_dict = {}

        self.exp_df = az_el_dot_world(self.exp_df)
        self.exp_df = az_el_dot_local(self.exp_df)

        az_vel = np.abs(self.exp_df['CyclopeanEyeDirection.az_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        az_vel = np.nan_to_num(az_vel, nan=0.0) #fill any nan with 0

        el_vel = np.abs(self.exp_df['CyclopeanEyeDirection.el_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        el_vel = np.nan_to_num(el_vel, nan=0.0) #fill any nan with 0

        az_vel_dot = np.abs(self.exp_df['WorldDotPostion.az'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        az_vel_dot = np.nan_to_num(az_vel_dot, nan=0.0) #fill any nan with 0

        el_vel_dot = np.abs(self.exp_df['WorldDotPostion.el'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        el_vel_dot = np.nan_to_num(el_vel_dot, nan=0.0) #fill any nan with 0

        #L2 norm velocity
        norm_eye = np.sqrt((az_vel - el_vel)**2)
        norm_dot = np.sqrt((az_vel_dot - el_vel_dot)**2)

        #smooth out dot position
        sp_vel_dot = savgol_filter(norm_dot,window_length=23,polyorder=2)

        #detect saccades in sp
        saccades_dict = self.detect_saccades_in_sp()

        # fig, ax1 = plt.subplots(figsize=(15, 8),nrows=2,ncols=1)
        # ax1[0].plot(self.exp_df['CyclopeanEyeDirection.az_filter'].to_numpy(),
        #              self.exp_df['CyclopeanEyeDirection.el_filter'].to_numpy(),label='eye')
        # ax1[0].plot(self.exp_df['WorldDotPostion.az'].to_numpy(),
        #             self.exp_df['WorldDotPostion.el'].to_numpy(),label='dot')
        # ax1[1].plot(self.exp_df['TimeStamp'],norm_eye,label='norm eye')
        # ax1[1].plot(self.exp_df['TimeStamp'],sp_vel_dot,label='norm dot')

        # plt.title(mean_gain)
        # plt.legend()
        # plt.show()

        #cases where target_velocity is zero to avoid division by zero
        smooth_pursuit_gain = norm_eye / sp_vel_dot
        smooth_pursuit_gain = np.where(norm_eye == 0, 0, smooth_pursuit_gain)

        #change saccadic portions in the sp signal to NAN
        timestamps = self.exp_df['TimeStamp'].values
        for _, (start_time, end_time) in saccades_dict.items():
            #range of data
            indices_to_nan = np.where((timestamps >= start_time) & (timestamps <= end_time))[0]
            
            #set to nan
            norm_eye[indices_to_nan] == np.nan
            smooth_pursuit_gain[indices_to_nan] = np.nan

        #metrics
        mean_gain = np.nanmean(smooth_pursuit_gain)
        std_dev = np.nanstd(smooth_pursuit_gain)
        variability = std_dev / mean_gain
        rms_error = rmse(sp_vel_dot,norm_eye)

        #here is my range that I am making up: 0.5 and 1.1. based on a moderate range of 
        #possibilites, as normative gain is usually between 0.8 - 1.0
        print('======================')
        print(mean_gain)
        print('======================')
        if mean_gain >= 0.5 and mean_gain <= 1.01:
            #Save data if condition met
            sp_dict['ID'] = self.exp_df['ParticipantID'].iloc[0]
            sp_dict['gain'] = mean_gain
            sp_dict['cv'] = variability
            sp_dict['rmse'] = rms_error
            return sp_dict
        else:
            return None

        # if self.methods_plot:
        #     # Prepare data for animation
        #     timestamps = self.exp_df['TimeStamp'].values
        #     sp_series = sp_time_series
        #     dot_series = dot_pos

        #     fig, ax = plt.subplots(figsize=(15, 8))
        #     line_sp, = ax.plot([], [], label='Smooth pursuit signal')
        #     line_dot, = ax.plot([], [], label='Dot position')
        #     ax.legend()
        #     plt.title(f'Gain: {gain}, Variability: {variability}')
        #     plt.ylabel('Degrees',fontweight='bold')
        #     plt.xlabel('Time (s)',fontweight='bold')

        #     def init():
        #         ax.set_xlim(timestamps[0], timestamps[-1])
        #         ax.set_ylim(min(np.min(sp_series), np.min(dot_series)), max(np.max(sp_series), np.max(dot_series)))
        #         line_sp.set_data([], [])
        #         line_dot.set_data([], [])
        #         return line_sp, line_dot

        #     def update(frame):
        #         line_sp.set_data(timestamps[:frame], sp_series[:frame])
        #         line_dot.set_data(timestamps[:frame], dot_series[:frame])

        #         # Color the smooth pursuit signal
        #         colors = np.where(np.isin(range(frame), saccade_indices[:frame]), 'red', 'blue')
        #         ax.collections.clear()
        #         ax.scatter(timestamps[:frame], sp_series[:frame], c=colors[:frame], s=20)

        #         return line_sp, line_dot

        #     ani = animation.FuncAnimation(fig, update, frames=len(timestamps), init_func=init, blit=True, repeat=True)

        #     # Save the animation using Pillow
        #     ani.save('smooth_pursuit.gif', writer='pillow', fps=30)

        #     plt.show()

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

    def run_analysis(self):
        if self.check_trials:
            #concussion
            sp_dict_src = []
            for dir in tqdm(self.src):
                files = find_files(dir, "experiment_data_pID")
                # file_paths = next(iter(files.values()))
                for file_paths in files.values():
                    for file in file_paths:
                        try:
                            sp_dict = self.parse_condition_per(file)
                            if sp_dict != None:
                                sp_dict_src.append(sp_dict)
                        except Exception as e:
                            print(f'{Fore.RED}file is not included in analysis: {e}{Style.RESET_ALL}')
            pd.DataFrame(sp_dict_src).to_csv('src_sp.csv',index=False)
            
            #control
            sp_dict_con = []
            for dir in tqdm(self.control):
                files = find_files(dir, "experiment_data_pID")
                for file_paths in files.values():
                    for file in file_paths:
                        try:
                            sp_dict = self.parse_condition_per(file)
                            if sp_dict != None:
                                sp_dict_con.append(sp_dict)
                        except Exception as e:
                            print(f'{Fore.RED}file is not included in analysis: {e}{Style.RESET_ALL}')
            pd.DataFrame(sp_dict_con).to_csv('con_sp.csv',index=False)
            print(f'{Fore.CYAN} TRIALS ARE VETTED. Use this command python3 vr_voms_smooth_pursuit.py {Style.RESET_ALL}')
            exit()

        # #iterate over good saccades
        # with open(self.good_sp_trials_path_src, 'r') as f:
        #     files_src= f.readlines()
        
        # with open(self.good_sp_trials_path_con, 'r') as f:
        #     files_control = f.readlines()
        
        # sp_dict_src = []
        # for file in files_src:
        #     file = file.strip()  #remove any leading/trailing whitespace
        #     if os.path.isfile(file):  #check if the file exists
        #         all_sp = self.parse_condition_per(file)
        #         sp_dict_src.append(all_sp)
        #     else:
        #         print(f'{Fore.RED} File not found: {file}{Style.RESET_ALL}')
        # pd.DataFrame(sp_dict_src).to_csv('src_sp.csv',index=False)

        # sp_dict_con = []
        # for file in files_control:
        #     file = file.strip()  #remove any leading/trailing whitespace
        #     if os.path.isfile(file):  #check if the file exists
        #         all_sp = self.parse_condition_per(file)
        #         sp_dict_con.append(all_sp)
        #     else:
        #         print(f'{Fore.RED} File not found: {file}{Style.RESET_ALL}')
        # pd.DataFrame(sp_dict_con).to_csv('con_sp.csv',index=False)

def main():
    parser = argparse.ArgumentParser(description='Process Saccades from VOMS.')
    parser.add_argument('--check_trials', action='store_true', help='Check for good and bad trials.')
    parser.add_argument('--methods_plot', action='store_true', help='Methods plot for publication and grants.')
    args = parser.parse_args()

    vrVoms(check_trials=args.check_trials, 
           methods_plot=args.methods_plot).run_analysis()

if __name__ == "__main__":
    main()