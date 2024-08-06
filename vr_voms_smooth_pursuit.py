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

class vrVoms():
    def __init__(self,check_trials=False, methods_plot=False) -> None:
        print('vrVoms Smooth Pursuit Class')
        self.all_files_dict = find_files(os.path.join(os.getcwd(),'2023_2024'), "experiment_data_pID")
        self.good_sp_trials_path_src = '/home/brianszekely/Desktop/ProjectsResearch/murray_lab/vr_eye/good_saccade_trials_src.txt'
        self.good_sp_trials_path_con = '/home/brianszekely/Desktop/ProjectsResearch/murray_lab/vr_eye/good_saccade_trials_control.txt'
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
            
    def smooth_pursuit(self,file):
        self.exp_df = az_el(self.exp_df)
        if 'CyclopeanEyeDirection.az_filter' not in self.exp_df.columns:
            self.exp_df['CyclopeanEyeDirection.az_filter'] = savgol_filter(self.exp_df['CyclopeanEyeDirection.az'],
                                                                    window_length=23,polyorder=2)
            self.exp_df['CyclopeanEyeDirection.el_filter'] = savgol_filter(self.exp_df['CyclopeanEyeDirection.el'],
                                                                window_length=23,polyorder=2)
        self.sp_heuristics()
        if self.check_trials:
            fig, ax1 = plt.subplots(figsize=(15, 8),nrows=1,ncols=1)

            # ax1.plot(self.exp_df['TimeStamp'].values, self.exp_df['CyclopeanEyeDirection.az_filter'].to_numpy(), label='azimuth')
            # ax1.plot(self.exp_df['TimeStamp'].values, self.exp_df['CyclopeanEyeDirection.el_filter'].to_numpy(), label='elevation')
            # ax1.plot(self.exp_df['TimeStamp'].values, -self.exp_df['WorldDotPostion.az'],label='world dot AZ')
            # ax1.plot(self.exp_df['TimeStamp'].values, self.exp_df['WorldDotPostion.el'],label='world dot EL')
            ax1.plot(self.exp_df['CyclopeanEyeDirection.az_filter'].to_numpy(),
                     self.exp_df['CyclopeanEyeDirection.el_filter'].to_numpy(),label='eye')
            ax1.plot(self.exp_df['WorldDotPostion.az'].to_numpy(),
                     self.exp_df['WorldDotPostion.el'].to_numpy(),label='dot')
            # ax1.plot(self.exp_df['TimeStamp'].values, self.exp_df['LocalDotPostion.az'],label='local dot AZ')
            # ax1.plot(self.exp_df['TimeStamp'].values, self.exp_df['LocalDotPostion.el'],label='local dot EL')
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

    
    def sp_heuristics(self):
        self.exp_df = az_el_dot_world(self.exp_df)
        self.exp_df = az_el_dot_local(self.exp_df)

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
            dot_pos = self.exp_df['LocalDotPostion.az']
            sp_vel = az_vel
            sp_vel_dot = az_vel_dot
        else:
            sp_time_series = self.exp_df['CyclopeanEyeDirection.el_filter']
            dot_pos = self.exp_df['LocalDotPostion.el']
            sp_vel = el_vel
            sp_vel_dot = el_vel_dot

        # norm_sp = np.linalg.norm(df['col1'] - df['col2'])
        # norm_dot
        # plt.figure()
        # # plt.plot(self.exp_df['CyclopeanEyeDirection.el_filter'],self.exp_df['CyclopeanEyeDirection.az_filter'])
        # plt.plot(self.exp_df['CyclopeanEyeDirection.el_filter'])
        # plt.plot(self.exp_df['CyclopeanEyeDirection.az_filter'])
        # plt.show()
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
        if self.methods_plot:
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

    def run_analysis(self):
        if self.check_trials:
            #concussion
            for dir in tqdm(self.src):
                files = find_files(dir, "experiment_data_pID")
                # file_paths = next(iter(files.values()))
                for file_paths in files.values():
                    for file in file_paths:
                        try:
                            self.parse_condition_per(file)
                        except Exception as e:
                            print(f'{Fore.RED}file is not included in analysis: {e}{Style.RESET_ALL}')
            
            #control
            for dir in tqdm(self.control):
                files = find_files(dir, "experiment_data_pID")
                for file_paths in files.values():
                    for file in file_paths:
                        try:
                            self.parse_condition_per(file)
                        except Exception as e:
                            print(f'{Fore.RED}file is not included in analysis: {e}{Style.RESET_ALL}')
            print(f'{Fore.CYAN} TRIALS ARE VETTED. Use this command python3 vr_voms_smooth_pursuit.py {Style.RESET_ALL}')
            exit()

        #iterate over good saccades
        with open(self.good_sp_trials_path_src, 'r') as f:
            files_src= f.readlines()
        
        with open(self.good_sp_trials_path_con, 'r') as f:
            files_control = f.readlines()
        
        sp_dict_src = []
        for file in files_src:
            file = file.strip()  #remove any leading/trailing whitespace
            if os.path.isfile(file):  #check if the file exists
                all_sp = self.parse_condition_per(file)
                sp_dict_src.append(all_sp)
            else:
                print(f'{Fore.RED} File not found: {file}{Style.RESET_ALL}')
        pd.DataFrame(sp_dict_src).to_csv('src_sp.csv',index=False)

        sp_dict_con = []
        for file in files_control:
            file = file.strip()  #remove any leading/trailing whitespace
            if os.path.isfile(file):  #check if the file exists
                all_sp = self.parse_condition_per(file)
                sp_dict_con.append(all_sp)
            else:
                print(f'{Fore.RED} File not found: {file}{Style.RESET_ALL}')
        pd.DataFrame(sp_dict_con).to_csv('con_sp.csv',index=False)

def main():
    parser = argparse.ArgumentParser(description='Process Saccades from VOMS.')
    parser.add_argument('--check_trials', action='store_true', help='Check for good and bad trials.')
    parser.add_argument('--methods_plot', action='store_true', help='Methods plot for publication and grants.')
    args = parser.parse_args()

    vrVoms(check_trials=args.check_trials, 
           methods_plot=args.methods_plot).run_analysis()

if __name__ == "__main__":
    main()