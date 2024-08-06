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
from vr_voms_utils import find_files, az_el, detect_square_wave_periods, extract_con_and_control
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
    def __init__(self,check_trials=False, check_saccade_trials=False, methods_plot=False) -> None:
        print('vrVoms Saccades Class')
        self.all_files_dict = find_files(os.path.join(os.getcwd(),'2023_2024'), "experiment_data_pID")
        self.good_sac_trials_path_src = '/home/brianszekely/Desktop/ProjectsResearch/murray_lab/vr_eye/good_saccade_trials_src.txt'
        self.good_sac_trials_path_con = '/home/brianszekely/Desktop/ProjectsResearch/murray_lab/vr_eye/good_saccade_trials_control.txt'
        self.check_trials = check_trials
        self.check_saccade_trials = check_saccade_trials
        self.methods_plot = methods_plot
        self.src, self.control = extract_con_and_control()
        prefix = '/home/brianszekely/Desktop/ProjectsResearch/murray_lab/vr_eye/2023_2024/'
        self.src = [prefix + file_name for file_name in self.src]
        self.control = [prefix + file_name for file_name in self.control]

    def parse_condition_per(self,file):
            data = pd.read_csv(file)
            if data['Experiment'].iloc[0] == "SACCADES":
                self.exp_df = data
                # plt.figure()
                # plt.plot(self.exp_df['CyclopeanEyeDirection.x'],label='hor')
                # plt.plot(self.exp_df['CyclopeanEyeDirection.y'],label='vert')
                # plt.plot(self.exp_df['CyclopeanEyeDirection.z'],label='z')
                # plt.legend()
                # plt.show()
                # plt.close()
                all_saccades = self.saccades(file)
                return all_saccades

    def saccades(self,file):
        self.exp_df = az_el(self.exp_df)
        if 'CyclopeanEyeDirection.az_filter' not in self.exp_df.columns:
            self.exp_df['CyclopeanEyeDirection.az_filter'] = savgol_filter(self.exp_df['CyclopeanEyeDirection.az'],
                                                                    window_length=23,polyorder=2)
            self.exp_df['CyclopeanEyeDirection.el_filter'] = savgol_filter(self.exp_df['CyclopeanEyeDirection.el'],
                                                                window_length=23,polyorder=2)
        az_periods = detect_square_wave_periods(self.exp_df,'CyclopeanEyeDirection.az')
        saccades_dict, saccade_dir = self.saccade_heurstics()
        print('=================')
        print(saccade_dir)
        print('=================')
        # plt.figure()
        # plt.plot(self.exp_df['CyclopeanEyeDirection.az'],label='hor')
        # plt.plot(self.exp_df['CyclopeanEyeDirection.el'],label='vert')
        # plt.legend()
        # plt.show()
        # plt.close()
        all_saccades = {}
        sacc_velocity_az, sacc_duration, sacc_velocity_el, sacc_amp_az, sacc_amp_el = [], [], [], [], []
        for keys, items in saccades_dict.items():
            filtered_df = self.exp_df[(self.exp_df['TimeStamp'] >= items[0]) & (self.exp_df['TimeStamp'] <= items[1])]

            saccade_vel_check_az = np.nanmean(filtered_df['CyclopeanEyeDirection.az_filter'].diff().abs()
                                                     / filtered_df["TimeStamp"].diff())
            saccade_vel_check_el = np.nanmean(filtered_df['CyclopeanEyeDirection.el_filter'].diff().abs()
                                                     / filtered_df["TimeStamp"].diff())
            # if saccade_vel_check_az >= 90 or saccade_vel_check_el >= 90:
            sacc_duration.append((items[1] - items[0]) * 1000)
            if saccade_dir == "horizontal":
                sacc_velocity_az.append(saccade_vel_check_az)
                sacc_velocity_el.append(np.nan)
                sacc_amp_az.append((np.abs(filtered_df['CyclopeanEyeDirection.az_filter'].iloc[-1] - 
                                filtered_df['CyclopeanEyeDirection.az_filter'].iloc[0])) / 2) #divide by two to get the middle point as the start
                sacc_amp_el.append(np.nan)
            elif saccade_dir == "vertical":
                sacc_velocity_az.append(np.nan)
                sacc_velocity_el.append(saccade_vel_check_el)
                sacc_amp_az.append(np.nan)
                sacc_amp_el.append((np.abs(filtered_df['CyclopeanEyeDirection.az_filter'].iloc[-1] - 
                                filtered_df['CyclopeanEyeDirection.az_filter'].iloc[0])) / 2)
        
        all_saccades['num_saccades'] = len(saccades_dict)
        all_saccades['sacc_duration'] = np.mean(sacc_duration)
        all_saccades['sacc_velocity_az'] = np.mean([x for x in sacc_velocity_az if np.isfinite(x)])
        all_saccades['sacc_velocity_el'] = np.mean([x for x in sacc_velocity_el if np.isfinite(x)])
        # all_saccades['experiment'] = self.exp_df['Experiment'].iloc[0]
        all_saccades['ID'] = self.exp_df['ParticipantID'].iloc[0]
        all_saccades['direction'] = saccade_dir
        all_saccades['saccade_amp_az'] = np.mean(sacc_amp_az)
        all_saccades['saccade_amp_el'] = np.mean(sacc_amp_el)

        if self.check_trials:
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
                lines = []
                
                if os.path.exists(textfile):
                    with open(textfile, "r") as f:
                        lines = f.readlines()
                
                if filename + "\n" not in lines:
                    with open(textfile, "a") as f:
                        f.write(filename + "\n")

            def on_key(event):
                if event.key == 'y':
                    if 'CON1' in file or 'C1' in file:
                        check_and_append(file, "good_saccade_trials_src.txt")
                    elif not any(substring in file for substring in ['CON', 'PC', 'C1', 'SF']):
                        check_and_append(file, "good_saccade_trials_control.txt")
                    plt.close(fig)
                elif event.key == 'n':
                    if 'CON1' in file or 'C1' in file:
                        check_and_append(file, "bad_saccade_trials_src.txt")
                    elif not any(substring in file for substring in ['CON', 'PC', 'C1', 'SF']):
                        check_and_append(file, "bad_saccade_trials_control.txt")
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
        # plt.figure()
        # plt.plot(self.exp_df['CyclopeanEyeDirection.az_filter'],label='az')
        # plt.plot(self.exp_df['CyclopeanEyeDirection.el_filter'],label='el')
        # plt.title(f"var az {np.var(self.exp_df['CyclopeanEyeDirection.az_filter'])} || var el {np.var(self.exp_df['CyclopeanEyeDirection.el_filter'])}")
        # plt.legend() 
        # plt.show()
        if np.var(self.exp_df['CyclopeanEyeDirection.az_filter']) > np.var(self.exp_df['CyclopeanEyeDirection.el_filter']):
            saccade_time_series = self.exp_df['CyclopeanEyeDirection.az_filter']
            sacc_vel = az_vel
            saccade_dir = 'horizontal'
        else:
            saccade_time_series = self.exp_df['CyclopeanEyeDirection.el_filter']
            sacc_vel = el_vel
            saccade_dir = 'vertical'
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
        if self.methods_plot:
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
        
        return saccades_dict, saccade_dir
    
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
            print(f'{Fore.CYAN} TRIALS ARE VETTED. Use this command python3 vr_voms_saccade.py {Style.RESET_ALL}')
            exit()

        #iterate over good saccades
        with open(self.good_sac_trials_path_src, 'r') as f:
            files_src= f.readlines()
        
        with open(self.good_sac_trials_path_con, 'r') as f:
            files_control = f.readlines()
        
        saccades_dict_src = []
        for file in files_src:
            file = file.strip()  #remove any leading/trailing whitespace
            if os.path.isfile(file):  #check if the file exists
                all_saccades = self.parse_condition_per(file)
                saccades_dict_src.append(all_saccades)
            else:
                print(f'{Fore.RED} File not found: {file}{Style.RESET_ALL}')
        pd.DataFrame(saccades_dict_src).to_csv('src_saccades.csv',index=False)

        saccades_dict_con = []
        for file in files_control:
            file = file.strip()  #remove any leading/trailing whitespace
            if os.path.isfile(file):  #check if the file exists
                all_saccades = self.parse_condition_per(file)
                saccades_dict_con.append(all_saccades)
            else:
                print(f'{Fore.RED} File not found: {file}{Style.RESET_ALL}')
        pd.DataFrame(saccades_dict_con).to_csv('con_saccades.csv',index=False)
        

def main():
    parser = argparse.ArgumentParser(description='Process Saccades from VOMS.')
    parser.add_argument('--check_trials', action='store_true', help='Check for good and bad trials.')
    parser.add_argument('--check_saccade_trials', action='store_true', help='Check saccade trials specifically.')
    parser.add_argument('--methods_plot', action='store_true', help='Methods plot for publication and grants.')
    args = parser.parse_args()

    vrVoms(check_trials=args.check_trials, 
           check_saccade_trials=args.check_saccade_trials, 
           methods_plot=args.methods_plot).run_analysis()

if __name__ == "__main__":
    main()