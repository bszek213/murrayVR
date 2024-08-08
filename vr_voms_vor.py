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

class vrVoms():
    def __init__(self,check_trials=False, methods_plot=False) -> None:
        print('vrVoms VOR Class')
        self.all_files_dict = find_files(os.path.join(os.getcwd(),'2023_2024'), "experiment_data_pID")
        self.check_trials = check_trials
        self.methods_plot = methods_plot
        self.src, self.control = extract_con_and_control()
        prefix = '/home/brianszekely/Desktop/ProjectsResearch/murray_lab/vr_eye/2023_2024/'
        self.src = [prefix + file_name for file_name in self.src]
        self.control = [prefix + file_name for file_name in self.control]

    def parse_condition_per(self,file):
            data = pd.read_csv(file)
            if data['Experiment'].iloc[0] == "VOR":
                self.exp_df = data
                all_vor = self.VOR(file)
                # plt.figure()
                # plt.plot(self.exp_df['CyclopeanEyeDirection.x'],label='hor')
                # plt.plot(self.exp_df['CyclopeanEyeDirection.y'],label='vert')
                # plt.plot(self.exp_df['CyclopeanEyeDirection.z'],label='z')
                # plt.legend()
                # plt.show()
                # plt.close()
                return all_vor
            else:
                return None
    
    def VOR(self,file):
        """
        elevation needs to be inversed
        find the average amount of lag between the head and eye data using find peaks
        phase shift the data backwards by that amount of samples. 
        fill the rest with interpolation
        """
        self.exp_df = az_el(self.exp_df)

        self.exp_df['HeadOrientation.x'] = self.exp_df['HeadOrientation.x'].apply(lambda x: x - 360 if x > 100 else x)
        self.exp_df['HeadOrientation.y'] = self.exp_df['HeadOrientation.y'].apply(lambda x: x - 360 if x > 100 else x)
        self.exp_df['HeadOrientation.z'] = self.exp_df['HeadOrientation.z'].apply(lambda x: x - 360 if x > 100 else x)
        
        self.exp_df['HeadOrientation.x'] = self.exp_df['HeadOrientation.x'] - self.exp_df['HeadOrientation.x'].mean()
        self.exp_df['HeadOrientation.y'] = self.exp_df['HeadOrientation.y'] - self.exp_df['HeadOrientation.y'].mean()
        self.exp_df['HeadOrientation.z'] = self.exp_df['HeadOrientation.z'] - self.exp_df['HeadOrientation.z'].mean()

        if 'CyclopeanEyeDirection.az_filter' not in self.exp_df.columns:
            self.exp_df['CyclopeanEyeDirection.az_filter'] = savgol_filter(-self.exp_df['CyclopeanEyeDirection.az'],
                                                                    window_length=23,polyorder=2)
            self.exp_df['CyclopeanEyeDirection.el_filter'] = savgol_filter(self.exp_df['CyclopeanEyeDirection.el'],
                                                                window_length=23,polyorder=2)
        
        az_vel = self.exp_df['CyclopeanEyeDirection.az_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp']))
        az_vel = np.nan_to_num(az_vel, nan=0.0)
        el_vel = self.exp_df['CyclopeanEyeDirection.el_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp']))
        el_vel = np.nan_to_num(el_vel, nan=0.0)

        pitch_vel = self.exp_df['HeadOrientation.x'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp']))
        pitch_vel = np.nan_to_num(pitch_vel, nan=0.0)
        yaw_vel = self.exp_df['HeadOrientation.y'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp']))
        yaw_vel = np.nan_to_num(yaw_vel, nan=0.0)

        what = np.nanmean(self.exp_df['CyclopeanEyeDirection.az_filter'] / self.exp_df['HeadOrientation.x'])
        what2 = np.nanmean(self.exp_df['CyclopeanEyeDirection.el_filter'] / self.exp_df['HeadOrientation.y'])
        # print(what2)
        # print(what)
        # input()
        if not any(substring in file for substring in ['CON', 'PC', 'C1', 'SF']):
            plt.plot(self.exp_df['CyclopeanEyeDirection.az_filter'],label='azimuth')
            plt.plot(self.exp_df['CyclopeanEyeDirection.el_filter'],label='elevation')
            plt.plot(self.exp_df['HeadOrientation.x'],label='pitch')
            plt.plot(self.exp_df['HeadOrientation.y'],label='yaw')
            plt.title(file)
            plt.legend()
            plt.show()

    def run_analysis(self):
        if self.check_trials:
            #concussion
            vor_dict_src = []
            for dir in tqdm(self.src):
                files = find_files(dir, "experiment_data_pID")
                # file_paths = next(iter(files.values()))
                for file_paths in files.values():
                    for file in file_paths:
                        try:
                            vor_dict = self.parse_condition_per(file)
                            if vor_dict != None:
                                vor_dict_src.append(vor_dict)
                        except Exception as e:
                            print(f'{Fore.RED}file is not included in analysis: {e}{Style.RESET_ALL}')
            pd.DataFrame(vor_dict_src).to_csv('src_vor.csv',index=False)

            #control
            vor_dict_con = []
            for dir in tqdm(self.control):
                files = find_files(dir, "experiment_data_pID")
                for file_paths in files.values():
                    for file in file_paths:
                        try:
                            vor_dict = self.parse_condition_per(file)
                            if vor_dict != None:
                                vor_dict_con.append(vor_dict)
                        except Exception as e:
                            print(f'{Fore.RED}file is not included in analysis: {e}{Style.RESET_ALL}')
            pd.DataFrame(vor_dict_con).to_csv('con_vor.csv',index=False)

def main():
    parser = argparse.ArgumentParser(description='Process Saccades from VOMS.')
    parser.add_argument('--check_trials', action='store_true', help='Check for good and bad trials.')
    parser.add_argument('--methods_plot', action='store_true', help='Methods plot for publication and grants.')
    args = parser.parse_args()

    vrVoms(check_trials=args.check_trials, 
           methods_plot=args.methods_plot).run_analysis()

if __name__ == "__main__":
    main()
