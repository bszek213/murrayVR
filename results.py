import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from numpy import ceil

def main():
    #SACCADES
    df_saccade_con = pd.read_csv('con_saccades.csv')
    df_saccade_src = pd.read_csv('src_saccades.csv')
    columns_to_plot = ['num_saccades', 'sacc_duration', 
                       'sacc_velocity_az', 'sacc_velocity_el',
                       'saccade_amp_az', 'saccade_amp_el']

    #equivalent sampling between each group
    min_samples = min(len(df_saccade_con), len(df_saccade_src))
    df_saccade_con = df_saccade_con.sample(n=min_samples, random_state=42)
    df_saccade_src = df_saccade_src.sample(n=min_samples, random_state=42)

    sns.set_style("whitegrid")
    sns.set_palette("deep")

    num_plots = len(columns_to_plot) #total number of plots
    num_cols = 2  #num cols in each grid
    num_rows = int(ceil(num_plots / num_cols))  #num rows in each grid

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 6 * num_rows))
    axes = axes.flatten()

    for i, column in enumerate(columns_to_plot):
        #test normality
        stat_src, p_value_src = stats.shapiro(df_saccade_src[column].dropna())
        stat_con, p_value_con = stats.shapiro(df_saccade_con[column].dropna())

        if p_value_src > 0.05 and p_value_con > 0.05:
            #both normally distributed, use t-test
            stat, p_value = stats.ttest_ind(df_saccade_src[column], df_saccade_con[column], nan_policy='omit')
            test_name = "t-test"
        else:
            #least one sample not normally distributed, use Mannu test
            stat, p_value = stats.mannwhitneyu(
                df_saccade_src[column].dropna(), 
                df_saccade_con[column].dropna(), 
                alternative='two-sided'
            )
            test_name = "Mann-Whitney U"

        sns.histplot(data=df_saccade_src, x=column, kde=True, 
                     color="skyblue", ax=axes[i], label='src')
        sns.histplot(data=df_saccade_con, x=column, kde=True, 
                     color="#FF9999", ax=axes[i], label='con',alpha=0.3) #FF9999 = light red
        
        axes[i].set_title(f"Distribution of {column.replace('_', ' ').title()}\n{test_name} p-value: {p_value:.4f}", 
                          fontsize=14)
        axes[i].set_xlabel(column.replace('_', ' ').title(), fontsize=12)
        axes[i].set_ylabel("Count", fontsize=12)
        axes[i].legend()

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.savefig('voms_saccades_src_vs_con.png',dpi=400)
    plt.close()

    #SMOOTH PURSUIT
if __name__ == "__main__":
    main()