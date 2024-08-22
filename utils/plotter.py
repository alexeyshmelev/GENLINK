import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
import numpy as np

from os import listdir
from os.path import isdir, join

def visualize_classifier_data(data_path, fig_path=None, weight_type=None, mask_percent=None, sort_bars=False, annotate=False):   

    all_dataset_dirs = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    for dataset_dir in all_dataset_dirs:
        all_models_per_dataset_path = join(data_path, dataset_dir)
    
        all_dirs = [f for f in listdir(all_models_per_dataset_path) if isdir(join(all_models_per_dataset_path, f))]
        # print(all_dirs)

        results = dict()
        for dir_path in all_dirs:
            with open(join(all_models_per_dataset_path, dir_path+'/results.json'), 'r') as f:
                curr_res = json.load(f)
                if 'graph_based' in dir_path:
                    ft = 'graph_based'
                elif 'one_hot' in dir_path:
                    ft = 'one_hot'
                else:
                    ft = ''
                if tuple([curr_res['model_name'], ft]) not in results.keys():
                    results[tuple([curr_res['model_name'], ft])] = []
                    results[tuple([curr_res['model_name'], ft])].append(curr_res['f1_macro'])
                else:
                    results[tuple([curr_res['model_name'], ft])].append(curr_res['f1_macro'])

        # print(results)

        classifiers = dict()
        for name_ft, metrics in results.items():
            name, ft = name_ft
            # print(len(metrics))
            classifiers[tuple([name, ft])] = []
            classifiers[tuple([name, ft])].append(np.mean(metrics))
            classifiers[tuple([name, ft])].append(np.std(metrics))

        classifier_names, means, std_devs, all_fts = [], [], [], []
        
        for name_ft, metrics in classifiers.items():
            name, ft = name_ft
            all_fts.append(ft)
            classifier_names.append(name)
            means.append(metrics[0])
            std_devs.append(metrics[1])

        
        # Create a DataFrame for easier plotting with seaborn
        print('classifiers', len(classifier_names), len(means))
        df = pd.DataFrame({
            'Classifier': classifier_names,
            'Mean': means,
            'StdDev': std_devs,
            'feature_type': all_fts
        })

        # Optionally sort the bars by their mean values
        if sort_bars:
            df = df.sort_values('Mean', ascending=False)

        df = df.reset_index(drop=True)

        # print(df)

        # Plotting
        plt.figure(figsize=(14, 6))
        # sns.set(style="whitegrid")
        sns.set_theme()
        
        cols = []
        color_model_scheme = {'GNN graph based':'#05F140', 'GNN one hot':'#253957', 'MLP':'#FFB400', 'Heuristics':'#00B4D8', 'Community detection': '#EF233C'}#9B7EDE #FFDF64 #5FBFF9 #FF595E
        for index, row in df.iterrows():
            # print(row)
            model_name = row['Classifier']
            ft = row['feature_type']
            if 'MLP' in model_name:
                cols.append(color_model_scheme['MLP'])
            elif ft == 'graph_based':
                cols.append(color_model_scheme['GNN graph based'])
            elif model_name in ['MaxEdgeCount', 'MaxEdgeCountPerClassSize', 'MaxIbdSum', 'MaxIbdSumPerClassSize', 'LongestIbd', 'MaxSegmentCount']:
                cols.append(color_model_scheme['Heuristics'])
            elif model_name in ["Spectral clustering", "Agglomerative clustering", "Girvan-Newman", "Label propagation", "Relational neighbor classifier", "Multi-Rank-Walk", "Ridge regression"]:
                cols.append(color_model_scheme['Community detection'])
            else:
                cols.append(color_model_scheme['GNN one hot'])

        bar_plot = sns.barplot(x=df.index, y=df.Mean, data=df, ci=None, palette=cols)  # , palette="viridis")
        bar_plot.set_xticklabels(df.Classifier)

        # Adding error bars
        for i, (mean, std) in enumerate(zip(df['Mean'], df['StdDev'])):
            bar_plot.errorbar(i, mean, yerr=std, fmt='none', c='black', capsize=5)

            # Optionally annotate the bars with the exact mean values
            if annotate:
                bar_plot.text(i, mean + std + 0.01, f'{mean:.2f}', ha='center', va='bottom', fontsize=6)
                
        
        for k, v in color_model_scheme.items():
            plt.scatter([],[], c=v, label=k)

        plt.title(f'Model performance for {dataset_dir}')
        plt.xlabel('Model')
        plt.ylabel('Mean f1-macro score')
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor', verticalalignment='center')  # Rotate x-axis labels for better readability
        plt.legend()
        plt.tight_layout()  # Adjust the layout to make room for the rotated labels
        if fig_path is not None:
            plt.savefig(f'{fig_path}/{dataset_dir}_mask_percent_{mask_percent}.pdf', bbox_inches="tight")
        plt.show()
