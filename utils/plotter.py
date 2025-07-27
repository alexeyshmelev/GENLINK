import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu, ttest_ind, ttest_rel
import os

from os import listdir
from os.path import isdir, join

def visualize_classifier_data(data_path, fig_path=None, mask_percent=None, sort_bars=False, annotate=False, dataset_plot_only=None, class_plot_only=None, highlight_best=False, fig_size=(16, 6),
                              bar_annotation_size=10,
                              std_annotation_size=10,
                              title_size=24,
                              xlabel_size=18,
                              ylabel_size=18):   

    all_dataset_dirs = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    all_dataset_dirs = (np.array(all_dataset_dirs)[np.array(all_dataset_dirs) != 'dataset_stats']).tolist()
    if dataset_plot_only is not None:
        all_dataset_dirs = (np.array(all_dataset_dirs)[np.array(all_dataset_dirs) == dataset_plot_only]).tolist()

    if os.path.isdir(join(data_path, 'dataset_stats')):

        all_dataset_stats = [f for f in listdir(join(data_path, 'dataset_stats')) if isdir(join(join(data_path, 'dataset_stats'), f))]
        results = dict()
        for dataset_dir in all_dataset_stats:
            all_splits_per_dataset_path = join(join(data_path, 'dataset_stats'), dataset_dir)
            all_dirs = [f for f in listdir(all_splits_per_dataset_path) if isdir(join(all_splits_per_dataset_path, f))]

            for dir_path in all_dirs:
                with open(join(join(join(data_path, 'dataset_stats'), dataset_dir), dir_path+'/stats.json'), 'r') as f:
                    curr_stat = json.load(f)
                    if dataset_dir not in results.keys():
                        results[dataset_dir] = [curr_stat['number_connected_components']]
                    else:
                        results[dataset_dir].append(curr_stat['number_connected_components'])

        stats_dataset_name, stats_dataset_mean, stats_dataset_std = [], [], []
        for dataset_name, stats in results.items():
            stats_dataset_name.append(dataset_name)
            stats_dataset_mean.append(np.mean(stats))
            stats_dataset_std.append(np.std(stats))

        df = pd.DataFrame({
                'Dataset name': stats_dataset_name,
                'Number of components': np.round(stats_dataset_mean, 4),
                'std': stats_dataset_std
            })

        sns.set_theme()
        stats_bar_plot = sns.barplot(x='Dataset name', y='Number of components', data=df, color='#05F140')
        for i, (mean, std) in enumerate(zip(df['Number of components'], df['std'])):
                stats_bar_plot.errorbar(i, mean, yerr=std, fmt='none', c='black', capsize=5)

                # Optionally annotate the bars with the exact mean values
                if annotate:
                    stats_bar_plot.text(i, mean + std + 0.01, f'{std:.2f}', ha='center', va='bottom', fontsize=6)

        plt.xticks(rotation=90, ha='right', rotation_mode='anchor', verticalalignment='center', fontsize=10)
        plt.show()

    for dataset_dir in all_dataset_dirs:
        all_models_per_dataset_path = join(data_path, dataset_dir)
    
        all_dirs = [f for f in listdir(all_models_per_dataset_path) if isdir(join(all_models_per_dataset_path, f))]
        # print(all_dirs)

        results = dict()
        for dir_path in all_dirs:
            if True: #'relu' not in dir_path and 'gelu' not in dir_path and 'leaky_relu' not in dir_path and 'nw' not in dir_path:
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
                        if class_plot_only is None:
                            results[tuple([curr_res['model_name'], ft])].append(curr_res['f1_macro'])
                        else:
                            results[tuple([curr_res['model_name'], ft])].append(curr_res['class_scores'][class_plot_only])
                    else:
                        if class_plot_only is None:
                            results[tuple([curr_res['model_name'], ft])].append(curr_res['f1_macro'])
                        else:
                            results[tuple([curr_res['model_name'], ft])].append(curr_res['class_scores'][class_plot_only])

        # print(results)

        classifiers = dict()
        for name_ft, metrics in results.items():
            name, ft = name_ft
            # print(len(metrics))
            assert len(metrics) == 10
            classifiers[tuple([name, ft])] = []
            classifiers[tuple([name, ft])].append(np.mean(metrics))
            classifiers[tuple([name, ft])].append(np.std(metrics))
            classifiers[tuple([name, ft])].append(metrics)

        classifier_names, means, std_devs, all_fts, all_metrics = [], [], [], [], []
        
        for name_ft, metrics in classifiers.items():
            name, ft = name_ft
            all_fts.append(ft)
            classifier_names.append(name)
            means.append(metrics[0])
            std_devs.append(metrics[1])
            all_metrics.append(metrics[2])

        
        # Create a DataFrame for easier plotting with seaborn
        print('classifiers', len(classifier_names), len(means))
        df = pd.DataFrame({
            'Classifier': classifier_names,
            'Mean': np.round(means, 4),
            'StdDev': std_devs,
            'feature_type': all_fts,
            'all_metrics': all_metrics
        })

        # Optionally sort the bars by their mean values
        if sort_bars:
            df = df.sort_values('Mean', ascending=False)

        df = df.reset_index(drop=True)

        all_pvalues = []
        for i in range(df.shape[0]):
            if i == 0:
                all_pvalues.append(1)
            else:
                statistic, p_value = ttest_ind(df.iloc[0, 4], df.iloc[i, 4], equal_var=False)
                all_pvalues.append(p_value)

        n_tests = len(all_pvalues)
        all_pvalues = [p * n_tests for p in all_pvalues]



        # print(df)

        # Plotting
        plt.figure(figsize=fig_size)
        # sns.set(style="whitegrid")
        sns.set_theme()
        
        cols, hue_names = [], []
        color_model_scheme = {'GNN graph based':'#05F140', 'GNN one hot':'#253957', 'MLP':'#FFB400', 'Heuristics':'#00B4D8', 'Community detection': '#EF233C'}#9B7EDE #FFDF64 #5FBFF9 #FF595E
        for index, row in df.iterrows():
            # print(row)
            model_name = row['Classifier']
            ft = row['feature_type']
            if 'MLP' in model_name:
                cols.append(color_model_scheme['MLP'])
                hue_names.append('MLP')
            elif ft == 'graph_based':
                cols.append(color_model_scheme['GNN graph based'])
                hue_names.append('GNN graph based')
            elif model_name in ['MaxEdgeCount', 'MaxEdgeCountPerClassSize', 'MaxIbdSum', 'MaxIbdSumPerClassSize', 'LongestIbd', 'MaxSegmentCount']:
                cols.append(color_model_scheme['Heuristics'])
                hue_names.append('Heuristics')
            elif model_name in ['AgglomerativeClustering', 'GirvanNewmann', 'LabelPropagation', 'MultiRankWalk', 'RelationalNeighborClassifier', 'RidgeRegression', 'SpectralClustering']:
                cols.append(color_model_scheme['Community detection'])
                hue_names.append('Community detection')
            else:
                cols.append(color_model_scheme['GNN one hot'])
                hue_names.append('GNN one hot')

        bar_plot = sns.barplot(x=df.index, y=df.Mean, data=df, palette=color_model_scheme, hue=hue_names)  # , palette="viridis")

        # Collect all bars from all containers
        bars = []
        for container in bar_plot.containers:
            for bar in container:
                bars.append(bar)

        # Create a list of bars with their center x-positions
        bars_with_x = [(bar, bar.get_x() + bar.get_width() / 2) for bar in bars]

        # Sort the bars based on their x-position
        sorted_bars_with_x = sorted(bars_with_x, key=lambda x: x[1])

        # Extract the sorted bars
        sorted_bars = [bar for bar, x in sorted_bars_with_x]

        # Apply hatching to the first six bars as they appear in the plot
        # for i, bar in enumerate(sorted_bars):
        #     if all_pvalues[i] > 0.05:
        #         bar.set_hatch('//')
        #         bar.set_edgecolor('red')
        #         bar.set_linewidth(0)
            # else:
            #     print(all_pvalues[i])

        for i in range(len(bar_plot.containers)):
            bar_plot.bar_label(bar_plot.containers[i], label_type='center', rotation=90, color='white', fontsize=bar_annotation_size) # fontsize=6
        bar_plot.set_xticklabels(df.Classifier)
        # print(bar_plot.containers[0])
        

        # Adding error bars
        for i, (mean, std) in enumerate(zip(df['Mean'], df['StdDev'])):
            bar_plot.errorbar(i, mean, yerr=std, fmt='none', c='black', capsize=5)

            # Optionally annotate the bars with the exact mean values
            if annotate:
                bar_plot.text(i, mean + std + 0.01, f'{std:.2f}', ha='center', va='bottom', fontsize=std_annotation_size) # fontsize=6


        
                
        
        # for k, v in color_model_scheme.items():
        #     if k != 'Community detection':
        #         plt.scatter([],[], c=v, label=k)

        # if class_plot_only is None:
        #     plt.title(f'Model performance for {dataset_dir}', fontsize=title_size)
        # else:
        #     plt.title(f'Model performance for {dataset_dir} (class {class_plot_only})', fontsize=title_size)
        plt.xlabel('model', fontsize=xlabel_size, fontweight='bold')
        plt.ylabel('f1-macro score', fontsize=ylabel_size, fontweight='bold')
        plt.xticks(rotation=90, ha='right', rotation_mode='anchor', verticalalignment='center', fontsize=20)  # Rotate x-axis labels for better readability
        ax = plt.gca()
        # for i in range(len(ax.get_xticklabels())):
        #     if ax.get_xticklabels()[i].get_text() in ['MaxEdgeCount', 'MaxEdgeCountPerClassSize', 'MaxIbdSum', 'MaxIbdSumPerClassSize', 'LongestIbd', 'MaxSegmentCount']:
        #         break
        #     else:
        #         plt.setp(ax.get_xticklabels()[i], color='red', weight='bold')

        ax.tick_params(axis='y', labelsize=20)

        plt.legend(fontsize=15)
        plt.tight_layout()  # Adjust the layout to make room for the rotated labels
        if fig_path is not None:
            if class_plot_only is None:
                plt.savefig(f'{fig_path}/{dataset_dir}_mask_percent_{mask_percent}.pdf', bbox_inches="tight")
            else:
                plt.savefig(f'{fig_path}/{dataset_dir}_mask_percent_{mask_percent}_{class_plot_only}.pdf', bbox_inches="tight")
        plt.show()
