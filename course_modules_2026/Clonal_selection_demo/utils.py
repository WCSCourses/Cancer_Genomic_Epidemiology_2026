#!/usr/bin/env python

import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib as mpl
import matplotlib.ticker as mticker

from scipy.stats import norm

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# from plot_selection_omega import plot_omega_vertical, build_counts_from_df_complete



plots_general_config = {

                        # fonsizes
                        "ylabel_fontsize": 6,
                        "xlabel_fontsize": 6,
                        "xylabel_fontsize": 6,
                        "title_fontsize": 7,
                        "xyticks_fontsize": 5,
                        "xticks_fontsize": 5,
                        "yticks_fontsize": 5,
                        "legend_fontsize": 5,
                        "annots_fontsize": 5,

                        "dot_size_scplot": 15,
                        "dot_size_coeffplot": 5,
                        "dot_sizebelow_coeffplot": 40,
                        "dot_color_coeffplot": "#D3D3D3",
                        "dot_colorabove_coeffplot": "#D62728",
                        "dot_colorbelow_coeffplot": "#f29c9e",
                        "dot_edgethres_coeffplot": 0.2,
                        "dot_edgewidth_coeffplot": 0.5
                        }


metrics_colors_dictionary = {"ofml"        : "viridis_r",
                                "ofml_score"  : "#6A33E0",
                                "omega_trunc" : "#FA5E32",
                                "omega_synon" : "#89E4A2",
                                "omega_miss"  : "#FABE4A",
                                "o3d_score"   : "#6DBDCC",
                                "o3d_cluster" : "skyblue",
                                "o3d_prob"    : "darkgray",
                                "frameshift"  : "#E4ACF4",
                                "inframe"     : "C5",
                                "hv_lines"    : "lightgray", # horizontal and vertical lines,
                                "hv_lines_needle" : "gray",
                                "needle_obs"  : "#003366",
                                "omega_miss_tert" : "#f5840c",
                                "omega_synon_tert": "#378c12",
                                "nonsense" : "#FA5E32",
                                "synonymous" : "#89E4A2",
                                "missense"  : "#FABE4A",
                                "indel"       : "#ECC4F7",
                                "splicing"    : "#A1C5DF",
                                }


custom_na_values_list = [" ", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
                    "1.#IND", "1.#QNAN", "<NA>", "N/A",
                    # "NA",
                    "NULL", "NaN", "None",
                    "n/a", "nan", "null "]

custom_na_values_list = ["Allele", "Gene", "Feature", "Feature_type", "Amino_acids", "Codons",
                            "Existing_variation", "FLAGS", "SYMBOL", "SYMBOL_SOURCE", "HGNC_ID",
                            "canonical_Allele", "canonical_Gene", "canonical_Feature", "canonical_Feature_type",
                            "canonical_Amino_acids", "canonical_Codons", "canonical_Existing_variation",
                            "canonical_FLAGS", "canonical_SYMBOL", "canonical_SYMBOL_SOURCE", "canonical_HGNC_ID",
                            "INVENTED_COLUMN"]

custom_na_values = {col : custom_na_values_list for col in custom_na_values_list}



mpl.rcParams.update({
    'axes.titlesize'    : plots_general_config["title_fontsize"],       # Title font size
    'axes.labelsize'    : plots_general_config["xylabel_fontsize"],     # X and Y axis labels
    'xtick.labelsize'   : plots_general_config["xyticks_fontsize"],     # X tick labels
    'ytick.labelsize'   : plots_general_config["xyticks_fontsize"],     # Y tick labels
    'legend.fontsize'   : plots_general_config["legend_fontsize"],      # Legend text
    'figure.titlesize'  : plots_general_config["title_fontsize"],       # Figure suptitle (if used)
})





def generate_all_side_figures(sample,
                                outdir = '.',
                                gene_list = None,
                                data_path = None,
                                tools = ["oncodrivefml", "omega_trunc", "omega_mis"]
                                ):

    maf_file = f"{data_path}/{sample}.somatic.mutations.tsv"
    if not os.path.exists(maf_file):
        print(f"Warning: MAF file {maf_file} not found. Skipping side figures generation.")
        return
    
    maf = pd.read_table(maf_file, na_values = custom_na_values)
    snvs_maf = maf[maf["TYPE"] == "SNV"].reset_index(drop = True)

    possible_genes = []
    oncodrivefml_genes = []
    omega_truncating_genes = []
    omega_missense_genes = []
    indels_genes = []
    
    # Check and load oncodrivefml data
    if "oncodrivefml" in tools:
        oncodrivefml_file = f"{data_path}/{sample}-oncodrivefml.tsv.gz"
        if os.path.exists(oncodrivefml_file):
            oncodrivefml_data = pd.read_table(oncodrivefml_file)
            oncodrivefml_data = oncodrivefml_data[["GENE_ID", "Z-SCORE", "Q_VALUE", "AVG_SCORE_OBS", "POPULATION_MEAN", "STD_OF_MEANS"]]
            oncodrivefml_data.columns = ["GENE", "OncodriveFML", "pvalue", "OBSERVED_MEAN", "BACKGROUND_MEAN", "BACKGROUND_STD"]
            oncodrivefml_genes = list(pd.unique(oncodrivefml_data["GENE"]))
            possible_genes += oncodrivefml_genes
        else:
            print(f"Warning: OncodriveFML file {oncodrivefml_file} not found. Skipping OncodriveFML plots.")


    # Check and load omega data
    if "omega_trunc" in tools or "omega_mis" in tools:
        omega_file = f"{data_path}/all_omega_values.tsv"
        if os.path.exists(omega_file):
            omega_data = pd.read_table(omega_file)
            omega_data = omega_data[(omega_data["impact"].isin(['missense', 'truncating']))
                                        & (omega_data["sample"] == sample)
                                        & ~(omega_data["gene"].str.contains('--'))  # select only genes
                                    ]
            if "omega_trunc" in tools :
                omega_truncating = omega_data[omega_data["impact"] == "truncating"].reset_index(drop = True)[["gene", "mutations", "dnds", "pvalue", "lower", "upper"]]
                omega_truncating.columns = ["GENE", "mutations_trunc", "omega_trunc", "pvalue", "lower", "upper"]
                omega_truncating_genes = list(pd.unique(omega_truncating["GENE"]))
                possible_genes += omega_truncating_genes

            if "omega_mis" in tools :
                omega_missense = omega_data[omega_data["impact"] == "missense"].reset_index(drop = True)[["gene", "mutations", "dnds", "pvalue", "lower", "upper"]]
                omega_missense.columns = ["GENE", "mutations_mis", "omega_mis", "pvalue", "lower", "upper"]
                omega_missense_genes = list(pd.unique(omega_missense["GENE"]))
                possible_genes += omega_missense_genes
        else:
            print(f"Warning: Omega file {omega_file} not found. Skipping Omega plots.")


    # Check and load indels data
    if "excess_indels" in tools:
        indels_file = f"{data_path}/{sample}.sample.indels.tsv"
        if os.path.exists(indels_file):
            indels_data = pd.read_table(indels_file,
                                            sep = '\t',
                                            header = 0)
            indels_genes = list(pd.unique(indels_data["SYMBOL"]))
            possible_genes += indels_genes
        else:
            print(f"Warning: Indels file {indels_file} not found. Skipping indels plots.")

    valid_genes = list(set(possible_genes).intersection(set(snvs_maf["canonical_SYMBOL"].unique())))
    if gene_list is None:
        gene_list = valid_genes
    else:
        gene_list = [g for g in gene_list if g in valid_genes]

    for genee in gene_list:
        print(genee, end = '\t')
        try :

            if "oncodrivefml" in tools:
                # there is no run of oncodrivefml with ALL_GENES
                if genee in oncodrivefml_genes:
                    oncodrivefml_gene_data = oncodrivefml_data[oncodrivefml_data["GENE"] == genee].to_dict(orient='records')[0]

                    fig_gene_fml = plot_oncodrivefml_side(oncodrivefml_gene_data)
                    fig_gene_fml.savefig(f"{outdir}/{genee}.{sample}.oncodrivefml.pdf", bbox_inches='tight', dpi = 100)
                    plt.show()
                    plt.close()
                    print("ofml done", end = '\t')

            if "omega_trunc" in tools:
                if genee in omega_truncating_genes and genee in omega_missense_genes:
                    omega_df = build_counts_from_df_complete(genee, snvs_maf, omega_truncating, omega_missense)

                    fig_gene_omega = plot_omega_vertical(omega_df)
                    fig_gene_omega.savefig(f"{outdir}/{genee}.{sample}.omega.pdf", bbox_inches='tight', dpi = 100)
                    plt.show()
                    plt.close()
                    print("omega done", end = '\t')

        except Exception as exe:
            print("failed processing of")
            print(genee)
            print(exe)






def plot_oncodrivefml_side(geneee_data):
    legend_fontsize = 12
    xaxis_fontsize = 12

    # Extract the necessary values
    observed_mean = geneee_data['OBSERVED_MEAN']
    background_mean = geneee_data['BACKGROUND_MEAN']
    background_std = geneee_data['BACKGROUND_STD']
    p_value = geneee_data['pvalue']

    observed_color = metrics_colors_dictionary["ofml_score"]

    deviation = abs(observed_mean - background_mean)

    # Calculate the Z-score
    z_score = (observed_mean - background_mean) / background_std

    # Generate a range of values for the x-axis
    x = np.linspace(background_mean - 1*background_std - deviation, background_mean + 1*background_std + deviation, 1000)
    # Generate the normal distribution based on background mean and std
    y = norm.pdf(x, background_mean, background_std)

    # Compute half of the maximum value
    mid_y = max(y) / 2

    # Plot the normal distribution vertically
    fig, ax = plt.subplots(figsize=(3, 1.75))

    background_color = 'dimgrey'
    background_color_line = 'dimgrey'
    ax.plot(x, y, color = background_color)
    ax.fill_betweenx(y, x, color = background_color, alpha=0.2) #, label = 'Randomized\nmeans' )

    # Arrows with annotations
    arrow_props = dict(facecolor= observed_color, edgecolor = observed_color, arrowstyle='<->')

    if z_score > 0:
        ax.set_xlim(background_mean - (1*background_std + deviation) / 2, background_mean + (1*background_std + deviation))
        # ax.text(background_mean + background_std, mid_y * 1.75, f'Randomized\nmeans', color=background_color, ha='left', va='top')
        #ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.2, 1.1), labelcolor = background_color)

        # # Set integer tick labels on the x-axis
        # x_ticks = np.linspace(background_mean - (1*background_std + deviation) / 2, background_mean + (1*background_std + deviation), num=3)
        # x_ticks_int = np.round(x_ticks).astype(int)
        # ax.set_xticks(x_ticks_int)

        # Arrow 1
        ax.annotate('', xy=(background_mean, mid_y), xytext=(observed_mean, mid_y), arrowprops=arrow_props)

    else:
        ax.set_xlim(background_mean - (1*background_std + deviation), background_mean + (1*background_std + deviation) / 2)
        # ax.text(background_mean - background_std, mid_y * 1.75, f'Randomized\nmeans', color=background_color, ha='left', va='top')
        # Add legend without border
        #ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.2, 1.1), labelcolor = background_color)

        # # Set integer tick labels on the x-axis
        # x_ticks = np.linspace(background_mean - (1*background_std + deviation), background_mean + (1*background_std + deviation) / 2, num=3)
        # x_ticks_int = np.round(x_ticks).astype(int)
        # ax.set_xticks(x_ticks_int)

        # Arrow
        ax.annotate('', xy=(background_mean, mid_y), xytext=(observed_mean, mid_y), arrowprops=arrow_props)



    # Legend
    legend_handles = [Patch(facecolor=background_color_line, alpha=0.2, edgecolor='none', label='Randomized means'),
                        Line2D([0], [0], color="black", linestyle='--', label='Observed mean')]
    legend = ax.legend(handles=legend_handles, frameon=False, loc='upper right', bbox_to_anchor=(1.6, 1.1), fontsize=legend_fontsize)

    # Adjust the color of the text labels in the legend
    for text, color in zip(legend.get_texts(), ["black", "black"]):
        text.set_color(color)


    ax.set_xticks([])

    # Add a vertical line for the observed mean
    ax.axvline(observed_mean, color="black", linestyle='--', ymin=0, ymax=0.5, label="Observed mean")

    # Add a label for the observed mean
    if p_value == 1e-6:
        #ax.text(observed_score*1.1, max(y)/2, text, ha='left', va='center', fontsize=text_fontsize, color=observed_color)
        #ax.text(observed_score*1.1, max(y)/2 - 0.35*(max(y)/2), fr'$\mathit{{p}}$-value < {pvalue}', ha='left', va='center', fontsize=text_fontsize, color=observed_color)
        ax.text(observed_mean * 1.03, mid_y, f'$Score$ = {z_score:.2f}',
                color=observed_color, ha='left', va='center', fontsize = 13)
        ax.text(observed_mean * 1.03, mid_y - 0.35*mid_y, f'$p$-value < {p_value:.2g}',
                color=observed_color, ha='left', va='center', fontsize = 13)
    else:
        ax.text(observed_mean * 1.03, mid_y, f'$Score$ = {z_score:.2f}',
                color=observed_color, ha='left', va='center', fontsize = 13)
        ax.text(observed_mean * 1.03, mid_y - 0.35*mid_y, f'$p$-value = {p_value:.2g}',
                color=observed_color, ha='left', va='center', fontsize = 13)

    # Set labels and title
    ax.set_xlabel('Impact score', fontsize = xaxis_fontsize)

    # Hide the bottom, right, and top borders (spines) of the plot
    for spine in ['left', 'right', 'top']:
        ax.spines[spine].set_visible(False)

    # Hide the entire y-axis
    ax.yaxis.set_visible(False)

    # Display the plot
    plt.show()


    return fig





# ## __Define functions__

def plot_all_positive_selection(omega_truncating,
                                omega_missense,
                                indels_panel_df,
                                oncodrive3d_data_scores,
                                oncodrivefml_data,
                                gene_order,
                                title = None,
                                pvalue_thres = 0.05,
                                linewidth_def = 0.6,
                                tracks = ("omega_trunc", "omega_mis", "oncodrive3d", "oncodrivefml")
                                ):

    num_genes = len(gene_order)
    # Determine which tracks to plot and their order
    all_tracks = ["omega_trunc", "omega_mis", "oncodrive3d", "oncodrivefml", "indels"]
    plot_tracks = [t for t in all_tracks if t in tracks]
    n_tracks = len(plot_tracks)
    
    # Check if we have any tracks to plot
    if n_tracks == 0:
        print("Warning: No tracks to plot. Skipping plot generation.")
        return None
    
    fig, axes = plt.subplots(n_tracks, 1, figsize=(2.5, 0.6 + 0.6 * n_tracks), gridspec_kw={'height_ratios': [5]*n_tracks})
    if n_tracks == 1:
        axes = [axes]
    if title:
        fig.suptitle(title)
    ax_idx = 0

    if "omega_trunc" in plot_tracks and omega_truncating is not None:
        ax = axes[ax_idx]
        omega_truncating_sig = omega_truncating[omega_truncating["pvalue"] <= pvalue_thres].reset_index(drop = True)
        omega_truncating_notsig = omega_truncating[omega_truncating["pvalue"] > pvalue_thres].reset_index(drop = True)
        sns.barplot(data=omega_truncating_notsig, x='GENE', y='omega_trunc',
                    ax=ax, alpha=1,
                    fill = False,
                    legend = False,
                    linewidth = linewidth_def,
                    order = gene_order,
                    color = metrics_colors_dictionary["omega_trunc"])
        sns.barplot(data=omega_truncating_sig, x='GENE', y='omega_trunc',
                    ax=ax, alpha=1,
                    legend = False,
                    linewidth = linewidth_def,
                    order = gene_order,
                    color = metrics_colors_dictionary["omega_trunc"],
                    edgecolor = None
                    )
        ax.set_xlabel('')
        ax.set_ylabel('dN/dS of\ntruncating', rotation = 0, labelpad=17, verticalalignment = 'center')
        # Only set xticklabels on last axis
        if ax_idx == n_tracks - 1:
            ax.set_xticks(range(num_genes))
            ax.set_xticklabels(gene_order, rotation=90)
        else:
            ax.set_xticks(range(len(gene_order)))
            ax.set_xticklabels([])
        ax.axhline(1, color='black', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax_idx += 1




    if "omega_mis" in plot_tracks and omega_missense is not None:
        ax = axes[ax_idx]
        omega_missense_sig = omega_missense[omega_missense["pvalue"] <= pvalue_thres].reset_index(drop = True)
        omega_missense_notsig = omega_missense[omega_missense["pvalue"] > pvalue_thres].reset_index(drop = True)
        sns.barplot(data=omega_missense_notsig, x='GENE', y='omega_mis',
                    ax=ax, alpha=1,
                    fill = False,
                    legend = False,
                    linewidth = linewidth_def,
                    order = gene_order,
                    color = metrics_colors_dictionary["omega_miss"])
        sns.barplot(data=omega_missense_sig, x='GENE', y='omega_mis',
                    ax=ax, alpha=1,
                    legend = False,
                    linewidth = linewidth_def,
                    order = gene_order,
                    color = metrics_colors_dictionary["omega_miss"],
                    edgecolor = None
                    )
        ax.set_xlabel('')
        ax.set_ylabel('dN/dS of\nmissense', rotation = 0, labelpad=17, verticalalignment = 'center')
        if ax_idx == n_tracks - 1:
            ax.set_xticks(range(num_genes))
            ax.set_xticklabels(gene_order, rotation=90)
        else:
            ax.set_xticks(range(len(gene_order)))
            ax.set_xticklabels([])
        ax.axhline(1, color='black', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax_idx += 1



    if "oncodrive3d" in plot_tracks and oncodrive3d_data_scores is not None:
        ax = axes[ax_idx]
        df = oncodrive3d_data_scores
        name_metric = "o3d_score"
        variable_name = df.columns[1]
        max_score = df[variable_name].max()
        for j, gene in enumerate(gene_order):
            try:
                value_original = df.loc[df['GENE'] == gene, variable_name].values[0]
                value =  value_original / max_score * 5
                pvalue = df.loc[df['GENE'] == gene, 'pvalue'].values[0]
                color = metrics_colors_dictionary[name_metric] if pvalue < pvalue_thres else 'none'
                edgecolor = metrics_colors_dictionary[name_metric]
                size = value * 12  # Scale size for better visualization
                ax.scatter(j, 0, s=size, color=color, edgecolors=edgecolor, linewidth = linewidth_def, alpha=0.9,)
            except Exception as e:
                print("Gene", gene, "failed because of", e)
                continue
        if ax_idx == n_tracks - 1:
            ax.set_xticks(range(len(gene_order)))
            ax.set_xticklabels(gene_order, rotation=90)
        else:
            ax.set_xticks(range(len(gene_order)))
            ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel('3D\nclustering', rotation = 0, labelpad=17,
                        verticalalignment = 'center',
                        horizontalalignment = 'right'
                        )
        ax.set_ylim(-0.5, 0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('')
        ax_idx += 1




    if "oncodrivefml" in plot_tracks and oncodrivefml_data is not None:
        ax = axes[ax_idx]
        df = oncodrivefml_data
        name_metric = "ofml_score"
        variable_name = df.columns[1]
        max_score = df[variable_name].max()
        for j, gene in enumerate(gene_order):
            try:
                value_original = df.loc[df['GENE'] == gene, variable_name].values[0]
                value =  value_original / max_score * 5
                pvalue = df.loc[df['GENE'] == gene, 'pvalue'].values[0]
                color = metrics_colors_dictionary[name_metric] if pvalue < pvalue_thres else 'none'
                edgecolor = metrics_colors_dictionary[name_metric]
                size = value * 12  # Scale size for better visualization
                if size > 0:
                    ax.scatter(j, 0, s=size, color=color, edgecolors=edgecolor, linewidth = linewidth_def, alpha=0.9)
                else:
                    ax.scatter(j, 0, s=-size, color=color, edgecolors=edgecolor, linewidth = linewidth_def, linestyle = '--', alpha=0.9)
            except Exception as e:
                print("Gene", gene, "failed because of", e)
                continue
        if ax_idx == n_tracks - 1:
            ax.set_xticks(range(len(gene_order)))
            ax.set_xticklabels(gene_order, rotation=90)
        else:
            ax.set_xticks(range(len(gene_order)))
            ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel('Functional\nimpact\nbias', rotation = 0, labelpad=17,
                        verticalalignment = 'center',
                        horizontalalignment = 'right'
                        )
        ax.set_ylim(-0.5, 0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax_idx += 1




    if "indels" in plot_tracks and indels_panel_df is not None:
        ax = axes[ax_idx]
        indels_panel_df_sig = indels_panel_df[indels_panel_df["pvalue"] <= pvalue_thres].reset_index(drop = True)
        indels_panel_df_notsig = indels_panel_df[indels_panel_df["pvalue"] > pvalue_thres].reset_index(drop = True)
        sns.barplot(data=indels_panel_df_notsig, x='GENE', y='Indels_score',
                    ax=ax, alpha=1, #0.6,
                    fill = False,
                    legend = False,
                    linewidth = linewidth_def,
                    order = gene_order,
                    color = metrics_colors_dictionary["frameshift"])
        sns.barplot(data=indels_panel_df_sig, x='GENE', y='Indels_score',
                    ax=ax, alpha=1,
                    legend = False,
                    linewidth = linewidth_def,
                    order = gene_order,
                    color = metrics_colors_dictionary["frameshift"],
                    edgecolor = None
                    )
        ax.set_xlabel('')
        ax.set_ylabel('Excess of\nframeshift\nindels', rotation = 0,
                        verticalalignment = 'center',
                        horizontalalignment = 'right'
                        )
        if ax_idx == n_tracks - 1:
            ax.set_xticks(range(num_genes))
            ax.set_xticklabels(gene_order, rotation=90)
        else:
            ax.set_xticks(range(len(gene_order)))
            ax.set_xticklabels([])
        ax.axhline(1, color='black', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax_idx += 1

    # Set consistent x-axis limits for all subplots
    separation = 2
    for ax in axes:
        ax.set_xlim([-separation, num_genes - 1 + separation])

    return fig


def get_all_data(sample, outdir,
                 pvaluee = 0.05,
                 tracks=("omega_trunc", "omega_mis", "oncodrive3d", "oncodrivefml"),
                 gene_order = None,
                 data_path = None
                 ):

    # Initialize variables for optional data
    oncodrivefml_data = None
    omega_truncating = None
    omega_missense = None
    oncodrive3d_data_scores = None
    indels_panel_df = None
    global_omega_decreasing = []
    available_tracks = []

    # Check and load oncodrivefml data
    oncodrivefml_file = f"{data_path}/{sample}-oncodrivefml.tsv.gz"
    if os.path.exists(oncodrivefml_file) and "oncodrivefml" in tracks:
        try:
            oncodrivefml_data = pd.read_table(oncodrivefml_file)
            oncodrivefml_data = oncodrivefml_data[["GENE_ID", "Z-SCORE", "Q_VALUE", "AVG_SCORE_OBS", "POPULATION_MEAN", "STD_OF_MEANS"]]
            oncodrivefml_data.columns = ["GENE", "OncodriveFML", "pvalue", "OBSERVED_MEAN", "BACKGROUND_MEAN", "BACKGROUND_STD"]
            available_tracks.append("oncodrivefml")
            print(f"Loaded OncodriveFML data from {oncodrivefml_file}")
        except Exception as e:
            print(f"Warning: Failed to load OncodriveFML data: {e}")
    else:
        print(f"Warning: OncodriveFML file {oncodrivefml_file} not found. Skipping OncodriveFML track.")

    # Check and load omega data
    omega_file = f"{data_path}/all_omega_values.tsv"
    if os.path.exists(omega_file) and ("omega_trunc" in tracks or "omega_mis" in tracks):
        try:
            omega_data = pd.read_table(omega_file)
            omega_data = omega_data[(omega_data["impact"].isin(['missense', 'truncating']))
                                    & (omega_data["sample"] == sample)
                                    & ~(omega_data["gene"].str.contains('--'))  # select only genes
                                    ]
            
            if "omega_trunc" in tracks:
                omega_truncating = omega_data[omega_data["impact"] == "truncating"].reset_index(drop = True)[["gene", "dnds", "pvalue", "lower", "upper"]]
                omega_truncating.columns = ["GENE", "omega_trunc", "pvalue", "lower", "upper"]
                truncating_decreasing = list(omega_truncating.sort_values("omega_trunc", ascending= False)["GENE"].values)
                print("Truncating\n", truncating_decreasing)
                available_tracks.append("omega_trunc")

            if "omega_mis" in tracks:
                omega_missense = omega_data[omega_data["impact"] == "missense"].reset_index(drop = True)[["gene", "dnds", "pvalue", "lower", "upper"]]
                omega_missense.columns = ["GENE", "omega_mis", "pvalue", "lower", "upper"]
                missense_decreasing = list(omega_missense.sort_values("omega_mis", ascending= False)["GENE"].values)
                print("Missense\n", missense_decreasing)
                available_tracks.append("omega_mis")

            # merge omegas to decide sorting if both are available
            if omega_truncating is not None and omega_missense is not None:
                omega_df = omega_truncating[["GENE", "omega_trunc", "pvalue"]].merge(omega_missense[["GENE", "omega_mis", "pvalue"]],
                                                                                        on = ["GENE"],
                                                                                        suffixes = ("_trunc", "_mis"))
                omega_df["mean_omega"] = omega_df[["omega_trunc", "omega_mis"]].mean(axis = 1)
                omega_df["any_signif"] = omega_df[["pvalue_trunc", "pvalue_mis"]].apply(lambda x: (x < 0.05).any(), axis = 1)
                global_omega_decreasing = list(omega_df.sort_values(by = ["any_signif", "mean_omega"], ascending = False)["GENE"].values)
                if "ALL_GENES" in global_omega_decreasing:
                    global_omega_decreasing.remove("ALL_GENES")

                print("Global\n", global_omega_decreasing)
                if len(global_omega_decreasing) > 20:
                    print("Keeping top 20 genes")
                    global_omega_decreasing = global_omega_decreasing[:20]

                positively_selected_trunc = omega_truncating[(omega_truncating["pvalue"] < pvaluee) &
                                                                (omega_truncating["omega_trunc"] > 1)
                                                            ]["GENE"].values
                positively_selected_mis = omega_missense[(omega_missense["pvalue"] < pvaluee) &
                                                            (omega_missense["omega_mis"] > 1)
                                                        ]["GENE"].values

                all_positively_selected = set(positively_selected_trunc).union(set(positively_selected_mis))
                print( "all_positively_selected", sorted(all_positively_selected))
                positively_selected_both = set(positively_selected_trunc).intersection(set(positively_selected_mis))
                print( "positively_selected_both", sorted(positively_selected_both))
                positively_selected_trunc_only = set(positively_selected_trunc) - set(positively_selected_mis)
                print( "positively_selected_trunc_only", sorted(positively_selected_trunc_only))
                positively_selected_mis_only = set(positively_selected_mis) - set(positively_selected_trunc)
                print( "positively_selected_mis_only", sorted(positively_selected_mis_only))
            elif omega_truncating is not None:
                global_omega_decreasing = list(omega_truncating.sort_values("omega_trunc", ascending=False)["GENE"].values)
                if "ALL_GENES" in global_omega_decreasing:
                    global_omega_decreasing.remove("ALL_GENES")
                if len(global_omega_decreasing) > 20:
                    global_omega_decreasing = global_omega_decreasing[:20]
            elif omega_missense is not None:
                global_omega_decreasing = list(omega_missense.sort_values("omega_mis", ascending=False)["GENE"].values)
                if "ALL_GENES" in global_omega_decreasing:
                    global_omega_decreasing.remove("ALL_GENES")
                if len(global_omega_decreasing) > 20:
                    global_omega_decreasing = global_omega_decreasing[:20]
                    
            print(f"Loaded Omega data from {omega_file}")
        except Exception as e:
            print(f"Warning: Failed to load Omega data: {e}")
    else:
        print(f"Warning: Omega file {omega_file} not found. Skipping Omega tracks.")

    # Check and load oncodrive3d data
    oncodrive3d_file = f"{data_path}/{sample}.3d_clustering_genes.csv"
    if os.path.exists(oncodrive3d_file) and "oncodrive3d" in tracks:
        try:
            oncodrive3d_data = pd.read_table(oncodrive3d_file, sep = ',')
            oncodrive3d_data_scores = oncodrive3d_data[["Gene", "Score_obs_sim_top_vol", "qval"]]
            oncodrive3d_data_scores.columns = ["GENE", "Oncodrive3D", 'pvalue']
            available_tracks.append("oncodrive3d")
            print(f"Loaded Oncodrive3D data from {oncodrive3d_file}")
        except Exception as e:
            print(f"Warning: Failed to load Oncodrive3D data: {e}")
    else:
        print(f"Warning: Oncodrive3D file {oncodrive3d_file} not found. Skipping Oncodrive3D track.")


    # Check if we have any data to plot
    if not available_tracks:
        print("Warning: No data files found for any of the requested tracks. Skipping plot generation.")
        return

    print(f"Generating plot with available tracks: {available_tracks}")
    
    # Generate plot with available tracks
    figuree = plot_all_positive_selection(omega_truncating,
                                            omega_missense,
                                            indels_panel_df,
                                            oncodrive3d_data_scores,
                                            oncodrivefml_data,
                                            global_omega_decreasing if gene_order is None else gene_order,
                                            title = sample,
                                            pvalue_thres = pvaluee,
                                            tracks = tuple(available_tracks))

    if figuree is not None:
        figuree.savefig(f"{outdir}/{sample}.positive_selection_summary.pdf", bbox_inches='tight')
    else:
        print("Warning: No figure was generated due to missing data or tracks.")








def get_counts_per_position_n_consequence(somatic_maf_file):
    somatic_maf = pd.read_table(somatic_maf_file, na_values = custom_na_values)

    somatic_maf_clean = somatic_maf[(somatic_maf["TYPE"] == 'SNV')
                                    & (~somatic_maf["FILTER.not_in_exons"])
                                    & (somatic_maf['canonical_Protein_position'] != '-')
                                    ].reset_index(drop = True)
    somatic_maf_clean['canonical_Protein_position'] = somatic_maf_clean['canonical_Protein_position'].astype(int)
    counts_per_position = somatic_maf_clean.groupby(by = ["SAMPLE_ID", "canonical_SYMBOL", 'canonical_Consequence_broader', 'canonical_Protein_position'])['ALT_DEPTH'].size().to_frame('Count').reset_index()
    counts_per_position.columns = ["SAMPLE_ID", 'Gene', 'Consequence', 'Pos', 'Count']

    return counts_per_position


def plot_count_track(count_df,
                        gene_len,
                        axes,
                        colors_dict,
                        ax=0,
                        alpha=1,
                        indel=False,
                        n_batches = 10
                    ):

    # Shuffle the data and split into batches
    shuffled_df = count_df.sample(frac=1, random_state=42).reset_index(drop=True)
    batches = np.array_split(shuffled_df, n_batches)

    legend_list = []
    pos_df = pd.DataFrame({"Pos" : range(1, gene_len+1)})

    for batch_idx, batch in enumerate(batches):
        for cnsq in ['nonsense', 'missense', 'synonymous']:
            if indel == False and cnsq == "indel":
                continue
            count_cnsq_df = batch[batch["Consequence"] == cnsq].reset_index(drop=True)
            count_cnsq_df = pos_df.merge(count_cnsq_df, on="Pos", how="left")
            axes[ax].vlines(count_cnsq_df["Pos"], ymin=0, ymax=count_cnsq_df["Count"], lw=1, zorder=1, alpha=0.5, color=colors_dict["hv_lines_needle"])
            axes[ax].scatter(count_cnsq_df["Pos"], count_cnsq_df["Count"], s=20, color='white', zorder=3, lw=0.1, ec="none") # To cover the overlapping needle top part
            if cnsq not in legend_list:
                axes[ax].scatter(count_cnsq_df["Pos"].values, count_cnsq_df["Count"].values, zorder=4,
                                    alpha=alpha, lw=0.1, ec="none", s=20, label= "Truncating" if cnsq == 'nonsense' else cnsq.capitalize(), color=colors_dict[cnsq])
                legend_list.append(cnsq)
            else:
                axes[ax].scatter(count_cnsq_df["Pos"].values, count_cnsq_df["Count"].values, zorder=4,
                                    alpha=alpha, lw=0.1, ec="none", s=20, color=colors_dict[cnsq])

    axes[ax].spines['right'].set_visible(False)
    axes[ax].spines['top'].set_visible(False)
    axes[ax].set_ylabel("Mutation count")
    axes[ax].set_xlabel("Protein position")

    # Add right Y axis with proportion labels
    total_mutations = count_df["Count"].sum()
    if total_mutations > 0:
        ax_right = axes[ax].twinx()
        left_ticks = axes[ax].get_yticks()
        right_ticks = left_ticks / total_mutations
        ax_right.set_ylim(axes[ax].get_ylim())
        ax_right.set_yticks(left_ticks)
        ax_right.set_yticklabels([f"{rt:.2f}" for rt in right_ticks])
        ax_right.set_ylabel("Proportion of mutations")
    else:
        ax_right = axes[ax].twinx()
        ax_right.set_ylim(axes[ax].get_ylim())
        ax_right.set_yticks(axes[ax].get_yticks())
        ax_right.set_yticklabels(["0.00"] * len(axes[ax].get_yticks()))
        ax_right.set_ylabel("Proportion of mutations")

def plot_stacked_bar_track_binned(count_df,
                                    gene_len,
                                    axes,
                                    colors_dict,
                                    ax=0,
                                    alpha=1,
                                    indel=False,
                                    min_bin_size=3,
                                    num_bins = 100,
                                    num_ticks=5):
    """
    Plots stacked barplot of mutation counts binned by position.

    Parameters:
        count_df: DataFrame with ['Pos', 'Consequence', 'Count'] columns
        gene_len: Length of the protein sequence
        axes: matplotlib axes array
        colors_dict: dictionary mapping consequence -> color
        ax: index of subplot
        alpha: transparency
        indel: whether to include 'indel' consequence
        bin_size: size of non-overlapping bins
        tick_every: show x-axis ticks every N bins
    """

    # Compute bin_size or result to default
    candidate_bin_size = max(1, gene_len // num_bins)
    bin_size = max(min_bin_size, candidate_bin_size)

    valid_consequences = ['nonsense', 'missense', 'synonymous']
    if indel:
        valid_consequences.append('indel')

    filtered_df = count_df[count_df["Consequence"].isin(valid_consequences)].copy()

    # Assign bin start
    filtered_df["Bin"] = ((filtered_df["Pos"] - 1) // bin_size) * bin_size + 1

    # Group and pivot
    binned_df = (
        filtered_df
        .groupby(["Bin", "Consequence"])["Count"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=valid_consequences, fill_value=0)
    )

    # Ensure all bins are represented
    all_bins = list(range(1, gene_len + 1, bin_size))
    binned_df = binned_df.reindex(all_bins, fill_value=0)

    # Plot stacked bars
    bottom = np.zeros(len(binned_df))
    for cnsq in valid_consequences:
        axes[ax].bar(
            binned_df.index,
            binned_df[cnsq],
            bottom=bottom,
            width=bin_size * 0.8,
            align="edge",
            color=colors_dict.get(cnsq, 'gray'),
            alpha=alpha,
            label="Truncating" if cnsq == 'nonsense' else cnsq.capitalize(),
            linewidth=0
        )
        bottom += binned_df[cnsq].values


    # Clean up plot
    axes[ax].spines['right'].set_visible(False)
    axes[ax].spines['top'].set_visible(False)
    axes[ax].set_ylabel(f"Mutation count\n({bin_size} AA bin)")
    axes[ax].set_xlabel("Protein position")

    # Add right Y axis with proportion labels
    total_mutations = count_df["Count"].sum()
    if total_mutations > 0:
        ax_right = axes[ax].twinx()
        left_ticks = axes[ax].get_yticks()
        right_ticks = left_ticks / total_mutations
        ax_right.set_ylim(axes[ax].get_ylim())
        ax_right.set_yticks(left_ticks)
        ax_right.set_yticklabels([f"{rt:.2f}" for rt in right_ticks])
        ax_right.set_ylabel("Proportion of mutations")
    else:
        ax_right = axes[ax].twinx()
        ax_right.set_ylim(axes[ax].get_ylim())
        ax_right.set_yticks(axes[ax].get_yticks())
        ax_right.set_yticklabels(["0.00"] * len(axes[ax].get_yticks()))
        ax_right.set_ylabel("Proportion of mutations")

    # Sparse x-ticks
    tick_every = len(all_bins) // num_ticks
    sparse_ticks = all_bins[::tick_every]
    sparse_ticks = [x-1 for x in sparse_ticks]

    axes[ax].set_xticks(sparse_ticks)
    axes[ax].set_xticklabels(sparse_ticks)
    axes[ax].set_xlim(0, gene_len + bin_size)


def manager(sample_name, mutations_file, o3d_seq_file, outdir):

    counts_per_position = get_counts_per_position_n_consequence(mutations_file)

    gene_order = sorted(pd.unique(counts_per_position["Gene"]))


    # Loop over each gene to plot
    for gene in gene_order:
        print(gene)
        try:
            mut_count_df = counts_per_position[(counts_per_position["Gene"] == gene)]
            mut_count_df = mut_count_df.groupby(by=["Pos", "Consequence"])["Count"].sum().reset_index()

            fig, ax = plt.subplots(1, 1, figsize=(5, 1.2))
            plot_count_track(
                mut_count_df,
                gene_len=mut_count_df["Pos"].max(),  # FIXME: this is not ideal, the max position is the biggest position with mutation

                axes=[ax], ax=0,
                colors_dict=metrics_colors_dictionary, indel=False, alpha=0.7
            )
            ax.set_title(f"{gene}")
            plt.savefig(f"{outdir}/{gene}.needle.pdf", bbox_inches='tight', dpi=100)
            plt.show()
            plt.close()

        except Exception as exe:
            print(gene)
            print(exe)

        # stacked version
        try:
            mut_count_df = counts_per_position[(counts_per_position["Gene"] == gene)]
            mut_count_df = mut_count_df.groupby(by=["Pos", "Consequence"])["Count"].sum().reset_index()

            fig, ax = plt.subplots(1, 1, figsize=(5, 1.2))
            plot_stacked_bar_track_binned(
                count_df=mut_count_df,
                gene_len=mut_count_df["Pos"].max(),  # FIXME: this is not ideal, the max position is the biggest position with mutation
                axes=[ax], ax=0,
                colors_dict=metrics_colors_dictionary,
                alpha=1,
                indel=False
            )
            ax.set_title(f"{gene}")
            plt.savefig(f"{outdir}/{gene}.stacked.pdf", bbox_inches='tight', dpi=100)
            plt.show()
            plt.close()

        except Exception as exe:
            print(gene)
            print(exe)





def build_counts_from_df_complete(genee, snvs_maf, omega_truncating, omega_missense):

    trunc_omega = float(omega_truncating[omega_truncating["GENE"] == genee]["omega_trunc"].values[0])
    trunc_pvalue = float(omega_truncating[omega_truncating["GENE"] == genee]["pvalue"].values[0])

    miss_omega = float(omega_missense[omega_missense["GENE"] == genee]["omega_mis"].values[0])
    miss_pvalue = float(omega_missense[omega_missense["GENE"] == genee]["pvalue"].values[0])
    snvs_gene = snvs_maf[snvs_maf["canonical_SYMBOL"] == genee].reset_index(drop = True)


    # Calculate counts based on canonical consequences
    truncating_count = float(omega_truncating[omega_truncating["GENE"] == genee]["mutations_trunc"].values[0])
    missense_count = float(omega_missense[omega_missense["GENE"] == genee]["mutations_mis"].values[0])
    synonymous_count = snvs_gene[snvs_gene["canonical_Consequence_broader"].isin(["synonymous"])].shape[0]

    # Compute
    expected_missense = (1 - ((miss_omega - 1) / miss_omega)) * missense_count
    expected_truncating = (1 - ((trunc_omega - 1) / trunc_omega)) * truncating_count


    # Create a dataframe from the counts and expected values
    data = {
        'type': ['truncating', 'synonymous', 'missense'],
        'number_obs': [truncating_count, synonymous_count, missense_count],
        'expected': [expected_truncating, None, expected_missense],
        'omega': [trunc_omega, None, miss_omega],
        'pvalue': [trunc_pvalue, None, miss_pvalue]
    }
    df = pd.DataFrame(data)
    print(df)

    # Print the final dataframe
    return df






def plot_omega_vertical(df,
                        ymax = None,
                        bar_width=0.8,
                        figsize=(1.4, 1.17),
                        between_text = 1.5,
                        withinbartext_off = 1.8,
                        text_off = 0.5,
                        min_pvalue = 1e-6,
                        gene = None,
                        legenddd = True
                        ):
    consequence_order = ['truncating', 'missense', 'synonymous',]

    # Define colors
    colors = {
        'truncating': metrics_colors_dictionary["omega_trunc"],
        'missense': metrics_colors_dictionary["omega_miss"],
        'synonymous': metrics_colors_dictionary["omega_synon"]
    }

    # Filter relevant data
    df = df[df['type'].isin(consequence_order)]

    t_obs = df[df['type'] == 'truncating']['number_obs'].item()
    t_omega = df[df['type'] == 'truncating']['omega'].item()
    t_pvalue = df[df['type'] == 'truncating']['pvalue'].item()

    m_obs = df[df['type'] == 'missense']['number_obs'].item()
    m_omega = df[df['type'] == 'missense']['omega'].item()
    m_pvalue = df[df['type'] == 'missense']['pvalue'].item()

    s_obs = df[df['type'] == 'synonymous']['number_obs'].item()  # Added synonymous mutations

    # Compute x positions for bars
    spacing_factor = bar_width * 1.1  # Adjust spacing based on bar width
    x_positions = np.arange(len(consequence_order)) * spacing_factor

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # **Matplotlib Barplot**
    ax.bar(x_positions,
            [t_obs, m_obs, s_obs],
            color=[colors[x] for x in consequence_order],
            width=bar_width,
            edgecolor='none')

    # Overlay expected values as hatched bars (only for truncating & missense)
    for i, row in df.iterrows():
        if row['type'] != 'synonymous':  # No hatch for synonymous
            if legenddd:
                ax.bar(x_positions[consequence_order.index(row['type'])], row['expected'],
                        color='none', edgecolor="grey", hatch= '//////',
                        linewidth=0,
                        width=bar_width,
                        label = 'expected'
                        )
                legenddd = False
            else:
                ax.bar(x_positions[consequence_order.index(row['type'])], row['expected'],
                        color='none', edgecolor="grey", hatch='//////',
                        linewidth=0,
                        width=bar_width
                        )

    # Remove top/right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Customize ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels([])
    # ax.set_yticklabels(ax.get_yticklabels())

    plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))


    # ax.spines['left'].set_visible(False)
    ax.set_ylabel('Number of mutations')

    # Positioning text annotations
    between_text_offset = max(max(df['number_obs'].max(), df['expected'].max()) * 0.1, between_text)
    text_offset = max(max(df['number_obs'].max(), df['expected'].max()) * 0.02, text_off)
    within_bar_text_offset = max(max(df['number_obs'].max(), df['expected'].max()) * 0.1, withinbartext_off)

    for i, row in df.iterrows():
        x_pos = x_positions[consequence_order.index(row['type'])]
        y_pos = max(row['number_obs'], row['expected']) + text_offset
        y_pos_low = max(row['number_obs'], row['expected']) - within_bar_text_offset
        omega_value = t_omega if row['type'] == 'truncating' else (m_omega if row['type'] == 'missense' else None)
        p_value = t_pvalue if row['type'] == 'truncating' else (m_pvalue if row['type'] == 'missense' else None)

        # Omega annotation (above the bar) - Only for truncating/missense
        if omega_value is not None:
            excess_mutss = row["number_obs"]*((omega_value-1)/omega_value)
            ax.text(x_pos, y_pos + between_text_offset,
                    rf'dN/dS={omega_value:.2f}',
                    fontsize=plots_general_config["annots_fontsize"], ha='center', va='bottom',
                    color='black'
                    )

            # P-value annotation (below omega)
            ax.text(x_pos, y_pos,
                    f'$p$<{min_pvalue:.1e}' if p_value < min_pvalue else (f'$p$={p_value:.1e}' if p_value < 0.01 else f'$p$={p_value:.2f}'),
                    fontsize=plots_general_config["annots_fontsize"], ha='center', va='bottom',
                    color='black'
                    )

            # Add excess mutations in bar
            if excess_mutss >= 1:
                ax.text(x_pos, y_pos_low,
                        f'{excess_mutss:,.0f}',
                        fontsize=plots_general_config["annots_fontsize"], ha='center', va='bottom', color= 'black')

        else:
            mutations = row['number_obs']
            ax.text(x_pos,
                    y_pos,
                    rf'{mutations:.0f}',
                    fontsize=plots_general_config["annots_fontsize"], ha='center', va='bottom', color='gray')


    plt.legend(frameon=False, bbox_to_anchor = (1,1))

    if ymax is not None:
        plt.ylim(0,ymax)

    if gene is not None:
        plt.title(gene, pad = 12)

    return fig

