# Basic
import numpy as np
import pandas as pd


# Plotting
import matplotlib.pyplot as plt
import seaborn as sns


def create_countplots(
    data: pd.DataFrame, 
    features: list[str], 
    ax_cols: int=2, 
    hue : str=None, 
    is_ordered: bool=None
) -> None:
    """
    Creates count plots for a multiple features in a Dataset

    Each feature is plotted as a bar chart with optional hue-based grouping.
    Can sort bars either by frequency or by natural numeric/category order.
    Plots are arranged in a grid layout
    
    Args:
        data (pd.DataFrame): Full input dataset containing both features and target
        features (list[str]): List of feature names to plot
        ax_cols (int, optional): Number of plots per axes row. Default is 2
        hue (str, optional): Column name to use for color grouping
        is_ordered (bool, optional): If True, sorts categories numerically or alphabetically. If False, sorts by frequency. Default is None
    
    Returns:
        None
    """
    # defines axes and create figure
    ax_rows = (len(features) + ax_cols - 1) // ax_cols
    fig, axes = plt.subplots(ax_rows, ax_cols, figsize=(10,4*ax_rows))

    # assures axes is 2D-array
    if ax_rows == 1 and ax_cols == 1:
        axes = np.array([[axes]])
    elif ax_rows == 1:
        axes = np.array([axes])
    elif ax_cols == 1:
        axes = np.array([axes]).T

    
    
    # creates countplots
    for i, col in enumerate(features):
        r, c = i // ax_cols, i % ax_cols   # calculation of coordinates

        if is_ordered:
            if pd.api.types.is_numeric_dtype(data[col]):
                sorted_order = sorted(data[col].unique())
            elif pd.api.types.is_objects_dtype(data[col]) or isinstance(dtype(data[col]), pd.CategoricalDtype):
                try:
                    sorted_order = sorted(data[col].unique(), key=lambda x: int(x))
                except (ValueError, TypeError) as e:
                    sorted_order = list(data[col].dropna().unique())
        else:
            sorted_order = data[col].value_counts().index
            
        # transforms names of columns to look prettier on plots
        pretty_column_title = col.replace('_',' ').title()

        # main plotting part
        ax=axes[r,c]
        sns.countplot(
            data=data, 
            x=col, 
            ax=ax, 
            edgecolor='black', 
            hue=hue, 
            order=sorted_order
        )

        # percentage over bars
        total = len(data[col].dropna())
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                pct_text = f'{(height/total)*100:.1f}%'
                ax.text(
                    p.get_x() + p.get_width()/2, 
                    height, 
                    pct_text, 
                    ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=0
                )
        axes[r,c].set_xlabel(pretty_column_title, fontsize=8, fontweight='bold')
        axes[r,c].set_ylabel('Qty', fontsize=8, fontweight='bold')
        axes[r,c].set_title(pretty_column_title, fontsize=12, fontweight='bold')
        axes[r,c].grid(axis='y', linestyle='--',alpha=0.7)
    # deletes non-used axes
    for j in range(len(features), ax_rows*ax_cols):
        r, c = j // ax_cols, j % ax_cols
        fig.delaxes(axes[r,c])

    # plot display
    plt.tight_layout()
    plt.show()


def create_histplots(
    data: pd.DataFrame, 
    features: list[str], 
    ax_cols: int=2, 
    bins: int=10, 
    hue : str=None
) -> None:
    """
    Creates histograms for a multiple features in a Dataset

    Features are displayed as individual histogram plots arranged in a grid layout.
    Allows custom binning and optional grouping by a categorical variable via `hue`.
    Displays percentage annotations on bars for interpretability.
    
    Args:
        data (pd.DataFrame): Full input dataset containing both features and target
        features (list[str]): List of feature names to plot
        ax_cols (int, optional): Number of plots per axes row. Default is 2
        bins (int, optional): Number of histogram bins per feature. Default is 10
        hue (str, optional): Column name to use for color grouping
    
    Returns:
        None
    """
    
    # defines axes and create figure
    ax_rows = (len(features) + ax_cols - 1) // ax_cols
    fig, axes = plt.subplots(ax_rows, ax_cols, figsize=(10,4*ax_rows))

    # assures axes is 2D-array
    if ax_rows == 1 and ax_cols == 1:
        axes = np.array([[axes]])
    elif ax_rows == 1:
        axes = np.array([axes])
    elif ax_cols == 1:
        axes = np.array([axes]).T

    # creates histplots
    for i, col in enumerate(features):
        r, c = i // ax_cols, i % ax_cols   # calculation of coordinates
        
        # transforms names of columns to look prettier on plots
        pretty_column_title = col.replace('_',' ').title()

        # create custom bins for histogram using 'bins' parameter
        bin_edges = np.linspace(data[col].min(), data[col].max(), bins+1)
        
        # main plotting part
        ax=axes[r,c]
        sns.histplot(data=data, x=col, bins=bin_edges, ax=ax, edgecolor='black', hue=hue)

        # percentage over bars
        total = len(data[col].dropna())
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                pct_text = f'{(height/total)*100:.1f}%'
                ax.text(
                    p.get_x() + p.get_width()/2, 
                    height, 
                    pct_text, 
                    ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=0
                )
        # additional features
        axes[r,c].set_xlabel(pretty_column_title, fontsize=8, fontweight='bold')
        axes[r,c].set_ylabel('Qty', fontsize=8, fontweight='bold')
        axes[r,c].set_title(pretty_column_title, fontsize=12, fontweight='bold')
        axes[r,c].grid(axis='y', linestyle='--',alpha=0.7)
        axes[r,c].set_xticks(bin_edges)
        axes[r,c].set_xticklabels(np.round(bin_edges,2), rotation=45)

    # deletes non-used axes
    for j in range(len(features), ax_rows*ax_cols):
        r, c = j // ax_cols, j % ax_cols
        fig.delaxes(axes[r,c])

    # plot display
    plt.tight_layout()
    plt.show()


def create_boxplots(
    data: pd.DataFrame, 
    features: list[str], 
    target: pd.Series, 
    ax_cols: int=2, 
    grid_step: int=5
) -> None:
    """
    Creates histograms for a multiple features in a Dataset

    Features are displayed as individual histogram plots arranged in a grid layout.
    Allows custom binning and optional grouping by a categorical variable via `hue`.
    Displays percentage annotations on bars for interpretability.
    
    Args:
        data (pd.DataFrame): Full input dataset containing both features and target
        features (list[str]): List of feature names to plot
        target (str): Name of the numeric target column
        ax_cols (int, optional): Number of plots per axes row. Default is 2
        grid_step (int, optional): Step size for Y-axis grid. If None, uses automatic ticks. Defaults to 5
    
    Returns:
        None
    """
    
    # defines axes and create figure
    ax_rows = (len(features) + ax_cols - 1) // ax_cols
    fig, axes = plt.subplots(ax_rows, ax_cols, figsize=(10,6*ax_rows))

    # assures axes is 2D-array
    if ax_rows == 1 and ax_cols == 1:
        axes = np.array([[axes]])
    elif ax_rows == 1:
        axes = np.array([axes])
    elif ax_cols == 1:
        axes = np.array([axes]).T

    # creates boxplots
    for i, col in enumerate(features):
        r, c = i // ax_cols, i % ax_cols   # calculation of coordinates

        # transforms names of columns to look prettier on plots
        pretty_x_title = col.replace('_',' ').title()
        pretty_y_title = target.replace('_',' ').title()

        # basic plotting part
        sns.boxplot(data=data, x=col, y=target, ax=axes[r,c])
        axes[r,c].set_xlabel( pretty_x_title, fontsize=8, fontweight='bold')
        axes[r,c].set_ylabel(pretty_y_title, fontsize=8, fontweight='bold')
        axes[r,c].set_title(f"""{pretty_x_title}\ncorrelation with\n{pretty_y_title}""", 
                            fontsize=12, 
                            fontweight='bold')
        axes[r,c].grid(axis='y', linestyle='--',alpha=0.7)

        # create custom grid with use of 'grid_step' parameter
        start = np.floor(data[target].min() / 10) * 10
        end = np.ceil(data[target].max() / 10) * 10 + 5
        axes[r,c].set_yticks(np.arange(start,end,grid_step))
            
    # deletes non-used axes
    for j in range(len(features), ax_rows*ax_cols):
        r, c = j // ax_cols, j % ax_cols
        fig.delaxes(axes[r,c])

    # plot display
    plt.tight_layout()
    plt.show()