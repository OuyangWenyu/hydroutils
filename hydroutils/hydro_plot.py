"""
Author: Wenyu Ouyang
Date: 2022-12-02 10:59:30
LastEditTime: 2025-08-02 11:58:30
LastEditors: Wenyu Ouyang
Description: Some common plots for hydrology
FilePath: \\hydroutils\\hydroutils\\hydro_plot.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import os
from typing import Dict, Union
import warnings
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from matplotlib import gridspec, pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.lines as mlines
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature

from hydroutils.hydro_stat import ecdf


def setup_matplotlib_chinese():
    """
    Configure matplotlib for Chinese font support.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        # --- 新增：为LaTeX数学公式渲染配置字体 ---
        # 这可以帮助确保数学符号和中文字体看起来更和谐
        plt.rcParams["mathtext.fontset"] = (
            "stix"  # 'stix' 是一种与Times New Roman相似的科学字体
        )
        plt.rcParams["font.family"] = "sans-serif"  # 保持其他文本为无衬线字体
        return True
    except Exception as e:
        warnings.warn(
            f"Warning: Chinese font 'SimHei' not found, Chinese text may not display correctly. Error: {e}"
        )
        return False


def plot_scatter_with_11line(
    x: np.array,
    y: np.array,
    point_color="blue",
    line_color="black",
    xlim=[0.0, 1.0],
    ylim=[0.0, 1.0],
    xlabel=None,
    ylabel=None,
):
    """plot a scatter plot for two varaibles with a 1:1 line
    Parameters
    ----------
    x : np.array
        the first variable to be plotted
    y : np.array
        the second variable to be plotted
    point_color: str
        the color of scatter points, by default "blue"
    line_color: str
        the color of 1:1 line, by default "black"
    xlim: list
        points' x_range shown in the plot
    ylim: list
        points' y_range shown in the plot
    Returns
    -------
    tuple[fig, ax]
        the figure and the ax
    """
    fig, ax = plt.subplots()
    # set background color for ax
    ax.set_facecolor("whitesmoke")
    # plot the grid of the figure
    # plt.grid(color="whitesmoke")
    ax.scatter(x, y, c=point_color, s=10)
    line = mlines.Line2D([0, 1], [0, 1], color=line_color, linestyle="--")
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.xticks(np.arange(xlim[0], xlim[1], 0.1), fontsize=16)
    plt.yticks(np.arange(ylim[0], ylim[1], 0.1), fontsize=16)
    # set xlable and ylabel
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=16)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=16)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    return fig, ax


def plot_heat_map(
    data,
    mask=None,
    fig_size=None,
    fmt="d",
    square=True,
    annot=True,
    xticklabels=True,
    yticklabels=True,
):
    """Plot a heat map for data
    https://zhuanlan.zhihu.com/p/96040773?from_voters_page=true
    Parameters
    ----------
    data : pd.DataFrame
        2-d array
    mask: np.array
        a boolean array, if True, data in the position will not be shown
    fig_size: tuple
        the size of this figure
    fmt: str, optional
        String formatting code to use when adding annotations.
    annot: boolean
        Annotate each cell with the numeric value using integer formatting
    """
    if fig_size is not None:
        fig = plt.figure(figsize=fig_size)
    ax = sns.heatmap(
        data=data,
        square=square,
        annot=annot,
        fmt=fmt,
        cmap="RdBu_r",
        mask=mask,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
    )


def plot_boxes_matplotlib(
    data: list,
    label1: list = None,
    label2: list = None,
    leg_col: int = None,
    colorlst="rbgcmywrbgcmyw",
    title=None,
    figsize=(8, 6),
    sharey=False,
    xticklabel=None,
    axin=None,
    ylim=None,
    ylabel=None,
    notch=False,
    widths=0.5,
    subplots_adjust_wspace=0.2,
    show_median=True,
    median_line_color="black",
    median_font_size="small",
):
    """Plot multiplt boxes for multiple indicators
    Parameters
    ----------
    data : list
        one element for one indictor, which could have multiple numpy array and each of them will be showed in a box
    label1 : list, optional
        name of each subplot, by default None
    label2 : list, optional
        legends' names, i.e. name of each box in one subplot (same in all subplots), by default None
    leg_col: int, optional
        number of cols for legend
    colorlst : str, optional
        _description_, by default "rbkgcmywrbkgcmyw"
    title : _type_, optional
        _description_, by default None
    figsize : tuple, optional
        _description_, by default (10, 8)
    sharey : bool, optional
        If true, all subplots share same y axis, by default False
    xticklabel : _type_, optional
        _description_, by default None
    axin : _type_, optional
        _description_, by default None
    ylim : _type_, optional
        _description_, by default None
    ylabel : _type_, optional
        _description_, by default None
    notch: boolean, optional
        if True, the median of a box will be a notch, by default False
    widths : float, optional
        _description_, by default 0.5
    subplots_adjust_wspace: float
        specifies the size of width for white space between subplots (called padding), as a fraction of the average Axes width. Default size is 0.2
    show_median: boolean
        if True, we show the median values, by default True
    median_line_color: str
        color of median lines, by default "black"
    median_font_size: str
        size of median font
    Returns
    -------
    _type_
        _description_
    """
    nc = len(data)
    if axin is None:
        fig, axes = plt.subplots(
            ncols=nc, sharey=sharey, figsize=figsize, constrained_layout=False
        )
    else:
        axes = axin

    # the next few lines are for showing median values
    decimal_places = "2"
    for k in range(nc):
        ax = axes[k] if nc > 1 else axes
        temp = data[k]
        if type(temp) is list:
            for kk in range(len(temp)):
                tt = temp[kk]
                if tt is not None and len(tt) > 0:
                    tt = tt[~np.isnan(tt)]
                    temp[kk] = tt
                else:
                    temp[kk] = []
        else:
            temp = temp[~np.isnan(temp)]
        bp = ax.boxplot(
            temp, patch_artist=True, notch=notch, showfliers=False, widths=widths
        )
        for median in bp["medians"]:
            median.set_color(median_line_color)
        medians_value = [np.median(tmp) for tmp in temp]
        percent25value = [np.percentile(tmp, 25) for tmp in temp]
        percent75value = [np.percentile(tmp, 75) for tmp in temp]
        per25min = np.min(percent25value)
        per75max = np.max(percent75value)
        median_labels = [format(s, f".{decimal_places}f") for s in medians_value]
        pos = range(len(medians_value))
        if show_median:
            for tick, label in zip(pos, ax.get_xticklabels()):
                # params of ax.text could be seen here: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html
                ax.text(
                    pos[tick] + 1,
                    medians_value[tick] + (per75max - per25min) * 0.01,
                    median_labels[tick],
                    horizontalalignment="center",
                    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html
                    size=median_font_size,
                    weight="semibold",
                    color=median_line_color,
                )
        for kk in range(len(bp["boxes"])):
            plt.setp(bp["boxes"][kk], facecolor=colorlst[kk])

        if label1 is not None:
            ax.set_xlabel(label1[k])
        else:
            ax.set_xlabel(str(k))
        if xticklabel is None:
            ax.set_xticks([])
        else:
            ax.set_xticks([y + 1 for y in range(0, len(data[k]), 2)])
            ax.set_xticklabels(xticklabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel[k])
        if ylim is not None:
            ax.set_ylim(ylim[k])
    if label2 is not None:
        plt.legend(
            bp["boxes"],
            label2,
            # explanation for bbox_to_anchor: https://zhuanlan.zhihu.com/p/101059179
            bbox_to_anchor=(1.0, 1.02, 0.25, 0.05),
            loc="upper right",
            borderaxespad=0,
            ncol=len(label2) if leg_col is None else leg_col,
            frameon=False,
            fontsize=12,
        )
    if title is not None:
        # fig.suptitle(title)
        ax.set_title(title)
    plt.tight_layout()
    plt.subplots_adjust(wspace=subplots_adjust_wspace)
    return fig if axin is None else (ax, bp)


def swarmplot_without_legend(x, y, hue, vmin, vmax, cmap, **kwargs):
    fig = plt.gcf()
    ax = sns.swarmplot(x, y, hue, **kwargs)
    # remove the legend, because we want to set a colorbar instead
    ax.legend().remove()
    norm = plt.Normalize(vmin, vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    return fig


def plot_scatter_xyc(
    x_label,
    x,
    y_label,
    y,
    c_label=None,
    c=None,
    size=20,
    is_reg=False,
    xlim=None,
    ylim=None,
    quadrant=None,
):
    """
    scatter plot: x-y relationship with c as colorbar
    Parameters
    ----------
    x_label : _type_
        _description_
    x : _type_
        _description_
    y_label : _type_
        _description_
    y : _type_
        _description_
    c_label : _type_, optional
        _description_, by default None
    c : _type_, optional
        _description_, by default None
    size : int, optional
        size of points, by default 20
    is_reg : bool, optional
        _description_, by default False
    xlim : _type_, optional
        _description_, by default None
    ylim : _type_, optional
        _description_, by default None
    quadrant: list, optional
        if it is not None, it should be a list like [0.0,0.0],
        the first means we put a new axis in x=0.0, second for y=0.0,
        so that we can build a 4-quadrant plot
    """
    fig, ax = plt.subplots()
    if type(x) is list:
        for i in range(len(x)):
            ax.plot(
                x[i], y[i], marker="o", linestyle="", ms=size, label=c_label[i], c=c[i]
            )
        ax.legend()

    elif c is None:
        df = pd.DataFrame({x_label: x, y_label: y})
        points = plt.scatter(df[x_label], df[y_label], s=size)
        if quadrant is not None:
            plt.axvline(quadrant[0], c="grey", lw=1, linestyle="--")
            plt.axhline(quadrant[1], c="grey", lw=1, linestyle="--")
            q2 = df[(df[x_label] < 0) & (df[y_label] > 0)].shape[0]
            q3 = df[(df[x_label] < 0) & (df[y_label] < 0)].shape[0]
            q4 = df[(df[x_label] > 0) & (df[y_label] < 0)].shape[0]
            q5 = df[(df[x_label] == 0) & (df[y_label] == 0)].shape[0]
            q1 = df[(df[x_label] > 0) & (df[y_label] > 0)].shape[0]
            q = q1 + q2 + q3 + q4 + q5
            r1 = int(round(q1 / q, 2) * 100)
            r2 = int(round(q2 / q, 2) * 100)
            r3 = int(round(q3 / q, 2) * 100)
            r4 = int(round(q4 / q, 2) * 100)
            r5 = 100 - r1 - r2 - r3 - r4
            plt.text(
                xlim[1] - (xlim[1] - xlim[0]) * 0.1,
                ylim[1] - (ylim[1] - ylim[0]) * 0.1,
                f"{r1}%",
                fontsize=16,
            )
            plt.text(
                xlim[0] + (xlim[1] - xlim[0]) * 0.1,
                ylim[1] - (ylim[1] - ylim[0]) * 0.1,
                f"{r2}%",
                fontsize=16,
            )
            plt.text(
                xlim[0] + (xlim[1] - xlim[0]) * 0.1,
                ylim[0] + (ylim[1] - ylim[0]) * 0.1,
                f"{r3}%",
                fontsize=16,
            )
            plt.text(
                xlim[1] - (xlim[1] - xlim[0]) * 0.1,
                ylim[0] + (ylim[1] - ylim[0]) * 0.1,
                f"{r4}%",
                fontsize=16,
            )
            plt.text(0.2, 0.02, f"{str(r5)}%", fontsize=16)
    else:
        df = pd.DataFrame({x_label: x, y_label: y, c_label: c})
        points = plt.scatter(
            df[x_label], df[y_label], c=df[c_label], s=size, cmap="Spectral"
        )  # set style options
        # add a color bar
        plt.colorbar(points)

    # set limits
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # build the regression plot
    if is_reg:
        plot = sns.regplot(x_label, y_label, data=df, scatter=False)  # , color=".1"
        plot = plot.set(xlabel=x_label, ylabel=y_label)  # add labels
    else:
        plt.xlabel(x_label, fontsize=18)
        plt.ylabel(y_label, fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)


def plot_quiver(
    exps_q_ssm_result_show,
    exps_ssm_q_result_show,
    q_diff_show,
    ssm_diff_show,
    x_label,
    y_label,
):
    fig, ax = plt.subplots()
    color = np.sqrt(q_diff_show**2 + ssm_diff_show**2)
    # normalize to get same length arrows
    r = np.power(np.add(np.power(q_diff_show, 2), np.power(ssm_diff_show, 2)), 0.5)
    plt.quiver(
        exps_q_ssm_result_show,
        exps_ssm_q_result_show,
        q_diff_show / r,
        ssm_diff_show / r,
        color,
        scale=25,
        width=0.005,
    )
    # Defining color
    plt.xlim(-0.01, 1)
    plt.ylim(-0.01, 1)
    plt.colorbar()
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


# fig, ax = plt.subplots()
# plt.xlim(-0.01, 1)
# plt.ylim(-0.01, 1)
# plt.quiver(
#     exps_q_ssm_result_show,
#     exps_ssm_q_result_show,
#     q_diff_show / r,
#     ssm_diff_show / r,
#     color,
#     scale=25,
#     width=0.005,
# )


def swarmplot_with_cbar(cmap_str, cbar_label, ylim, *args, **kwargs):
    fig = plt.gcf()
    ax = sns.swarmplot(*args, **kwargs)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    # remove the legend, because we want to set a colorbar instead
    ax.legend().remove()
    # create colorbar
    cmap = plt.get_cmap(cmap_str)
    for key, df in kwargs.items():
        # if any criteria is not matched, we can filter this site
        if key == "hue":
            hue_name = kwargs[key]
        elif key == "palette":
            color_str = kwargs[key]
    assert color_str == cmap_str
    norm = plt.Normalize(df[hue_name].min(), df[hue_name].max())
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(cbar_label, labelpad=10)
    return fig


def plot_boxs(
    data: pd.DataFrame,
    x_name: str,
    y_name: str,
    uniform_color=None,
    swarm_plot=False,
    hue=None,
    colormap=False,
    xlim=None,
    ylim=None,
    order=None,
    font="serif",
    rotation=45,
    show_median=False,
):
    """plot multiple boxes in one ax with seaborn
    Parameters
    ----------
    data : pd.DataFrame
        a tidy pandas dataframe;
        if you don't know what is "tidy data", please read: https://github.com/jizhang/pandas-tidy-data
    x_name : str
        the names of each box
    y_name : str
        what is shown
    uniform_color : str, optional
        unified color for all boxes, by default None
    swarm_plot : bool, optional
        _description_, by default False
    hue : _type_, optional
        _description_, by default None
    colormap : bool, optional
        _description_, by default False
    xlim : _type_, optional
        _description_, by default None
    ylim : _type_, optional
        _description_, by default None
    order : _type_, optional
        _description_, by default None
    font : str, optional
        _description_, by default "serif"
    rotation : int, optional
        rotation for labels in x-axis, by default 45
    show_median: bool, optional
        if True, show median value for each box, by default False
    Returns
    -------
    _type_
        _description_
    """
    fig = plt.figure()
    sns.set(style="ticks", palette="pastel", font=font, font_scale=1.5)
    # Draw a nested boxplot to show bills by day and time
    if uniform_color is not None:
        sns_box = sns.boxplot(
            x=x_name,
            y=y_name,
            data=data,
            color=uniform_color,
            showfliers=False,
            order=order,
        )
    else:
        sns_box = sns.boxplot(
            x=x_name, y=y_name, data=data, showfliers=False, order=order
        )
    if swarm_plot:
        if hue is None:
            sns_box = sns.swarmplot(
                x=x_name, y=y_name, data=data, color=".2", order=order
            )
        elif colormap:
            # Create a matplotlib colormap from the sns seagreen color palette
            cmap = sns.light_palette("seagreen", reverse=False, as_cmap=True)
            # Normalize to the range of possible values from df["c"]
            norm = matplotlib.colors.Normalize(
                vmin=data[hue].min(), vmax=data[hue].max()
            )
            colors = {cval: cmap(norm(cval)) for cval in data[hue]}
            # plot the swarmplot with the colors dictionary as palette, s=2 means size is 2
            sns_box = sns.swarmplot(
                x=x_name,
                y=y_name,
                hue=hue,
                s=2,
                data=data,
                palette=colors,
                order=order,
            )
            # remove the legend, because we want to set a colorbar instead
            plt.gca().legend_.remove()
            # create colorbar
            divider = make_axes_locatable(plt.gca())
            ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            fig = sns_box.get_figure()
            fig.add_axes(ax_cb)
            cb1 = matplotlib.colorbar.ColorbarBase(
                ax_cb, cmap=cmap, norm=norm, orientation="vertical"
            )
            cb1.set_label("Some Units")
        else:
            palette = sns.light_palette("seagreen", reverse=False, n_colors=10)
            sns_box = sns.swarmplot(
                x=x_name,
                y=y_name,
                hue=hue,
                s=2,
                data=data,
                palette=palette,
                order=order,
            )
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if show_median:
        medians = data.groupby([x_name], sort=False)[y_name].median().values
        create_median_labels(sns_box, medians_value=medians, size="x-small")
    sns.despine()
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=rotation)
    # plt.show()
    return sns_box.get_figure()


def create_median_labels(
    ax, medians_value, percent25value=None, percent75value=None, size="small"
):
    """ "create median labels for boxes in a boxplot
    Parameters
    ----------
    ax : plt.AxesSubplot
        an ax in a fig
    medians_value : np.array
        _description_
    percent25value : _type_, optional
        _description_, by default None
    percent75value : _type_, optional
        _description_, by default None
    size : str, optional
        the size of median-value labels, by default small
    """
    decimal_places = "2"
    if percent25value is None or percent75value is None:
        vertical_offset = np.min(medians_value * 0.01)  # offset from median for display
    else:
        per25min = np.min(percent25value)
        per75max = np.max(percent75value)
        vertical_offset = (per75max - per25min) * 0.01
    median_labels = [format(s, f".{decimal_places}f") for s in medians_value]
    pos = range(len(medians_value))
    for xtick in ax.get_xticks():
        ax.text(
            pos[xtick],
            medians_value[xtick] + vertical_offset,
            median_labels[xtick],
            horizontalalignment="center",
            color="w",
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html
            size=size,
            weight="semibold",
        )


def plot_diff_boxes(
    data,
    row_and_col=None,
    y_col=None,
    x_col=None,
    hspace=0.3,
    wspace=1,
    title_str=None,
    title_font_size=14,
):
    """plot boxplots in rows and cols"""
    # matplotlib.use('TkAgg')
    if type(data) is not pd.DataFrame:
        data = pd.DataFrame(data)
    subplot_num = data.shape[1] if y_col is None else len(y_col)
    if row_and_col is None:
        row_num = 1
        col_num = subplot_num
        f, axes = plt.subplots(row_num, col_num)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
    else:
        assert subplot_num <= row_and_col[0] * row_and_col[1]
        row_num = row_and_col[0]
        col_num = row_and_col[1]
        f, axes = plt.subplots(row_num, col_num)
        f.tight_layout()
    for i in range(subplot_num):
        if y_col is None:
            if row_num == 1 or col_num == 1:
                sns.boxplot(
                    y=data.columns.values[i],
                    data=data,
                    width=0.5,
                    orient="v",
                    ax=axes[i],
                    showfliers=False,
                ).set(xlabel=data.columns.values[i], ylabel="")
            else:
                row_idx = int(i / col_num)
                col_idx = i % col_num
                sns.boxplot(
                    y=data.columns.values[i],
                    data=data,
                    orient="v",
                    ax=axes[row_idx, col_idx],
                    showfliers=False,
                )
        else:
            assert x_col is not None
            if row_num == 1 or col_num == 1:
                sns.boxplot(
                    x=data.columns.values[x_col],
                    y=data.columns.values[y_col[i]],
                    data=data,
                    orient="v",
                    ax=axes[i],
                    showfliers=False,
                )
            else:
                row_idx = int(i / col_num)
                col_idx = i % col_num
                sns.boxplot(
                    x=data.columns.values[x_col],
                    y=data.columns.values[y_col[i]],
                    data=data,
                    orient="v",
                    ax=axes[row_idx, col_idx],
                    showfliers=False,
                )
    if title_str is not None:
        f.suptitle(title_str, fontsize=title_font_size)
    return f


def plot_ts(
    t: Union[list, np.array],
    y: Union[list, np.array],
    ax=None,
    t_bar=None,
    title=None,
    xlabel: str = None,
    ylabel: str = None,
    fig_size=(12, 4),
    c_lst="rbkgcmyrbkgcmyrbkgcmy",
    leg_lst=None,
    marker_lst=None,
    linewidth=2,
    linespec=None,
    dash_lines=None,
    alpha=1,
):
    """Plot time series for multi arrays with matplotlib

    Parameters
    ----------
    t : Union[list, np.array]
        time series but not just date; it can also be numbers like 1, 2, 3, ...
    y : Union[list, np.array]
        shown data series; the len of y should be equal to t's
    ax : _type_, optional
        _description_, by default None
    t_bar : _type_, optional
        _description_, by default None
    title : _type_, optional
        _description_, by default None
    xlabel: str, optional
        the name of x axis, by default None
    ylabel : str, optional
        the name of y axis, by default None
    fig_size : tuple, optional
        _description_, by default (12, 4)
    c_lst : str, optional
        _description_, by default "rbkgcmy"
    leg_lst : _type_, optional
        _description_, by default None
    marker_lst : _type_, optional
        _description_, by default None
    linewidth : int, optional
        _description_, by default 2
    linespec : _type_, optional
        _description_, by default None
    dash_lines : _type_, optional
        if dash_line, then we will plot dashed line, by default None

    Returns
    -------
    _type_
        _description_
    """
    is_new_fig = False
    if ax is None:
        fig = plt.figure(figsize=fig_size)
        ax = fig.subplots()
        is_new_fig = True
    if dash_lines is not None:
        assert isinstance(dash_lines, list)
    else:
        dash_lines = np.full(len(t), False).tolist()
        # dash_lines[-1] = True
    if type(y) is np.ndarray:
        y = [y]
    if type(linewidth) is not list:
        linewidth = [linewidth] * len(y)
    if type(alpha) is not list:
        alpha = [alpha] * len(y)
    for k in range(len(y)):
        tt = t[k] if type(t) is list else t
        yy = y[k]
        leg_str = None
        if leg_lst is not None:
            leg_str = leg_lst[k]
        if marker_lst is None:
            (line_i,) = (
                ax.plot(tt, yy, "*", color=c_lst[k], label=leg_str, alpha=alpha[k])
                if True in np.isnan(yy)
                else ax.plot(
                    tt,
                    yy,
                    color=c_lst[k],
                    label=leg_str,
                    linewidth=linewidth[k],
                    alpha=alpha[k],
                )
            )
        elif marker_lst[k] == "-":
            if linespec is not None:
                (line_i,) = ax.plot(
                    tt,
                    yy,
                    color=c_lst[k],
                    label=leg_str,
                    linestyle=linespec[k],
                    lw=linewidth[k],
                    alpha=alpha[k],
                )
            else:
                (line_i,) = ax.plot(
                    tt,
                    yy,
                    color=c_lst[k],
                    label=leg_str,
                    lw=linewidth[k],
                    alpha=alpha[k],
                )
        else:
            (line_i,) = ax.plot(
                tt,
                yy,
                color=c_lst[k],
                label=leg_str,
                marker=marker_lst[k],
                lw=linewidth[k],
                alpha=alpha[k],
            )
        if dash_lines[k]:
            line_i.set_dashes([2, 2, 10, 2])
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=18)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=18)
    if t_bar is not None:
        ylim = ax.get_ylim()
        t_bar = [t_bar] if type(t_bar) is not list else t_bar
        for tt in t_bar:
            ax.plot([tt, tt], ylim, "-k")

    if leg_lst is not None:
        ax.legend(loc="upper right", frameon=False)
        plt.legend(prop={"size": 16})
    if title is not None:
        ax.set_title(title, loc="center", fontdict={"fontsize": 17})
    # plot the grid of the figure
    plt.grid()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    return (fig, ax) if is_new_fig else ax


def plot_ecdfs(
    xs,
    ys,
    legends=None,
    style=None,
    case_str="case",
    event_str="event",
    x_str="x",
    y_str="y",
    ax_as_subplot=None,
    interval=0.1,
):
    """Empirical cumulative distribution function"""
    assert isinstance(xs, list) and isinstance(ys, list)
    assert len(xs) == len(ys)
    if legends is not None:
        assert isinstance(legends, list)
        assert len(ys) == len(legends)
    if style is not None:
        assert isinstance(style, list)
        assert len(ys) == len(style)
    for y in ys:
        assert all(xi < yi for xi, yi in zip(y, y[1:]))
    frames = []
    for i in range(len(xs)):
        str_i = x_str + str(i) if legends is None else legends[i]
        assert all(xi < yi for xi, yi in zip(xs[i], xs[i][1:]))
        df_dict_i = {
            x_str: xs[i],
            y_str: ys[i],
            case_str: np.full([xs[i].size], str_i),
        }
        if style is not None:
            df_dict_i[event_str] = np.full([xs[i].size], style[i])
        df_i = pd.DataFrame(df_dict_i)
        frames.append(df_i)
    df = pd.concat(frames)
    sns.set_style("ticks", {"axes.grid": True})
    if style is None:
        return (
            sns.lineplot(x=x_str, y=y_str, hue=case_str, data=df, estimator=None).set(
                xlim=(0, 1),
                xticks=np.arange(0, 1, interval),
                yticks=np.arange(0, 1, interval),
            )
            if ax_as_subplot is None
            else sns.lineplot(
                ax=ax_as_subplot,
                x=x_str,
                y=y_str,
                hue=case_str,
                data=df,
                estimator=None,
            ).set(
                xlim=(0, 1),
                xticks=np.arange(0, 1, interval),
                yticks=np.arange(0, 1, interval),
            )
        )
    elif ax_as_subplot is None:
        return sns.lineplot(
            x=x_str,
            y=y_str,
            hue=case_str,
            style=event_str,
            data=df,
            estimator=None,
        ).set(
            xlim=(0, 1),
            xticks=np.arange(0, 1, interval),
            yticks=np.arange(0, 1, interval),
        )
    else:
        return sns.lineplot(
            ax=ax_as_subplot,
            x=x_str,
            y=y_str,
            hue=case_str,
            style=event_str,
            data=df,
            estimator=None,
        ).set(
            xlim=(0, 1),
            xticks=np.arange(0, 1, interval),
            yticks=np.arange(0, 1, interval),
        )


def plot_ecdf(mydataframe, mycolumn, save_file=None):
    """Empirical cumulative distribution function"""
    x, y = ecdf(mydataframe[mycolumn])
    df = pd.DataFrame({"x": x, "y": y})
    sns.set_style("ticks", {"axes.grid": True})
    sns.lineplot(x="x", y="y", data=df, estimator=None).set(
        xlim=(0, 1), xticks=np.arange(0, 1, 0.05), yticks=np.arange(0, 1, 0.05)
    )
    plt.show()
    if save_file is not None:
        plt.savefig(save_file)


def plot_ecdfs_matplot(
    xs,
    ys,
    legends=None,
    colors="rbkgcmy",
    dash_lines=None,
    x_str="x",
    y_str="y",
    x_interval=0.1,
    y_interval=0.1,
    x_lim=(0, 1),
    y_lim=(0, 1),
    show_legend=True,
    legend_font_size=16,
    fig_size=(8, 6),
):
    """Empirical cumulative distribution function with matplotlib
    Parameters
    ----------
    xs : _type_
        _description_
    ys : _type_
        _description_
    legends : _type_, optional
        _description_, by default None
    colors : str, optional
        _description_, by default "rbkgcmy"
    dash_lines : _type_, optional
        _description_, by default None
    x_str : str, optional
        _description_, by default "x"
    y_str : str, optional
        _description_, by default "y"
    x_interval : float, optional
        _description_, by default 0.1
    y_interval : float, optional
        _description_, by default 0.1
    x_lim : tuple, optional
        _description_, by default (0, 1)
    y_lim : tuple, optional
        _description_, by default (0, 1)
    show_legend : bool, optional
        _description_, by default True
    legend_font_size : int, optional
        _description_, by default 16
    fig_size : tuple, optional
        size of the figure, by default (8, 6)
    Returns
    -------
    _type_
        _description_
    """
    assert isinstance(xs, list) and isinstance(ys, list)
    assert len(xs) == len(ys)
    if legends is not None:
        assert isinstance(legends, list) and len(ys) == len(legends)
    if dash_lines is not None:
        assert isinstance(dash_lines, list)
    else:
        dash_lines = np.full(len(xs), False).tolist()
    for y in ys:
        assert all(xi < yi for xi, yi in zip(y, y[1:]))
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    for i in range(len(xs)):
        if (
            np.nanmax(np.array(xs[i])) == np.inf
            or np.nanmin(np.array(xs[i])) == -np.inf
        ):
            assert all(xi <= yi for xi, yi in zip(xs[i], xs[i][1:]))
        else:
            assert all(xi <= yi for xi, yi in zip(xs[i], xs[i][1:]))
        (line_i,) = ax.plot(xs[i], ys[i], color=colors[i], label=legends[i])
        if dash_lines[i]:
            line_i.set_dashes([2, 2, 10, 2])

    plt.xlabel(x_str, fontsize=18)
    plt.ylabel(y_str, fontsize=18)
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    # set x y number font size
    plt.xticks(np.arange(x_lim[0], x_lim[1] + x_lim[1] / 100, x_interval), fontsize=16)
    plt.yticks(np.arange(y_lim[0], y_lim[1] + y_lim[1] / 100, y_interval), fontsize=16)
    if show_legend:
        ax.legend()
        plt.legend(prop={"size": legend_font_size})
    plt.grid()
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return fig, ax


def plot_pdf_cdf(mydataframe, mycolumn):
    # settings
    f, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=320)
    axes[0].set_ylabel("fraction (PDF)")
    axes[1].set_ylabel("fraction (CDF)")

    # left plot (PDF) # REMEMBER TO CHANGE bins, xlim PROPERLY!!
    sns.distplot(
        mydataframe[mycolumn],
        kde=True,
        axlabel=mycolumn,
        hist_kws={"density": True},
        ax=axes[0],
    ).set(xlim=(0, 1))

    # right plot (CDF) # REMEMBER TO CHANGE bins, xlim PROPERLY!!
    sns.distplot(
        mydataframe[mycolumn],
        kde=False,
        axlabel=mycolumn,
        hist_kws={
            "density": True,
            "cumulative": True,
            "histtype": "step",
            "linewidth": 4,
        },
        ax=axes[1],
    ).set(xlim=(0, 1), ylim=(0, 1))
    plt.show()


def plot_ts_matplot(t, y, color="r", ax=None, title=None):
    assert isinstance(t, list)
    assert isinstance(y, list)
    if ax is None:
        fig = plt.figure()
        ax = fig.subplots()
    ax.plot(t, y[0], color=color, label="pred")
    ax.plot(t, y[1], label="obs")
    ax.legend()
    if title is not None:
        ax.set_title(title, loc="center")
    return (fig, ax) if ax is None else ax


def plot_map_carto(
    data,
    lat,
    lon,
    fig=None,
    ax=None,
    pertile_range=None,
    value_range=None,
    fig_size=(8, 8),
    need_colorbar=True,
    colorbar_size=[0.91, 0.318, 0.02, 0.354],
    cmap_str="jet",
    idx_lst=None,
    markers=None,
    marker_size=20,
    is_discrete=False,
    colors="rbkgcmywrbkgcmyw",
    category_names=None,
    legend_font_size=None,
    colorbar_font_size=None,
):
    """_summary_

    Parameters
    ----------
    data : np.array
        data shown in the map, 1-d array, one value for one point
    lat : np.array
        1-d array, latitude of each point
    lon : np.array
        1-d array, longitude of each point
    fig : _type_, optional
        _description_, by default None
    ax : _type_, optional
        _description_, by default None
    pertile_range : list, optional
        value's range shown in the map, by default None
        for example, [0, 100] means all data; [23, 75] means 25-quantile to 75-quantile values
    value_range: list, optinal
        if value_range is not None, its values are used rather than percential_range
    fig_size : tuple, optional
        _description_, by default (8, 8)
    need_colorbar : bool, optional
        _description_, by default True
    colorbar_size : list, optional
        size of colorbar, by default [0.91, 0.318, 0.02, 0.354]
    cmap_str : str, optional
        _description_, by default "jet"
    idx_lst : _type_, optional
        for scatter plot, it is better to use idx_lst to plot multiple-type points, by default None
    markers : list, optional
        the marker shown in the map, by default None
    marker_size : int, optional
        _description_, by default 20
    is_discrete : bool, optional
        if True, legend is used, else colorbar is used, by default False
    colors : str, optional
        colors for different parts, by default "rbkgcmywrbkgcmyw"
    category_names : list, optional
        shown in the legend when using discrete values, by default None
    legend_font_size : _type_, optional
        _description_, by default None
    colorbar_font_size : float, optional
        font size of colorbar, by default None

    Returns
    -------
    _type_
        _description_
    """
    if value_range is not None:
        vmin = value_range[0]
        vmax = value_range[1]
    elif pertile_range is None:
        # https://blog.csdn.net/chenirene510/article/details/111318539
        mask_data = np.ma.masked_invalid(data)
        vmin = np.min(mask_data)
        vmax = np.max(mask_data)
    else:
        assert 0 <= pertile_range[0] < pertile_range[1] <= 100
        vmin = np.nanpercentile(data, pertile_range[0])
        vmax = np.nanpercentile(data, pertile_range[1])
    llcrnrlat = (np.min(lat),)
    urcrnrlat = (np.max(lat),)
    llcrnrlon = (np.min(lon),)
    urcrnrlon = (np.max(lon),)
    extent = [llcrnrlon[0], urcrnrlon[0], llcrnrlat[0], urcrnrlat[0]]
    # Figure
    if fig is None or ax is None:
        fig, ax = plt.subplots(
            1, 1, figsize=fig_size, subplot_kw={"projection": ccrs.PlateCarree()}
        )
    ax.set_extent(extent)
    states = NaturalEarthFeature(
        category="cultural",
        scale="50m",
        facecolor="none",
        name="admin_1_states_provinces_shp",
    )
    ax.add_feature(states, linewidth=0.5, edgecolor="black")
    ax.coastlines("50m", linewidth=0.8)
    if idx_lst is not None:
        if isinstance(marker_size, list):
            assert len(marker_size) == len(idx_lst)
        else:
            marker_size = np.full(len(idx_lst), marker_size).tolist()
        if not isinstance(marker_size, list):
            markers = np.full(len(idx_lst), markers).tolist()
        else:
            assert len(markers) == len(idx_lst)
        if not isinstance(cmap_str, list):
            cmap_str = np.full(len(idx_lst), cmap_str).tolist()
        else:
            assert len(cmap_str) == len(idx_lst)
        if is_discrete:
            for i in range(len(idx_lst)):
                ax.plot(
                    lon[idx_lst[i]],
                    lat[idx_lst[i]],
                    marker=markers[i],
                    ms=marker_size[i],
                    label=category_names[i],
                    c=colors[i],
                    linestyle="",
                )
                ax.legend(prop=dict(size=legend_font_size))
        else:
            scatter = []
            for i in range(len(idx_lst)):
                scat = ax.scatter(
                    lon[idx_lst[i]],
                    lat[idx_lst[i]],
                    c=data[idx_lst[i]],
                    marker=markers[i],
                    s=marker_size[i],
                    cmap=cmap_str[i],
                    vmin=vmin,
                    vmax=vmax,
                )
                scatter.append(scat)
            if need_colorbar:
                if colorbar_size is not None:
                    cbar_ax = fig.add_axes(colorbar_size)
                    cbar = fig.colorbar(scat, cax=cbar_ax, orientation="vertical")
                else:
                    cbar = fig.colorbar(scat, ax=ax, pad=0.01)
                if colorbar_font_size is not None:
                    cbar.ax.tick_params(labelsize=colorbar_font_size)
            if category_names is not None:
                ax.legend(
                    scatter, category_names, prop=dict(size=legend_font_size), ncol=2
                )
    elif is_discrete:
        scatter = ax.scatter(lon, lat, c=data, s=marker_size)
        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(
            *scatter.legend_elements(), loc="lower left", title="Classes"
        )
        ax.add_artist(legend1)
    else:
        scat = plt.scatter(
            lon, lat, c=data, s=marker_size, cmap=cmap_str, vmin=vmin, vmax=vmax
        )
        if need_colorbar:
            if colorbar_size is not None:
                cbar_ax = fig.add_axes(colorbar_size)
                cbar = fig.colorbar(scat, cax=cbar_ax, orientation="vertical")
            else:
                cbar = fig.colorbar(scat, ax=ax, pad=0.01)
            if colorbar_font_size is not None:
                cbar.ax.tick_params(labelsize=colorbar_font_size)
    return ax


def plot_ts_map(dataMap, dataTs, lat, lon, t, sites_id, pertile_range=None):
    # show the map in a pop-up window
    matplotlib.use("TkAgg")
    assert isinstance(dataMap, list)
    assert isinstance(dataTs, list)
    # setup axes
    fig = plt.figure(figsize=(8, 8), dpi=100)
    gs = gridspec.GridSpec(2, 1)
    # plt.subplots_adjust(left=0.13, right=0.89, bottom=0.05)
    # plot maps
    ax1 = plt.subplot(gs[0], projection=ccrs.PlateCarree())
    ax1 = plot_map_carto(
        dataMap, lat=lat, lon=lon, fig=fig, ax=ax1, pertile_range=pertile_range
    )
    # line plot
    ax2 = plt.subplot(gs[1])

    # plot ts
    def onclick(event):
        print("click event")
        # refresh the ax2, then new ts data can be showed without previous one
        ax2.cla()
        xClick = event.xdata
        yClick = event.ydata
        d = np.sqrt((xClick - lon) ** 2 + (yClick - lat) ** 2)
        ind = np.argmin(d)
        titleStr = "site_id %s, lat %.3f, lon %.3f" % (
            sites_id[ind],
            lat[ind],
            lon[ind],
        )
        tsLst = dataTs[ind]
        plot_ts_matplot(t, tsLst, ax=ax2, title=titleStr)
        # following funcs both work
        fig.canvas.draw()
        # plt.draw()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()


def plot_rainfall_runoff(
    t,
    p,
    qs,
    fig_size=(8, 6),
    c_lst="rbkgcmy",
    leg_lst=None,
    dash_lines=None,
    title=None,
    xlabel=None,
    ylabel=None,
    prcp_ylabel="prcp(mm/day)",
    linewidth=1,
    prcp_interval=20,
):
    """Plot rainfall and runoff in one figure

    Parameters
    ----------
    t : a np.array or a list of some np.array and the length of the list is same as the length of qs
        time series, better to be a list with the same length of qs
    p : np.array
        precipitation, a time series
    qs : a np.array or a list of some np.array
        streamflow, a list with multiple time series
    fig_size : tuple, optional
        figure size, by default (8, 6)
    c_lst : str, optional
        colors, by default "rbkgcmy"
    leg_lst : list, optional
        legends, by default None
    dash_lines : list, optional
        if a line is dash line, by default None
    title : str, optional
        the title of the figure, by default None
    xlabel : str, optional
        label of x axis, by default None
    ylabel : str, optional
        label of y axis, by default None
    linewidth : int, optional
        the width of lines, by default 1
    prcp_interval : int, optional
        the interval of precipitation, by default 20
    """
    fig, ax = plt.subplots(figsize=fig_size)
    if dash_lines is not None:
        assert isinstance(dash_lines, list)
    else:
        dash_lines = np.full(len(qs), False).tolist()
    for k in range(len(qs)):
        tt = t[k] if type(t) is list else t
        q = qs[k]
        leg_str = None
        if leg_lst is not None:
            leg_str = leg_lst[k]
        (line_i,) = ax.plot(tt, q, color=c_lst[k], label=leg_str, linewidth=linewidth)
        if dash_lines[k]:
            line_i.set_dashes([2, 2, 10, 2])

    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.2)
    # Create second axes, in order to get the bars from the top you can multiply by -1
    ax2 = ax.twinx()
    # ax2.bar(tt, -p, color="b")
    ax2.fill_between(tt, 0, -p, step="mid", color="b", alpha=0.5)
    # ax2.plot(tt, -p, color="b", alpha=0.7, linewidth=1.5)

    # Now need to fix the axis labels
    # max_pre = max(p)
    max_pre = p.max().item()
    ax2.set_ylim(-max_pre * 5, 0)
    y2_ticks = np.arange(0, max_pre, prcp_interval)
    y2_ticklabels = [str(i) for i in y2_ticks]
    ax2.set_yticks(-1 * y2_ticks)
    ax2.set_yticklabels(y2_ticklabels, fontsize=16)
    # ax2.set_yticklabels([lab.get_text()[1:] for lab in ax2.get_yticklabels()])
    if title is not None:
        ax.set_title(title, loc="center", fontdict={"fontsize": 17})
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=18)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=18)
    ax2.set_ylabel(prcp_ylabel, fontsize=8, loc="top")
    # ax2.set_ylabel("precipitation (mm/day)", fontsize=12, loc='top')
    # https://github.com/matplotlib/matplotlib/issues/12318
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(bbox_to_anchor=(0.01, 0.85), loc="upper left", fontsize=16)
    ax.grid()
    return fig, ax


def plot_rainfall_runoff_xu(
    t,
    p,
    qs,
    fig_size=(10, 6),
    title="prcp-streamflow",
    leg_lst=None,
    ylabel="streamflow(m^3/s)",
    prcp_ylabel="prcp(mm/day)",
):
    obs, pred = qs

    fig, ax1 = plt.subplots(figsize=fig_size)

    ax1.bar(t, p, width=0.1, color="blue", alpha=0.6, label="Precipitation")
    ax1.set_ylabel(prcp_ylabel, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax1.set_ylim(0, p.max() * 5)
    ax1.invert_yaxis()

    ax2 = ax1.twinx()

    # transform the unit of obs and pred
    ax2.plot(
        t,
        obs,
        color="green",
        linestyle="-",
        label="observed value",
    )
    ax2.plot(
        t,
        pred,
        color="red",
        linestyle="--",
        label="predicted value",
    )

    ax2.set_ylabel(ylabel, color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title(title)

    plt.legend(loc="upper left")


def plot_rainfall_runoff_chai(
    t,
    ps,
    qs,
    c_lst="rbkgcmy",
    title="Observation of Precipitation and Streamflow",
    alpha_lst=None,
    p_labels=None,
    q_labels=None,
):
    if alpha_lst is None:
        alpha_lst = [0.5, 0.5]
    if p_labels is None:
        p_labels = ["era5land", "gauge"]
    if q_labels is None:
        q_labels = ["observation", "simulation"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8))
    fig.suptitle(title, fontsize=16)
    for i, p in enumerate(ps):
        ax1.bar(
            t,
            p,
            color=c_lst[i],
            label=p_labels[i],
            width=0.9,
            alpha=alpha_lst[i],
        )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Precipitation (mm/d)", color="b")
    ax1.invert_yaxis()
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.legend()

    for j, q in enumerate(qs):
        ax2.plot(t, q, color=c_lst[j], label=q_labels[j], alpha=alpha_lst[j])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Streamflow (m$^3$/s)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.set_xlim(ax1.get_xlim())
    ax2.text(
        0.05,
        0.95,
        "Streamflow",
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    ax2.legend()

    fig.tight_layout()
    return fig, ax2


def plot_event_characteristics(
    event_analysis: Dict,
    output_folder: str,
    delta_t_hours: float = 3.0,
    net_rain_key: str = "P_eff",
    obs_flow_key: str = "Q_obs_eff",
):
    """为单个洪水事件绘制特征图并保存，确保径流曲线在柱状图上层。@author: Zheng Zhang"""
    net_rain = event_analysis[net_rain_key]
    direct_runoff = event_analysis[obs_flow_key]
    event_filename = os.path.basename(event_analysis["filepath"])

    fig, ax1 = plt.subplots(figsize=(15, 7))
    fig.suptitle(f"洪水事件特征分析 - {event_filename}", fontsize=16)

    # 绘制径流曲线 (左Y轴)
    x_axis = np.arange(len(direct_runoff))
    # --- 核心修改：为曲线设置一个较高的 zorder ---
    ax1.plot(
        x_axis,
        direct_runoff,
        color="orangered",
        label=r"径流 ($m^3/s$)",
        zorder=10,
        linewidth=2,
    )  # zorder=10

    ax1.set_xlabel(f"时段 (Δt = {delta_t_hours}h)", fontsize=12)
    ax1.set_ylabel(r"径流流量 ($m^3/s$)", color="orangered", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="orangered")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, which="major", linestyle="--", linewidth="0.5", color="gray")

    # 创建共享X轴的第二个Y轴
    ax2 = ax1.twinx()
    # 绘制净雨柱状图 (右Y轴，向下)
    # --- 核心修改：为柱状图设置一个较低的 zorder (可选，但好习惯) ---
    ax2.bar(
        x_axis,
        net_rain,
        color="steelblue",
        label="净雨 (mm)",
        width=0.8,
        zorder=5,
    )  # zorder=5

    ax2.set_ylabel("净雨量 (mm)", color="steelblue", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2.invert_yaxis()
    ax2.set_ylim(top=0)

    # 准备并标注文本框 (与之前对齐版本相同)
    labels = [
        "洪峰流量:",
        "洪   量:",
        "洪水历时:",
        "总净雨量:",
        "洪峰雨峰延迟:",
    ]
    values = [
        f"{event_analysis['peak_obs']:.2f} " + r"$m^3/s$",
        f"{event_analysis['runoff_volume_m3'] / 1e6:.2f} " + r"$\times 10^6\ m^3$",
        f"{event_analysis['runoff_duration_hours']:.1f} 小时",
        f"{event_analysis['total_net_rain']:.2f} mm",
        f"{event_analysis['lag_time_hours']:.1f} 小时",
    ]
    label_text = "\n".join(labels)
    value_text = "\n".join(values)
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8)
    ax1.text(
        0.85,
        0.95,
        "--- 水文特征 ---",
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=bbox_props,
    )
    ax1.text(
        0.80,
        0.85,
        label_text,
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        family="SimSun",
    )
    ax1.text(
        0.82,
        0.85,
        value_text,
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="left",
    )

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.95),
    )

    # 保存图形
    output_filename = os.path.join(
        output_folder, f"{os.path.splitext(event_filename)[0]}.png"
    )
    try:
        plt.savefig(output_filename, dpi=150, bbox_inches="tight")
        print(f"  已生成图表: {output_filename}")
    except Exception as e:
        print(f"  保存图表失败: {output_filename}, 错误: {e}")

    plt.close(fig)


def plot_unit_hydrograph(
    U_optimized,
    title,
    smoothing_factor=None,
    peak_violation_weight=None,
    delta_t_hours=3.0,
):
    """
    绘制单位线图

    Args:
        U_optimized: 优化的单位线参数
        title: 图表标题
        smoothing_factor: 平滑因子（可选）
        peak_violation_weight: 单峰惩罚权重（可选）
    """
    if U_optimized is None:
        print(f"⚠️ 无法绘制单位线：{title} - 优化失败")
        return

    time_axis_uh = np.arange(1, len(U_optimized) + 1) * delta_t_hours

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis_uh, U_optimized, marker="o", linestyle="-")

    # 构建完整标题
    full_title = title
    if smoothing_factor is not None and peak_violation_weight is not None:
        full_title += f" (平滑={smoothing_factor}, 单峰罚={peak_violation_weight})"

    plt.title(full_title)
    plt.xlabel(f"时间 (小时, Δt={delta_t_hours}h)")
    plt.ylabel("1mm净雨单位线纵坐标 (mm/3h)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
