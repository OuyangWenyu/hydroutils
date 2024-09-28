import numpy as np
import matplotlib.pyplot as plt
from hydroutils.hydro_plot import plot_ts, plot_rainfall_runoff


def test_plot_ts():
    # Test case 1: Basic test case with one array
    t = np.array([1, 2, 3, 4, 5])
    y = np.array([10, 20, 30, 40, 50])
    fig, ax = plot_ts(t, y)

    # Test case 2: Multiple arrays with different colors and markers
    y = [np.array([10, 20, 30, 40, 50]), np.array([5, 15, 25, 35, 45])]
    leg_lst = ["Array 1", "Array 2"]
    marker_lst = ["o", "s"]
    fig, ax = plot_ts(t, y, leg_lst=leg_lst, marker_lst=marker_lst)

    # Test case 3: Custom title and axis labels
    y = [np.array([10, 20, 30, 40, 50])]
    title = "Time Series Plot"
    xlabel = "Time"
    ylabel = "Value"
    fig, ax = plot_ts(t, y, title=title, xlabel=xlabel, ylabel=ylabel)

    # Test case 4: Test with NaN values
    y = [np.array([10, np.nan, 30, 40, 50])]
    fig, ax = plot_ts(t, y)

    # Test case 5: Test with t_bar parameter
    y = [np.array([10, 20, 30, 40, 50])]
    t_bar = [2.5]
    fig, ax = plot_ts(t, y, t_bar=t_bar)

    # Test case 6: Test with legend and custom line styles
    y = [np.array([10, 20, 30, 40, 50]), np.array([5, 15, 25, 35, 45])]
    leg_lst = ["Array 1", "Array 2"]
    linespec = ["--", "-."]
    fig, ax = plot_ts(t, y, leg_lst=leg_lst, linespec=linespec)

    # Test case 7: Test with custom figure size
    y = [np.array([10, 20, 30, 40, 50])]
    fig_size = (8, 6)
    fig, ax = plot_ts(t, y, fig_size=fig_size)

    # Test case 8: Test with all parameters
    y = [np.array([10, 20, 30, 40, 50])]
    ax = plt.subplots()[1]
    t_bar = [2.5]
    title = "Time Series Plot"
    xlabel = "Time"
    ylabel = "Value"
    fig_size = (8, 6)
    c_lst = "rg"
    leg_lst = ["Array 1"]
    marker_lst = ["o"]
    linewidth = 1
    linespec = ["--"]
    plot_ts(
        t,
        y,
        ax=ax,
        t_bar=t_bar,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        fig_size=fig_size,
        c_lst=c_lst,
        leg_lst=leg_lst,
        marker_lst=marker_lst,
        linewidth=linewidth,
        linespec=linespec,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_plot_rainfall_runoff():
    # Test case 1: Basic test case with one streamflow array
    t = np.array([1, 2, 3, 4, 5])
    p = np.array([10, 20, 30, 40, 50])
    qs = [np.array([5, 15, 25, 35, 45])]
    fig, ax = plot_rainfall_runoff(t, p, qs)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Test case 2: Multiple streamflow arrays with different colors
    qs = [np.array([5, 15, 25, 35, 45]), np.array([10, 20, 30, 40, 50])]
    c_lst = "rg"
    fig, ax = plot_rainfall_runoff(t, p, qs, c_lst=c_lst)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Test case 3: Custom title and axis labels
    title = "Rainfall and Runoff Plot"
    xlabel = "Time"
    ylabel = "Streamflow"
    fig, ax = plot_rainfall_runoff(t, p, qs, title=title, xlabel=xlabel, ylabel=ylabel)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Test case 4: Test with NaN values in precipitation
    p = np.array([10, np.nan, 30, 40, 50])
    fig, ax = plot_rainfall_runoff(t, p, qs)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Test case 5: Test with dash lines
    dash_lines = [True, False]
    fig, ax = plot_rainfall_runoff(t, p, qs, dash_lines=dash_lines)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Test case 6: Test with legend
    leg_lst = ["Streamflow 1", "Streamflow 2"]
    fig, ax = plot_rainfall_runoff(t, p, qs, leg_lst=leg_lst)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Test case 7: Test with custom figure size
    fig_size = (10, 8)
    fig, ax = plot_rainfall_runoff(t, p, qs, fig_size=fig_size)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Test case 8: Test with all parameters
    p = np.array([10, 20, 30, 40, 50])
    qs = [np.array([5, 15, 25, 35, 45])]
    title = "Rainfall and Runoff Plot"
    xlabel = "Time"
    ylabel = "Streamflow"
    fig_size = (10, 8)
    c_lst = "rg"
    leg_lst = ["Streamflow 1"]
    dash_lines = [True]
    linewidth = 2
    prcp_interval = 10
    plot_rainfall_runoff(
        t,
        p,
        qs,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        fig_size=fig_size,
        c_lst=c_lst,
        leg_lst=leg_lst,
        dash_lines=dash_lines,
        linewidth=linewidth,
        prcp_interval=prcp_interval,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
