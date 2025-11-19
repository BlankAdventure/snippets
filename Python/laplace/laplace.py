# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 17:45:39 2025

@author: BlankAdventure

run as:
bokeh serve laplace.py --show

"""

import numpy as np
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, PointDrawTool, Span
from bokeh.plotting import figure
from bokeh.layouts import column, row


time_points = 100
time_dur = 10

init_r = -1
init_i = 1

point_counter = [0]

# Time axis
t = np.linspace(0, time_dur, time_points)


# Data sources
s_source = ColumnDataSource(data=dict(x=[init_r], y=[init_i], label=[None]))
real_source = ColumnDataSource(data=dict(t=t, y=np.zeros_like(t)))
imag_source = ColumnDataSource(data=dict(t=t, y=np.zeros_like(t)))
traj_source = ColumnDataSource(data=dict(x=[],y=[]))

# Function to compute e^(s*t)
def compute_exp_st(s_real, s_imag):
    s = s_real + 1j * s_imag
    y = np.exp(s * t)
    return y.real, y.imag


def update_plots(attr, old, new):
    current_data = dict(s_source.data)
    num_points = len(current_data['x'])
    num_labels = len(current_data['label'])
    
    # Add coordinate labels for new points
    if num_points > num_labels:
        for i in range(num_labels, num_points):
            x_val = current_data['x'][i]
            y_val = current_data['y'][i]
            z = complex(x_val, y_val)
            current_data['label'].append(f"{z:.2f}")
        s_source.data = current_data
        
    # Update labels when points are moved
    elif num_points == num_labels:
        updated = False
        for i in range(num_points):
            x_val = current_data['x'][i]
            y_val = current_data['y'][i]
            z = complex(x_val, y_val)
            new_label = f"{z:.2f}"
            if current_data['label'][i] != new_label:
                current_data['label'][i] = new_label
                updated = True
        if updated:
            s_source.data = current_data
    
    rr = 0; ii = 0
    for i in range(num_points):        
        x_val = current_data['x'][i]
        y_val = current_data['y'][i]
        y_real, y_imag = compute_exp_st(x_val, y_val)
        rr = rr + y_real
        ii = ii + y_imag

    real_source.data = dict(t=t, y=rr)
    imag_source.data = dict(t=t, y=ii)
    traj_source.data = dict(x=rr, y=ii)

# ****************************************************
# --- Complex plane for s ---
# ****************************************************
plane_fig = figure(
    title="Complex Plane (drag the red dot)",
    width=400, height=400,
    x_axis_label="σ (Real part)",
    y_axis_label="ω (Imag part)",
    x_range=(-3, 3), y_range=(-3, 3),
    tools="pan,wheel_zoom,reset",
    toolbar_location=None
)
plane_fig.title.align = "center"

# Add gray lines at origin
vline_plane = Span(location=0, dimension="height", line_color="gray", line_width=1)
hline_plane = Span(location=0, dimension="width", line_color="gray", line_width=1)
plane_fig.add_layout(vline_plane)
plane_fig.add_layout(hline_plane)

plane_fig.text('x', 'y', text='label', source=s_source, text_font_size='10pt', 
               text_align='center', text_baseline='bottom', y_offset=-12, x_offset=30,
               background_fill_color="white", text_color='red', border_line_color="white",)

renderer = plane_fig.circle('x', 'y', source=s_source, size=12, color='red', alpha=0.6)

draw_tool = PointDrawTool(renderers=[renderer], add=True)
plane_fig.add_tools(draw_tool)
plane_fig.toolbar.active_tap = draw_tool



# *********************************************
# --- Complex trajectory (Real vs Imag) ---
# *********************************************
traj_fig = figure(
    title="Trajectory of e^(s·t) in Complex Plane",
    width=400, height=400,
    x_axis_label="Real(e^{st})",
    y_axis_label="Imag(e^{st})",
    x_range=(-2, 2), y_range=(-2, 2),
    toolbar_location=None
)
traj_fig.title.align = "center"
traj_fig.line("x", "y", source=traj_source, line_color="green")

traj_fig.circle([0], [0], radius=1, color="navy", alpha=0.2)

# Centered axes for trajectory
vline_traj = Span(location=0, dimension="height", line_color="gray", line_width=1)
hline_traj = Span(location=0, dimension="width", line_color="gray", line_width=1)
traj_fig.add_layout(vline_traj)
traj_fig.add_layout(hline_traj)


# *********************************************
# --- Time-domain plot (Real and Imag) ---
# *********************************************
time_fig = figure(
    title = r"Real and Imag Components of $$\\e^{s·t}$$",
    width=800, height=300,
    x_axis_label="Time (t)",
    y_axis_label="Amplitude",
    x_range=(0, time_dur), 
    toolbar_location=None
)
time_fig.title.align = "center"
time_fig.line("t", "y", source=real_source, line_color="blue", legend_label="Real(e^{st})")
time_fig.line("t", "y", source=imag_source, line_color="orange", legend_label="Imag(e^{st})")
time_fig.legend.location = "top_left"

hline_time = Span(location=0, dimension="width", line_color="gray", line_width=1)
time_fig.add_layout(hline_time)


s_source.on_change('data', update_plots)

# Initialize once
update_plots(None,None,None)

# Layout
layout = column(row(plane_fig, traj_fig), time_fig)

curdoc().add_root(layout)
curdoc().title = "e^(s·t) Visualizer"
