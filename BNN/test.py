import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.offline as pyo
from scipy.stats import norm
import plotly.express as px

# Sample data
time = np.linspace(0, 10, 100)
mean_pred_lst = np.sin(time) + 5
var_pred_lst = np.cos(time) + 7
true_lst = np.sin(time) + 5  # Replace with your actual data
alpha = 0.1  # Replace with your desired alpha value
x_plot = np.linspace(0, 1, 100)  # Replace with your desired x values
B_RMSE = 0.0  # Replace with your Bayesian RMSE value
D_RMSE = 0.0  # Replace with your Deterministic RMSE value
engine = 1  # Replace with your engine number

# Create a subplot with 2 rows and 1 column for the sub-plot and table
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                    subplot_titles=[f'RUL prediction of engine {engine}', 'Table'])

std_dev = np.sqrt(var_pred_lst)

for i in range(len(mean_pred_lst)):
    
    y_sub = np.linspace(mean_pred_lst[i] - 3 * std_dev[i], mean_pred_lst[i] + 3 * std_dev[i], 100)
    x_sub = norm.pdf(y_sub, mean_pred_lst[i], std_dev[i])

    fig.add_trace(go.Scatter(x=x_sub, 
                            y=y_sub, 
                            visible=False,
                            mode='lines', 
                            line=dict(color='blue'),
                            fill='tozeroy',
                            name='RUL prediction distribution'),
                            row=2,
                            col=1)
    
    fig.add_trace(go.Scatter(x=np.array([0, max(x_sub)]),
                            y=np.array([true_lst[0], true_lst[0]]),
                            mode='lines',
                            line=dict(dash='dash', color='red'),
                            name='True RUL'),
                            row=2,
                            col=1)
    
    fig.add_trace(go.Scatter(x=x_plot, 
                            y=mean_pred_lst, 
                            mode='lines', 
                            line = dict(color='blue'),
                            name=f'Bayesian Mean Predicted RUL values for engine {engine}, RMSE = {B_RMSE}'),
                            row=1,
                            col=1)

for i in range(3):
    fig.data[i].visible = True

# Update layout
# Create and add slider
steps = []
for i in range(len(x_plot)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(x_plot)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Cycle: "},
    pad={"t": 0},
    steps=steps
)]

fig.update_layout(
    sliders=sliders,
    title=f'RUL prediction of engine {engine}', 
    showlegend=False
)


fig.update_layout(title=f'RUL prediction of engine {engine}', showlegend=False)
fig.show()

# Export the figure to an HTML file
pyo.plot(fig, filename='plotly_layout.html', auto_open=False)

#%%