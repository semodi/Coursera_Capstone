import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import flask
import os
from numpy.random import randint

data = pd.read_csv('normalized_data.csv').set_index('Neighborhood')
data_merged = pd.read_csv('merged.csv').set_index('Neighborhood')


data_merged['Venue density'] = data_merged['Venue density'].max() - data_merged['Venue density']

def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient

def find_better(data, neighborhood,return_idx = False):
    costs = 1 - data.iloc[:,1:].fillna(0)
    n_cost = costs.loc[neighborhood]
    costs = costs.values
    n_cost = n_cost.values
    
    better = (np.any(costs < n_cost,axis=1) &  ~np.any(costs > n_cost, axis = 1)  )
    if return_idx:
        return data.loc[better].index
    return data.loc[better]

data['pe'] = is_pareto_efficient(1 - data.iloc[:,1:].fillna(0).values)

costs = ['Rent','Subway distance','Crime','Midtown distance','Venue density']

THETA = np.linspace(0,1,len(data.loc[data.pe]))*2*np.pi
np.random.shuffle(THETA)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)

app.layout = \
        html.Div([
            html.Div([
            html.Label('I would like to move to '),
            dcc.Input(
            id="input_neighborhood",
            type="text",
            placeholder="Neighborhood"),
            dcc.Markdown(children='', id='out_label'),
            dcc.Markdown(children='', id='out_neighbor')
            ],style={'overflowY': 'scroll', 'height': 300,'width':1100}),
            html.Div([
                dcc.Graph(id='graph-trend')
            ],style={ 'height': 400,'width':400,'float':'left'}
            ),
            html.Div(
                [html.Label('Choose your priorities:')] +\
                [e for el in [
                (html.Label(i),
                dcc.Slider(
                id='slider_{}'.format(i),
                min=0,
                max=1,
                step=0.01,
                value=0.5,
                )) for i in costs] for e in el]
                ,style={ 'height': 250,'width':300,'float':'left'}),
           
        ],
       )

@app.callback(
    [Output('out_label','children'),
     Output('out_neighbor','children'),
     Output('graph-trend','figure')],
    [Input('input_neighborhood', 'value')] + [Input('slider_{}'.format(i),'value') for i in costs])
def update_trend_figure(neighborhood,sli_rent, sli_subway, sli_crime, sli_midtown,sli_venue):
    
    traces = []
    
    data_pe = data.loc[data['pe']].iloc[:,1:-1]
    data_pe *= [sli_rent, sli_subway, sli_crime, sli_midtown, sli_venue]
    cost = np.linalg.norm(data_pe.fillna(0).values, axis=1)
    cost = (cost-np.min(cost))/(np.max(cost)-np.min(cost))
    cost_idx = np.argsort(cost)
    borough = data.loc[data_pe.iloc[cost_idx].index].Borough
    neigh = data_pe.iloc[cost_idx].index
    cost = np.sort(cost)
    r = (1.05 - cost)**(1/(1.7))
    theta = THETA[cost_idx]
#     np.random.shuffle(theta)
    x = np.cos(theta)*r
    y = np.sin(theta)*r
    
    borough_color = [{'Bronx':'blue','Queens':'red','Manhattan':'green','Brooklyn':'orange'}[b] for b in borough]
    
    traces.append(go.Scatter(x= x,y=y, mode='markers', name='',
                             hovertemplate =\
                                '<b>%{text}</b>',
                             text = ['{}, {}'.format(n,b) for n,b in zip(neigh,borough)],
                             
                            marker=dict(
                                color=borough_color,
                                size=np.exp(cost*3.7),
                                opacity= cost,
                                line=dict(
                                    color='black',
                                    width=2
                                    )
                            )
                            )
                 )
         
    returns = []
    if neighborhood in data.index:
        if data.loc[neighborhood,'pe']:
            returns += ['Good choice!',data_merged.loc[[neighborhood]].to_markdown()]
        else:
            
            md_list =''''''
            
            for b in find_better(data, neighborhood).index:
                md_list += '* ' + str(b) +'''
'''
            
            returns += ['''Hmmm, this neighborhood is not Pareto efficient. Have you considered the following neighborhoods?''',
                    data_merged.loc[[neighborhood] + list(find_better(data, neighborhood,return_idx=True))].to_markdown()]
    else:
        returns += ['','']
    returns += [{'data': traces,
                 'layout':{
                'hovermode': 'closest',
                'transition' :{'duration': 200,
                              'animation':'cubic-in-out'},
                'margin' : {'l':0, 'b': 0, 't': 0, 'r': 0}}}]
                  
    return returns 

if __name__ == '__main__':
    app.run_server(debug=True, threaded = True)