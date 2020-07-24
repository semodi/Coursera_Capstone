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

costs = ['Rent','Subway distance','Safety','Midtown distance','Venue density']

THETA = np.linspace(0,1,len(data.loc[data.pe]))*2*np.pi
np.random.shuffle(THETA)

import json
import plotly.express as px

with open('data/newyork_polygon.json') as json_data:
    newyork_data = json.load(json_data)


selected = data.Crime.to_frame()

selected.columns= ['selection']

selected['selection'] = 0
selected.to_json('selection.json')

fig = px.choropleth(selected.reset_index(), geojson=newyork_data, locations='Neighborhood', color='selection',
                           color_continuous_scale="twilight",
                           featureidkey="properties.neighborhood",
#                            range_color=(0, 12),
                           projection='mercator'
    )
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},coloraxis={'showscale':False})
    # fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
maps = fig
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)

def build_map(selection):
    
    selected = data.Crime.to_frame()

    selected.columns= ['selection']
    
    selected['selection'] = selection
    fig = px.choropleth(selected.reset_index(), geojson=newyork_data, locations='Neighborhood', color='selection',
                           color_continuous_scale="twilight",
                            featureidkey="properties.neighborhood",
#                            range_color=(0, 12),
                           projection='mercator'
    )
    fig.update_traces({'text' : ['{}'.format(n) for n in selected.index]})
    return fig

app.layout = \
        html.Div([
            html.H2('NYC-Neighborhoods'),
            html.Div([
                html.Label('I would like to move to '),
                dcc.Input(
                id="input_neighborhood",
                type="text",
                placeholder="Neighborhood"),
                dcc.Markdown(children='', id='out_label')],style={'width':1100}),
                html.Div([
                dcc.Markdown(children='', id='out_neighbor')
                ],style={'overflowY': 'scroll', 'height': 300,'width':1100,'margin' : {'l':10, 'b': 0, 't': 0, 'r': 100}}),
                html.Div([
                html.Div(
                    [html.H6('Choose your priorities:')] +\
                    [e for el in [
                    (html.Label(i),
                    dcc.Slider(
                    id='slider_{}'.format(i),
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.5,
#                     updatemode='drag'
                    )) for i in costs] for e in el]
                    ,style={ 'height': 250,'width':250,'float':'left','margin' : {'l':0, 'b': 0, 't': 200, 'r': 0}}),
                 html.Div([
                    dcc.Graph(id='graph-trend')
                 ],style={ 'height': 400,'width':400,'float':'left','margin' : {'l':0, 'b': 0, 't': 200, 'r': 0}}
                 )]),
                 html.Div(id='intermediate-value',children=[], style={'display': 'none'}),
                 html.Div([
                    dcc.Graph(id='graph-map',figure=maps)
                ],style={ 'height': 400,'width':450,'float':'left','margin' : {'l':0, 'b': 0, 't': 200, 'r': 0}}
                ),
            ],
    )

@app.callback(
    [Output('out_label','children'),
     Output('out_neighbor','children'),
     Output('graph-trend','figure'),
     Output('intermediate-value','children')],
    [Input('input_neighborhood', 'value')] + [Input('slider_{}'.format(i),'value') for i in costs] + \
    [Input('graph-trend','hoverData') ])
def update_trend_figure(neighborhood,sli_rent, sli_subway, sli_crime, sli_midtown,sli_venue, mouseHover):
    
    traces = []
    with open('hoverfile.json','w') as file:
        file.write(json.dumps(mouseHover,indent=4))
    data_pe = data.loc[data['pe']].iloc[:,1:-1]
    data_pe *= [sli_rent, sli_subway, sli_crime, sli_midtown, sli_venue]
    cost = np.linalg.norm(data_pe.fillna(0).values, axis=1)
    cost = (cost-np.min(cost))/(np.max(cost)-np.min(cost))
    cost_idx = np.argsort(cost)
    borough = data.loc[data_pe.iloc[cost_idx].index].Borough
    neigh = data_pe.iloc[cost_idx].index
    cost = np.sort(cost)
    r = (1.05 - cost)**(1/(1.7))
    r /= np.max(r)
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
    
    #Hack to fix axis range (for stable animations):
    traces.append(go.Scatter(x= [-1,1,-1,1],y=[-1,1,1,-1], mode='markers', name='',
                            marker=dict(
                                color=borough_color,
                                size=0.01,
                                opacity= 1.0,
                            )
                            )
                 )
   
    
    returns = []
    selected = data.Crime.to_frame()

    selected.columns= ['selection']

    selected['selection'] = 0
   
    
    if neighborhood in data.index:
        selected.loc[neighborhood] = 1
        
    if neighborhood in data.index:
        if data.loc[neighborhood,'pe']:
            selected.loc[neighborhood] = 1
            returns += ['Good choice!',data_merged.loc[[neighborhood]].to_markdown()]
        else:
            
            md_list =''''''
            
            for b in find_better(data, neighborhood).index:
                selected.loc[b] = -1
                md_list += '* ' + str(b) +'''
'''
            
            returns += ['''Hmmm, this neighborhood is not Pareto efficient. Have you considered the following neighborhoods?''',
                    data_merged.loc[[neighborhood] + list(find_better(data, neighborhood,return_idx=True))].to_markdown()]
    else:
        returns += ['','']
    
    try:
        hover_neigh = mouseHover['points'][0]['text'].split(',')[0]

        if hover_neigh in selected.index:
            selected.loc[hover_neigh] = 0.5
    except Exception:
        pass

    selected.to_json('selection.json')
    returns += [{'data': traces,
                 'layout':{
                 'xaxes':{'range':[-1,1]},
                 'yaxes':{'range':[-1,1]},
                 'hovermode': 'closest',
                 'transition' :{'duration': 200},
                 'autosize':False,
                 'showlegend':False,
                 'margin' : {'l':10, 'b': 0, 't': 0, 'r': 10},
                 'width': 400,
                 'height': 400
                 }}]
    returns += [[selected.to_json()]]
    return returns 

@app.callback(
    [Output('graph-map','figure')],
    [Input('out_neighbor','children'),
     Input('intermediate-value','children')])
def update_map(out_neigh, selected):
    
    
    selected = pd.read_json(selected[0])
    fig = build_map(tuple(selected['selection'].values.tolist()))
 

    maps = fig
    return [{'data':maps.data,
                'layout':{
                    'margin' : {'l':10, 'b': 0, 't': 0, 'r': 10},
                    'geo':{
                        'fitbounds' : "locations",
                        'visible' :False,
                        'projection':go.layout.geo.Projection(type = 'mercator')
                    },
                    
                    'hoverinfo' : 'text',
                    'text' : ['{}'.format(n) for n in selected.index],
                    'showlegend':False,
                    'coloraxis':{'showscale':False,'range_color':[-1,1]},
                    'uirevision':'input_neighborhood'}
                }]


if __name__ == '__main__':
    app.run_server(debug=True, threaded = True)
