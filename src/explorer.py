import os
import re
import base64
import json

import numpy as np
from dash import Dash, dcc, html, Input, Output
from PIL import Image
from io import BytesIO
import plotly.graph_objects as go
from tqdm import tqdm

def split_text(text, n=10):
    words = text.split()
    return '<br>'.join([' '.join(words[i:i+n]) for i in range(0, len(words), n)])

def find_images(basepath):
    images = {}
    for dirname in os.listdir(basepath):
        if os.path.isdir(os.path.join(basepath, dirname)):
            for iname in os.listdir(os.path.join(basepath, dirname)):
                ipath = os.path.join(basepath, dirname, iname)
                matched = re.match(regex, ipath)
                if matched:
                    images[(matched.group(1), int(matched.group(2)), int(matched.group(3)), int(matched.group(4)))] = ipath
    return images, ('base_img', 'guidance', 'nsteps', 'variant')

def encode(image):
    with Image.open(image) as img:
        width, height = img.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        img_cropped = img.crop((left, top, right, bottom))
        buffered = BytesIO()
        img_cropped.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('ascii')

def load_for_party(party):
    global CURRENT_STATE
    print(f'loading data for {party}')
    CURRENT_STATE['party'] = party
    # load summary, reasoning and impact points
    files = {'prompt': 'prompts/visual_impact_points.txt', 'summary': 'program_sum.txt', 'reasoning': 'program_impact_point_reasoning.txt', 'impact_points': 'program_impact_points.json'}
    for key, fname in files.items():
        try:
            with open(f'contents/2020/{party}/{fname}', 'r', encoding='utf-8') as f:
                CURRENT_STATE[key] = ' '.join(f.read().split('\n')) if fname.endswith('txt') else json.load(f)['impact_points'] # json
        except:
            CURRENT_STATE[key] = None
    # load images
    CURRENT_STATE['images'], config_values = find_images(os.path.join('contents', '2020', party, 'images'))
    unique_configs = {}
    for idx, value in enumerate(config_values):
        unique_configs[value] = list(set([key[idx] for key in CURRENT_STATE['images'].keys()]))
    # set defaults to display
    if CURRENT_STATE['config'] is None or tuple(CURRENT_STATE['config']) not in CURRENT_STATE['images']:
        CURRENT_STATE['config'] = [option[0] for option in list(unique_configs.values())]
        CURRENT_STATE['base_pic'] = CURRENT_STATE['encoded'][(os.path.join('contents', '2020', f"{CURRENT_STATE['config'][0]}.jpg"))]
    CURRENT_STATE['gen_pic_path'] = CURRENT_STATE['images'][tuple(CURRENT_STATE['config'])]
    CURRENT_STATE['gen_pic'] = CURRENT_STATE['encoded'][(CURRENT_STATE['images'][tuple(CURRENT_STATE['config'])])]
    return unique_configs

def construct_for_party(party):
    global CURRENT_STATE
    unique_configs = load_for_party(party)
    image_controls = [
        # Image row
        html.Div([
            html.Label('Base image:', style={'marginRight': '8px', 'width': '150px', 'display': 'inline-block'}),
            dcc.Dropdown(
                id='image-dropdown',
                options=[{'label': filename, 'value': filename} for filename in unique_configs['base_img']],
                value=CURRENT_STATE['config'][0],
                style={'width': '600px', 'display': 'inline-block'}
            ),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '12px'}),
        # Guidance row
        html.Div([
            html.Label('Guidance:', style={'marginRight': '8px', 'width': '150px', 'display': 'inline-block'}),
            html.Div(dcc.Slider(
                id='guidance-slider',
                min=min(np.log10(unique_configs['guidance'])),
                max=max(np.log10(unique_configs['guidance'])),
                step=1,
                value=np.log10(CURRENT_STATE['config'][1]),
                marks={np.log10(i): str(i) for i in unique_configs['guidance']},
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode='drag',
            ), style={'width': '600px', 'display': 'inline-block'})
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '12px'}),
        # Number of Steps row
        html.Div([
            html.Label('Number of Steps:', style={'marginRight': '8px', 'width': '150px', 'display': 'inline-block'}),
            html.Div(dcc.Slider(
                id='nsteps-slider',
                min=min(unique_configs['nsteps']),
                max=max(unique_configs['nsteps']),
                step=5,
                value=CURRENT_STATE['config'][2],
                marks={i: str(i) for i in unique_configs['nsteps']},
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode='drag'
            ), style={'width': '600px', 'display': 'inline-block'})
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '12px'}),
        # Variant row
        html.Div([
            html.Label('Variant:', style={'marginRight': '8px', 'width': '150px', 'display': 'inline-block'}),
            html.Div(dcc.Slider(
                id='variant-slider',
                min=min(unique_configs['variant']),
                max=max(unique_configs['variant']),
                step=1,
                value=CURRENT_STATE['config'][3],
                marks={i: str(i) for i in unique_configs['variant']},
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode='drag'
            ), style={'width': '200px', 'display': 'inline-block'})
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '12px'})
    ]
    return image_controls


########################### INITIALIZATION
app = Dash(__name__)
regex = r'.*img_(.*)_guid(\d*)_nsteps(\d*).*_(\d*).png'
PARTY_OPTIONS = ['spd', 'fdp', 'cdu', 'linke', 'gruene']
CURRENT_STATE = {'party': PARTY_OPTIONS[0], 'impact_points': None, 'images': None, 'config': None, 'gen_pic': None, 'gen_pic_path': None, 'base_pic': None, 'encoded': {}}
# pre-encode images
for subdir in tqdm(os.listdir('contents/2020'), desc='Encoding images'):
    # store base images
    if os.path.isfile(os.path.join('contents', '2020', subdir)) and (subdir.endswith('.png') or subdir.endswith('.jpg')):
        CURRENT_STATE['encoded'][os.path.join('contents', '2020', subdir)] = encode(os.path.join('contents', '2020', subdir))
    # store images in subdirectories
    for root, _, files in os.walk(os.path.join('contents', '2020', subdir)):
        for fname in files:
            if fname.endswith('.png') or fname.endswith('.jpg'):
                CURRENT_STATE['encoded'][os.path.join(root, fname)] = encode(os.path.join(root, fname))
app.layout = html.Div([
    html.Div([
        html.Div([
            # Party selection row
            html.Div([
                html.Label('Party:', style={'marginRight': '8px', 'width': '150px', 'display': 'inline-block'}),
                dcc.Dropdown(
                    id='party-dropdown',
                    options=[{'label': party.upper(), 'value': party} for party in PARTY_OPTIONS],
                    value=CURRENT_STATE['party'],
                    style={'width': '600px', 'display': 'inline-block'}
                ),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '12px'}),
            # Party Program tabs
            html.Div([
                dcc.Tabs(id="tabs", value='tab-1', children=[
                    dcc.Tab(label='Visual Aspects', value='visual-tab'),
                    dcc.Tab(label='Impact Points', value='impact-tab'),
                    dcc.Tab(label='Reasoning', value='reasoning-tab'),
                    dcc.Tab(label='Summary', value='summary-tab'),
                ]),
                html.Div(id='tabs-content')
            ]),
            # Image Control
            html.Div(construct_for_party(CURRENT_STATE['party']), id='image-controls'),
            # Image Name row
            html.Div([
                html.Label('Image Name:', style={'marginRight': '8px', 'width': '100px', 'display': 'inline-block'}),
                html.Label(id='image-name', style={'display': 'inline-block'})
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '12px'}),
        ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
        html.Div([
            html.Div([
                html.Label('Generated Image', style={'display': 'block', 'textAlign': 'center', 'marginBottom': '8px'}),
                html.Div(id='image-display', style={'display': 'inline-block', 'marginRight': '20px'})
            ], style={'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '40px'}),
            html.Div([
                html.Label('Base Image', style={'display': 'block', 'textAlign': 'center', 'marginBottom': '8px'}),
                html.Div(id='base-display', style={'display': 'inline-block'})
            ], style={'display': 'inline-block', 'verticalAlign': 'top'}),
        ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'display': 'flex', 'flexDirection': 'row', 'width': '100%'})
])


# Callback to update selected party
@app.callback(
    [Output('tabs', 'value'), Output('image-controls', 'children')],
    Input('party-dropdown', 'value')
)
def update_party(party):
    image_controls = construct_for_party(party)
    return 'visual-tab', image_controls

# Callback to update selected tab
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    global CURRENT_STATE
    try:
        if tab == 'visual-tab': # plot
            content = CURRENT_STATE['prompt']
        elif tab == 'impact-tab': # plot
            try:                
                x, y, d = [], [], []
                for point in reversed(CURRENT_STATE['impact_points']):
                    x.append(point['importance'])
                    y.append(point['identifier'])
                    desc, reas = split_text(point['description']), split_text(point['importance_reasoning'])
                    d.append(f'<br>Description: {desc}<br><br>Importance reasoning: {reas}')
            except:
                raise RuntimeError('Could not read the corresponding Impact Point JSON File!')
            fig = go.Figure(go.Bar(x=x, y=y, hovertext=d, orientation='h'))
            content = dcc.Graph(
                id='impact-point-plot',
                figure=fig,
                responsive=True,
                config={'responsive': True},
                style={'height': '100%', 'width': '100%'}
            )
        elif tab == 'summary-tab':
            content = CURRENT_STATE['summary']
        elif tab == 'reasoning-tab':
            content = CURRENT_STATE['reasoning']
        else:
            raise RuntimeError('Invalid tab id!')
    except Exception as e:
        content = html.Div('ERROR! ' + str(e).replace('\n', '<br>'))
    return html.Div(content, style={'width': '750px', 'height': '400px', 'overflowY': 'auto', 'paddingRight': '8px'})

# Callback to update image display
@app.callback(
    [Output('image-display', 'children'), Output('base-display', 'children'), Output('image-name', 'children')],
    [Input('tabs', 'value'), Input('image-dropdown', 'value'), Input('guidance-slider', 'value'), Input('nsteps-slider', 'value'), Input('variant-slider', 'value')]
)
def update_image(party_plot, base, guidance, nsteps, variant):
    global CURRENT_STATE
    for idx, field in enumerate([base, guidance, nsteps, variant]):
        if idx == 1: # rescale guidance
            field = int(10 ** field)
        if field != CURRENT_STATE['config'][idx]:
            CURRENT_STATE['config'][idx] = field
            if idx == 0:
                CURRENT_STATE['base_pic'] = CURRENT_STATE['encoded'][os.path.join('contents', '2020', f'{base}.jpg')]
    config = tuple(CURRENT_STATE['config'])
    if config in CURRENT_STATE['images']:
        CURRENT_STATE['gen_pic_path'] = CURRENT_STATE['images'][config]
        CURRENT_STATE['gen_pic'] = CURRENT_STATE['encoded'][(CURRENT_STATE['images'][config])]
    gen = html.Img(src='data:image/png;base64,{}'.format(CURRENT_STATE['gen_pic']), style={'height': '500px'})
    base = html.Img(src='data:image/png;base64,{}'.format(CURRENT_STATE['base_pic']), style={'height': '500px'})
    return gen, base, CURRENT_STATE['gen_pic_path']

if __name__ == '__main__':
    app.run_server(debug=True)