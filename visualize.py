import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale, make_colorscale
from scipy.stats import chi2_contingency

from process_survey_export import DELIM, FNAME, QMAP

QMAP.update({
    "sm2_documentation": "Please indicate to what extent you agree with the following statements: [AI models and outputs must be documented more comprehensibly.]",
    "sm2_ai_for_polit": "Please indicate to what extent you agree with the following statements: [ AI technology should be used for political education.]",
    "sm1_pro_regulation": "Please rate the extent to which you agree with the following statements: [The use of AI in political contexts should be more heavily regulated.]",
    "usef_crssm": "How useful do you think AI is for political education, in terms of various possible applications?  [Reformulation & cross-media sharing]",
    "usef_vislz": "How useful do you think AI is for political education, in terms of various possible applications?  [Visualizing political agendas]",
    "usef_trnsl": "How useful do you think AI is for political education, in terms of various possible applications?  [Translating political platforms]",
    "usef_smmrz": "How useful do you think AI is for political education, in terms of various possible applications?  [Summarzing political platforms]",
    "usef_compr": "How useful do you think AI is for political education, in terms of various possible applications?  [Comparing different agendas]",
})

def hex_to_alpha(hex, alpha):
    hex = hex.lstrip('#')
    if len(hex) == 6:
        hex += 'FF'
    r, g, b, a = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4, 6))
    a = f"{alpha:3.1f}"
    return f'rgba({r},{g},{b},{a})'

def print_init(fname):
    print(f'                 - -- ---  {fname:<30}  --- -- -                 ')
    return fname

def finalize(fig, fname, show, ws=1, hs=1, top=0, bottom=0, yshift=0):
    fig.update_layout(font_family='Open-Sherif', width=PLOT_WIDTH*ws, height=PLOT_HEIGHT*hs, margin={'l': 0, 'r': 0, 'b': bottom, 't': top})
    fig.update_annotations(yshift=2+yshift) # to adapt tex titles
    if show:
        fig.show()
    fig.write_image(os.path.join(os.path.dirname(__file__), 'figures', f"{fname}.pdf"))

QUALIF = {
    # 0: 'N/A',
    1: 'Other', # 'Basic Education',
    # 2: 'Basic School',
    3: 'Other', # 'Secondary School',
    4: 'High School',
    5: 'Other', # 'Specialist',
    6: "Bachelor's",
    7: "Master's",
    8: 'PhD'
}

AISKILL = {
    0: 'Novice',
    1: 'User',
    2: 'Developer',
    3: 'Expert'
}
R_AISKILL = {v: k for k, v in AISKILL.items()}

POLPOS = {
    1: 'Progressive',
    3: 'Centrist',
    5: 'Conservative'
}

PLOT_WIDTH = 800
PLOT_HEIGHT = PLOT_WIDTH // 3

LAMARR_COLORS = [
    '#009ee3', #0 aqua
    '#983082', #1 fresh violet
    '#ffbc29', #2 sunshine
    '#35cdb4', #3 carribean
    '#e82e82', #4 fuchsia
    '#59bdf7', #5 sky blue
    '#ec6469', #6 indian red
    '#706f6f', #7 gray
    '#4a4ad8', #8 corn flower
    '#0c122b', #9 dark corn flower
    '#ffffff'
]
LAMARR_COL_SEL = [LAMARR_COLORS[i] for i in [1, 3, 8, 6, 0, 2, 7, 9]]
LAM_COL_SCALE = make_colorscale(LAMARR_COL_SEL[:3])
LAM_COL_SCALE_REV = make_colorscale(list(reversed(LAMARR_COL_SEL[:3])))
LAM_COL_SIX = sample_colorscale(LAM_COL_SCALE, np.linspace(0, 1, 6))
LAM_COL_TEN = sample_colorscale(LAM_COL_SCALE, np.linspace(1, 0, 10))
LAM_COL_EIGHT = sample_colorscale(LAM_COL_SCALE, np.linspace(1, 0, 8))

SM_CMAP = {key: LAMARR_COL_SEL[5 - idx % 6] for idx, key in enumerate([
    "sm1_i_can_recognize",  "sm1_i_use_ai",     "sm1_use_respon_ai",    "sm1_pro_regulation",   "sm1_ai_is_reliable",   "sm1_risk_democracy",
    "sm2_documentation",    "sm2_ai_for_polit", "sm2_public_benefit",   "sm2_pro_regulation",   "sm2_polarization",     "sm2_risk_democracy"])
}

if __name__ == '__main__':

    # read survey results
    df = pd.read_csv(FNAME, sep=DELIM)
    df['ai_skill'] = df['ai_skill'].map(lambda e: R_AISKILL[e.split()[0]] if isinstance(e, str) else e)
    # binarized ai skill groups
    count_usr, count_dev = sum(df['ai_skill'] <= 1), sum(df['ai_skill'] > 1)
    df['bin_ai_skill'] = df['ai_skill'].map(lambda e: f'Novices & Users (N={count_usr})' if e <= 1 else (e if pd.isna(e) else f'Devs & Experts (N={count_dev})'))
    # binarized age groups
    count_over_30, count_under_30 = sum(df['age'] > 30), sum(df['age'] <= 30)
    df['bin_age'] = df['age'].map(lambda e: f'Number of participants (up to 30 years, N={count_under_30})' if e <= 30 else (e if pd.isna(e) else f'Number of participants (over 30 years, N={count_over_30})'))
    # clean categorical columns
    df['pol_part'] = df['pol_part'].map(lambda e: e.split(' - ')[0] if isinstance(e, str) else 'N/A')
    df['pol_pos'] = df['pol_pos'].map(lambda e: int(e.split(' - ')[0]) if isinstance(e, str) else e)
    df['qualif'] = df['qualif'].map(lambda e: QUALIF[int(e[0])] if isinstance(e, str) else e)
    df['bin_gender'] = df['gender']
    df['gender'] = df['gender'].map(lambda e: 'N/A' if pd.isna(e) else e)
    print(f"/nOn average, the participants took {df['time'].median()/60:3.1f} minutes to complete the survey and left {df.map(lambda e: 1 if pd.isna(e) else 0).sum().sum()/df.size*100:3.1f}/% of questions unanswered.")
    print(f"/nAfter taking our survey, {df[df['perc_changed']=='I now see more risks than before.'].shape[0]/df.shape[0]*100:.0f}/% of the participants clarified that they now see more risks of using AI in political contexts, while only {df[df['perc_changed']=='I now see more potential than before.'].shape[0]/df.shape[0]*100:.0f}/% mentioned realizations about additional potential./n") 
    
    # read vlm results
    # dfs = []
    # for fname, cols in zip(['evaluation_results', 'evaluation_results_llm'], [['bleu', 'rouge1_f', 'rouge2_f', 'cosine_similarity'], ['llm_score']]):
    #     dfs.append( pd.read_csv(f'D:/Repos/ai-for-political-education/src/frontend/public/political_content_2025/{fname}.csv') )
    #     dfs[-1]['pp'] = dfs[-1].apply(lambda r: f"{r['party']}_{r['source'][0]}", axis=1)
    #     dfs[-1].set_index('pp', inplace=True)
    #     dfs[-1].drop(columns=[c for c in dfs[-1].columns if c not in cols and c not in ['party', 'source']], inplace=True)
    # vlm_results = pd.concat(dfs, axis=1)
    # vlm_results = vlm_results.loc[:, ~vlm_results.columns.duplicated()]
    parties = {'linke': 'The Left', 'diepartei': 'The PARTY', 'tierschutz': 'AWP', 'gruene': 'The Greens', 'spd': 'SPD', 'fdp': 'FDP', 'cdu': 'CDU'}
    vlm_results = pd.read_csv('D:/Repos/ai-for-political-education/src/frontend/public/political_content_2025/evaluations/evaluation_results_both.csv')
    vlm_results = vlm_results[vlm_results['party'].isin(parties.keys())]
    vlm_results['llm_score'] = vlm_results['llm_score'] / 10 # rescale to [0, 1]

    # test plotly image export
    show = True
    fig = px.scatter(x=[0, 1, 2], y=[0, 1, 4])
    fig.write_image("dummy.pdf")
    os.remove("dummy.pdf")

    fname = print_init('vlm_results') ###############################################################################
    fig = make_subplots(rows=2, cols=2, shared_yaxes=True, shared_xaxes=True, horizontal_spacing=0.02, vertical_spacing=0.03, column_titles=['Party Platform', 'Election Compass Responses'])
    for c_idx, src in enumerate(['program', 'kommunalomat']):
        for r_idx, (col, cname) in enumerate({'rouge1_f': 'ROUGE-1', 'llm_score': 'LLM Score'}.items()):
            all_points = []
            for idx, (party, name) in enumerate(parties.items()):
                data = vlm_results[(vlm_results['source'] == src) & (vlm_results['party'] == party)]
                assert data.shape[0] == 5, f"Expected five entries for party {party} and source {src}, but got {data.shape[0]}"
                fig.add_trace(go.Scatter(x=data['cosine_similarity'].values, y=data[col].values, mode='markers', marker={'color': LAMARR_COL_SEL[idx]}, name=name, showlegend=c_idx+r_idx==0), row=1+r_idx, col=1+c_idx)
                for x, y in zip(data['cosine_similarity'], data[col]):
                    all_points.append((x, y))
            # add regression line
            x_vals, y_vals = np.array([p[0] for p in all_points]), np.array([p[1] for p in all_points])
            m, b = np.polyfit(x_vals, y_vals, 1)
            x0, x1 = x_vals.min()-0.05, x_vals.max()+0.05
            fig.add_trace(go.Scatter(x=[x0, x1], y=[m*x0+b, m*x1+b], mode='lines', line=dict(color='rgba(0,0,0,0.4)', dash='dash'), showlegend=False), row=1+r_idx, col=1+c_idx)
            fig.add_annotation(yref="paper", yanchor="bottom", y=m*(x0+0.1)+b, text=f"r={np.corrcoef(x_vals, y_vals)[0,1]:.2f}", xref="paper", xanchor="center", x=x0+0.03, showarrow=False, font=dict(size=14), row=1+r_idx, col=1+c_idx)
            if c_idx == 0:
                fig.update_yaxes(title=cname, row=1+r_idx, col=1+c_idx)
                fig.update_xaxes(title='Cosine Similarity', row=2, col=1+r_idx)
    finalize(fig, fname, top=19, show=show)

    fname = print_init('ai_for_politics') ###############################################################################
    questions = [
        "Which AI-generated political content have you encountered?",
        "For which of the following tasks do you see high usefulness of AI?",
        "Have you used AI applications for political education?"
    ]
    row_df_cols = {0: {}, 1: {}, 2: {}}
    # prepare the relevant columns for unified binary information on all three questions
    for col in df.columns:
        for col_start, func, q_idx in zip(['enc_', 'usef_'], [lambda e: e == 'Yes', lambda e: e > 3], [0, 1]):
            if col.startswith(col_start): # answer columns to question 1 or 2
                df[f'bin_{col}'] = df[col].map(lambda e: func(e))
                if df[f'bin_{col}'].sum() > 0:
                    row_df_cols[q_idx][f'bin_{col}'] = QMAP[col].split('[')[1][:-1]
    for key in pd.unique(df['would_use']): # answer values for question 3
        if not pd.isna(key):
            df[f'bin_{key}'] = df['would_use'].map(lambda e: True if e==key else False)
            row_df_cols[2][f'bin_{key}'] = key
    # perform statistical significance tests (chi2)
    stat_pvalues = {}
    for split_col in ['bin_age', 'bin_ai_skill', 'bin_gender']:
        gr_counts = {}
        for r_idx, df_cols in row_df_cols.items():
            for col in df_cols.keys():
                results = {}
                for split, s_data in df.groupby(split_col):
                    results[split] = s_data[col].sum()
                gr_counts[col] = results
        gr_counts = pd.DataFrame(gr_counts).transpose()
        stat_pvalues[split_col] = [chi2_contingency(gr_counts.loc[list(cols.keys())])[1] for cols in row_df_cols.values()] # calc p-value for each question with respective cols
    for q_idx, ques in enumerate(questions):
        print(f'{ques:<80}' + ' - '.join([f"{split}: p={pvals[q_idx]:.2f}" for split, pvals in stat_pvalues.items()]))
    headings = [f'{head} (p-value={pval:4.2f})' for pval, head in zip(stat_pvalues['bin_age'], questions)]
    # create plot
    fig = make_subplots(rows=3, cols=2, shared_yaxes=True, shared_xaxes=True, vertical_spacing=0.08, horizontal_spacing=0.03, row_heights=[0.3, 0.4, 0.3])
    for c_idx, (age_label, a_data) in enumerate(df.groupby('bin_age')):
        for r_idx, df_cols in row_df_cols.items():
            for g_idx, (gender, g_data) in enumerate(a_data.groupby('gender')):
                y, x = zip(*[(y_descr, g_data[col].sum()) for col, y_descr in df_cols.items()])
                t = [''] * len(x)
                if g_idx == 2:
                    t = [f"{(a_data[col].sum()/a_data.shape[0] * 100):.1f}%" for col in df_cols.keys()] # only add the relative numbers for the N/A plot (counts for all genders)
                fig.add_trace(go.Bar(y=y, x=x, text=t, textposition='outside', orientation='h', name=gender, marker_color=LAMARR_COL_SEL[g_idx], showlegend=c_idx+r_idx==0), row=r_idx+1, col=c_idx+1)
            if c_idx == 1:
                fig.update_yaxes(range=[-0.5, len(df_cols)-0.5], row=1+r_idx)
        fig.update_xaxes(title=age_label, row=3, col=c_idx+1)
    for t, y in zip(headings, [1, 0.66, 0.24]):
        fig.add_annotation(yref="paper", yanchor="bottom", y=y, text=t, xref="paper", xanchor="center", x=0.5, showarrow=False, font=dict(size=16))
    fig.update_layout(barmode='stack', legend=dict(title='Gender', yanchor="top", y=1, xanchor="right", x=-0.25))
    fig.update_xaxes(range=[0, count_over_30], col=1)
    fig.update_xaxes(range=[0, count_under_30+3], col=2)
    finalize(fig, fname, hs=1.29, top=19, bottom=45, show=show)

    fname = print_init('correct_visualization') ###############################################################################
    fig = make_subplots(rows=2, cols=2, shared_yaxes=True, vertical_spacing=0.18, horizontal_spacing=0.03)
    for c_idx, (skill, s_data) in enumerate(df.groupby('bin_ai_skill')):
        for idx, (col, descr) in enumerate(zip(['descr_2', 'descr_3', 'descr_1'], ['Wrong Image/Prompt', 'Correct Image (Variant)', 'Correct Image/Prompt'])):
            fig.add_trace(go.Box(x=s_data[col], name=descr, showlegend=False, marker_color=LAMARR_COL_SEL[idx]), row=1, col=1+c_idx)
        fig.update_xaxes(tickvals=[1, 2, 3, 4, 5], range=[0.65, 5.35], ticktext=['Mismatching', '', '', '', 'Matching'], row=1, col=c_idx+1)
        count_descr = {a[:3]: s_data[s_data['which_descr'] == a].shape[0] for a in pd.unique(s_data['which_descr']) if isinstance(a, str)}
        bar_x, bar_y = zip(*[(t, count_descr[k] if k in count_descr else 0) for k, t in {'"Re': 'Wrong Prompt 2', '"So': 'Wrong Prompt 1', '"Af': 'Correct Prompt', 'Non': 'N/A'}.items()])
        bar_text = [f'{(count / s_data.shape[0] * 100):.1f}%' for count in bar_y]
        fig.add_trace(go.Bar(y=bar_x, x=bar_y, text=bar_text, orientation='h', marker_color=LAMARR_COL_SEL[3], showlegend=False, textposition='outside'), row=2, col=c_idx+1)
        fig.update_xaxes(title=skill, range=[0, count_dev if c_idx==0 else count_usr], row=2, col=c_idx+1)
    fig.add_annotation(yref="paper", yanchor="bottom", y=1, text="How well do the three generated images match the given prompt?", xref="paper", xanchor="center", x=0.5, showarrow=False, font=dict(size=16))
    fig.add_annotation(yref="paper", yanchor="bottom", y=0.4, text="Which of these three prompts matches the image best?", xref="paper", xanchor="center", x=0.5, showarrow=False, font=dict(size=16))
    finalize(fig, fname, hs=1, top=19, show=show)

    fname = print_init('statement_support') ###############################################################################
    sm1_cols = {col: QMAP[col].split('[')[1][:-1].strip() for col in SM_CMAP.keys() if col.startswith('sm1')}
    sm2_cols = {col: QMAP[col].split('[')[1][:-1].strip() for col in SM_CMAP.keys() if col.startswith('sm2')}
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.1, horizontal_spacing=0.04, subplot_titles=[s for s, _ in df.groupby('bin_ai_skill')])
    for c_idx, (skill, s_data) in enumerate(df.groupby('bin_ai_skill')):
        for r_idx, sm_cols in enumerate([sm1_cols, sm2_cols]):
            for col, tcol in sm_cols.items():
                fig.add_trace(go.Box(x=s_data[col], name=tcol, showlegend=False, marker_color=SM_CMAP[col]), row=1+r_idx, col=1+c_idx)
        fig.update_xaxes(tickvals=[1, 2, 3, 4, 5], ticktext=['Disagree', '', '', '', 'Agree'], row=2, col=c_idx+1)
    fig.add_annotation(yref="paper", yanchor="bottom", y=0.99, text="Asked at the start of the survey:", xref="paper", xanchor="center", x=-0.23, showarrow=False, font=dict(size=16))
    fig.add_annotation(yref="paper", yanchor="bottom", y=0.44, text="Asked at the end of the survey:", xref="paper", xanchor="center", x=-0.23, showarrow=False, font=dict(size=16))
    finalize(fig, fname, top=19, show=show)

    fname = print_init('survey_participants') ###############################################################################
    fig = make_subplots(1, 3, shared_yaxes=True, horizontal_spacing=0.02, subplot_titles=['AI Skill', 'Qualification', 'Political Leaning'], column_widths=[0.28, 0.39, 0.28])
    for (g, gdf), c in zip(df.groupby('gender'), [LAMARR_COL_SEL[0], LAMARR_COL_SEL[1], LAMARR_COL_SEL[2]]):
        g = f'{g} (N={gdf.shape[0]})'
        for i, (x, map) in enumerate(zip(['ai_skill', 'qualif', 'pol_pos'], [AISKILL, QUALIF, POLPOS])):
            x_gfd = gdf[[x,'age']].dropna()
            fig.add_trace(go.Scatter(x=x_gfd[x], y=x_gfd['age'], mode='markers', marker={'color': c, 'size': 5},
                                     name=g, legendgroup=g, showlegend=i==0), row=1, col=i+1)
            if pd.api.types.is_numeric_dtype(gdf[x]):
                fig.update_xaxes(tickvals=list(map.keys()), ticktext=list(map.values()), row=1, col=i+1)
            else:
                fig.update_xaxes(categoryorder='array', categoryarray=list(map.values()), row=1, col=i+1)
    fig.add_trace(go.Scatter(x=[5.4], y=[50], mode='markers', marker={'color': 'rgba(0,0,0,0)'}, showlegend=False), row=1, col=3) # add hidden point
    fig.add_trace(go.Scatter(x=[0.6], y=[50], mode='markers', marker={'color': 'rgba(0,0,0,0)'}, showlegend=False), row=1, col=3) # add hidden point
    fig.update_yaxes(title='Age', range=[17, 80], row=1, col=1)
    fig.update_layout(legend=dict(yanchor="top", y=1, xanchor="center", x=0.5, orientation="h"))
    finalize(fig, fname, top=19, show=show)

    print('Qualitative results:')
    for feedback in df['qualitative'].dropna():
        print(f'/n{feedback}')
