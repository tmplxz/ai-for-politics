import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale, make_colorscale
from scipy.stats import chi2_contingency, wilcoxon, mannwhitneyu, ttest_1samp, t, binomtest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
import statsmodels.formula.api as smf

from process_survey_export import DELIM, FNAME, QMAP

def hex_to_alpha(hex, alpha):
    hex = hex.lstrip('#')
    if len(hex) == 6:
        hex += 'FF'
    r, g, b, a = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4, 6))
    a = f"{alpha:3.1f}"
    return f'rgba({r},{g},{b},{a})'

def print_init(fname):
    print(f'\n                 - -- ---  {fname:<30}  --- -- -                 \n')
    return fname

def finalize(fig, fname, show, ws=1, hs=1, top=0, bottom=0, yshift=0):
    fig.update_layout(font = dict(color='#000000'), font_family='Open-Sherif', width=PLOT_WIDTH*ws, height=PLOT_HEIGHT*hs, margin={'l': 0, 'r': 0, 'b': bottom, 't': top})
    fig.update_annotations(yshift=2+yshift) # to adapt tex titles
    if show:
        fig.show()
    fig.write_image(os.path.join(os.path.dirname(__file__), 'figures', f"{fname}.pdf"))

def p_lookup(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'

def z_approx(stat, N):
    return (stat - N*(N+1)/4)/np.sqrt(N*(N+1)*(2*N+1)/24)

# --- Helper function: bootstrap CI for effect size r ---
def bootstrap_ci_r(sample_A, sample_B, n_boot=2000):
    N = len(sample_A)
    r_vals = []
    for _ in range(n_boot):
        idxs = np.random.choice(np.arange(N), N, replace=True)
        a_resamp = sample_A[idxs]
        b_resamp = sample_B[idxs]
        stat,_ = wilcoxon(a_resamp, b_resamp)
        r = np.abs(z_approx(stat, N) / np.sqrt(N))
        r_vals.append(r)
    ci_lower=np.percentile(r_vals,2.5)
    ci_upper=np.percentile(r_vals,97.5)
    return ci_lower,ci_upper

def mann_whitney_grouped(df, col, group_col, alpha=0.05, n_boot=1000, seed=None):
    """Compute Mann-Whitney U between two groups defined by `group_col` for column `col`.
    Returns U, p, effect size r and bootstrap CI for r if possible.
    Assumes exactly two groups (will take first two encountered)."""
    groups = []
    labels = []
    for g, gdf in df.groupby(group_col):
        # take raw values (not pair-masked here)
        arr = gdf[col].dropna().values
        groups.append(arr)
        labels.append(g)
        if len(groups) == 2:
            break
    if len(groups) < 2:
        return None
    x, y = groups[0], groups[1]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return None
    try:
        U, p = mannwhitneyu(x, y, alternative='two-sided')
    except Exception:
        U, p = np.nan, 1.0
    mean_U = n1 * n2 / 2.0
    var_U = n1 * n2 * (n1 + n2 + 1) / 12.0
    z = (U - mean_U) / np.sqrt(var_U) if not (np.isnan(U) or var_U == 0) else np.nan
    r = z / np.sqrt(n1 + n2) if not np.isnan(z) else np.nan
    out = {'U': U, 'p_val': p, 'n1': n1, 'n2': n2, 'r': r}

    # bootstrap CI for r
    if n_boot and n1 + n2 > 0:
        rng = np.random.default_rng(seed)
        r_boots = []
        for _ in range(n_boot):
            xs = rng.choice(x, size=n1, replace=True)
            ys = rng.choice(y, size=n2, replace=True)
            try:
                Ub, _ = mannwhitneyu(xs, ys, alternative='two-sided')
            except Exception:
                r_boots.append(np.nan)
                continue
            mean_Ub = n1 * n2 / 2.0
            var_Ub = n1 * n2 * (n1 + n2 + 1) / 12.0
            zb = (Ub - mean_Ub) / np.sqrt(var_Ub) if var_Ub != 0 else np.nan
            r_boots.append(zb / np.sqrt(n1 + n2) if not np.isnan(zb) else np.nan)
        r_boots = np.array(r_boots)
        r_boots = r_boots[~np.isnan(r_boots)]
        out['ci_lo'] = np.percentile(r_boots, alpha*100)
        out['ci_hi'] = np.percentile(r_boots, (1-alpha)*100)

    return out

def mann_whitney_on_change(df, col_pre, col_post, group_col, alpha=0.05, n_boot=1000, seed=None):
    """Compute Mann-Whitney U on change scores (post - pre) between groups."""
    tmp = df[[col_pre, col_post, group_col]].dropna()
    tmp['diff'] = tmp[col_post] - tmp[col_pre]
    return mann_whitney_grouped(tmp, 'diff', group_col, alpha=alpha, n_boot=n_boot, seed=seed)

QMAP.update({
    "sm2_ai_for_polit": "Please indicate to what extent you agree with the following statements: [AI should be used for political education.]",
    "sm2_risk_democracy": "Please indicate to what extent you agree with the following statements: [S6: AI poses a risk to democratic systems.]",
    "sm2_pro_regulation": "Please indicate to what extent you agree with the following statements: [S7: Generative AI should be stronger regulated.]",
    "sm2_public_benefit": "Please indicate to what extent you agree with the following statements: [S8: AI is developed for public-benefit purposes.]",
    "sm2_polarization": "Please indicate to what extent you agree with the following statements: [S9: AI promotes political polarization.]",
    "sm2_documentation": "Please indicate to what extent you agree with the following statements: [S10: AI use must be better documented.]",

    "sm1_i_use_ai": "Please rate the extent to which you agree with the following statements: [I use AI to form my own political opinions.]",
    "sm1_risk_democracy": "Please rate the extent to which you agree with the following statements: [S1: Generative AI poses a risk to democracy.]",
    "sm1_pro_regulation": "Please rate the extent to which you agree with the following statements: [S2: Political AI use needs more regulatation.]",
    "sm1_use_respon_ai": "Please rate the extent to which you agree with the following statements: [S3: I trust political actors to use AI responsibly.]",
    "sm1_ai_unreliable": "Please rate the extent to which you agree with the following statements: [S4: AI unreliably processes political content.]",
    "sm1_i_cant_recognize": "Please rate the extent to which you agree with the following statements: [S5: I can't recognize AI-generated content.]",

    "usef_crssm": "How useful do you think AI is for political education, in terms of various possible applications?  [Reformulation & cross-media sharing]",
    "usef_vislz": "How useful do you think AI is for political education, in terms of various possible applications?  [Visualizing political agendas]",
    "usef_trnsl": "How useful do you think AI is for political education, in terms of various possible applications?  [Translating political platforms]",
    "usef_smmrz": "How useful do you think AI is for political education, in terms of various possible applications?  [Summarizing political platforms]",
    "usef_compr": "How useful do you think AI is for political education, in terms of various possible applications?  [Comparing different agendas]",
})

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
LAMARR_COL_SEL = [LAMARR_COLORS[i] for i in [1, 6, 0, 3, 8, 2, 7, 9]]
LAM_COL_SCALE = make_colorscale(LAMARR_COL_SEL[:3])
LAM_COL_SCALE_REV = make_colorscale(list(reversed(LAMARR_COL_SEL[:3])))
LAM_COL_SIX = sample_colorscale(LAM_COL_SCALE, np.linspace(0, 1, 6))
LAM_COL_TEN = sample_colorscale(LAM_COL_SCALE, np.linspace(1, 0, 10))
LAM_COL_EIGHT = sample_colorscale(LAM_COL_SCALE, np.linspace(1, 0, 8))

SM_CMAP = [
    ["sm1_i_cant_recognize",    "sm1_ai_unreliable",    "sm1_use_respon_ai",    "sm1_pro_regulation",   "sm1_risk_democracy"], # "sm1_i_use_ai"
    ["sm2_documentation",       "sm2_polarization",     "sm2_public_benefit",   "sm2_pro_regulation",   "sm2_risk_democracy"] # "sm2_ai_for_polit"
]

# constants for confidence intervals and significance testing
ALPHA = 0.05
NBOOT = 200

if __name__ == '__main__':

    # read survey results
    df = pd.read_csv(FNAME, sep=DELIM).set_index('id')
    df['ai_skill'] = df['ai_skill'].map(lambda e: R_AISKILL[e.split()[0]] if isinstance(e, str) else e)
    # binarized ai skill groups
    count_usr, count_dev = sum(df['ai_skill'] <= 1), sum(df['ai_skill'] > 1)
    df['bin_ai_skill'] = df['ai_skill'].map(lambda e: f'Novices & Users (N={count_usr})' if e <= 1 else (e if pd.isna(e) else f'Devs & Experts (N={count_dev})'))
    bin_ai_skill = list(reversed([v for v in pd.unique(df['bin_ai_skill']) if not pd.isna(v)]))
    # binarized age groups
    AGE_CUT = df['age'].median() - 1
    count_young, count_old = sum(df['age'] <= AGE_CUT), sum(df['age'] > AGE_CUT)
    df['bin_age'] = df['age'].map(lambda e: f'Up to {AGE_CUT:.0f} years (N={count_young})' if e <= AGE_CUT else (e if pd.isna(e) else f'Over {AGE_CUT:.0f} years (N={count_old})'))
    bin_age = list(reversed([v for v in pd.unique(df['bin_age']) if not pd.isna(v)]))
    # clean categorical columns
    df['pol_part'] = df['pol_part'].map(lambda e: e.split(' - ')[0] if isinstance(e, str) else 'N/A')
    df['pol_pos'] = df['pol_pos'].map(lambda e: int(e.split(' - ')[0]) if isinstance(e, str) else e)
    df['qualif_cat'] = df['qualif'].map(lambda e: QUALIF[int(e[0])] if isinstance(e, str) else e)
    df['qualif'] = df['qualif'].map(lambda e: int(e[0]) if isinstance(e, str) else e)
    # binarized gender groups
    gender_counts = {g: gdf.shape[0] for g, gdf in df.groupby('gender')}
    gender_counts[np.nan] = df.shape[0] - df['gender'].dropna().size
    df['gender'] = df['gender'].map(lambda e: f'{"Other" if pd.isna(e) else e} (N={gender_counts[e]})')
    df['bin_gender'] = df['gender'].map(lambda e: np.nan if 'Other' in e else e)
    bin_gender = [pd.unique(df['gender'])[1], pd.unique(df['gender'])[2], pd.unique(df['gender'])[0]]
    assert bin_gender[0].startswith('M') and bin_gender[1].startswith('F') and bin_gender[2].startswith('O')
    # fix categorical would use responses
    map_would_use = {key: key.replace("'", "").replace(" ", "").replace(".", "").replace(",", "") for key in pd.unique(df['would_use']) if isinstance(key, str)}
    map_would_use_rev = {v: k for k, v in map_would_use.items()}
    df['would_use'] = df['would_use'].map(lambda e: map_would_use[e] if isinstance(e, str) else e)
    OVERALL = f'Overall (N={df.shape[0]})'
    GROUP_VARS = {"bin_ai_skill": "AI Skill", "bin_gender": "Gender", "bin_age": "Age"}
    CMAP = {key: LAMARR_COL_SEL[col] for key, col in [ # for plotting in unified colors
        (OVERALL, 0),
        (bin_age[0], 1),
        (bin_age[1], 4),
        (bin_gender[0], 2),
        (bin_gender[1], 5),
        (bin_gender[2], 0),
        (bin_ai_skill[0], 3),
        (bin_ai_skill[1], 6),
        ('start', 0),
        ('end', 6),
    ]}
    CMAP.update({gv: LAMARR_COL_SEL[idx] for idx, gv in zip([3, 2, 1], GROUP_VARS.keys())})    
    
    # print some general survey statistics
    print(f"/nOn average, the participants took {df['time'].median()/60:3.1f} minutes to complete the survey and left {df.map(lambda e: 1 if pd.isna(e) else 0).sum().sum()/df.size*100:3.1f}/% of questions unanswered.")
    print(f"Amount of German participants: {df[df['Country']=='Germany'].shape[0] / df.shape[0]*100:.1f}% Median age: {AGE_CUT+1:.1f}/n")
    print(f"/nAfter taking our survey, {df[df['perc_changed']=='I now see more risks than before.'].shape[0]/df.shape[0]*100:.0f}/% of the participants clarified that they now see more risks of using AI in political contexts, while only {df[df['perc_changed']=='I now see more potential than before.'].shape[0]/df.shape[0]*100:.0f}/% mentioned realizations about additional potential./n")
    for col, name in zip(['bin_gender', 'bin_age', 'bin_ai_skill'], ['Gender', 'Age', 'AI Skill']):
        print(f'Binary group info for {name}: {pd.unique(df[col])}')
    
    # read vlm results
    parties = {'linke': 'The Left', 'diepartei': 'The PARTY', 'tierschutz': 'AWP', 'gruene': 'The Greens', 'spd': 'SPD', 'fdp': 'FDP', 'cdu': 'CDU'}
    vlm_results = pd.read_csv('D:/Repos/ai-for-political-education/src/frontend/public/political_content_2025/evaluations/evaluation_results_both.csv')
    vlm_results = vlm_results[vlm_results['party'].isin(parties.keys())]
    vlm_results['llm_score'] = vlm_results['llm_score'] / 10 # rescale to [0, 1]

    show = True
    fname = print_init('colormap') ##################################################### test plotly image export with printing the colormap
    fig = go.Figure()
    for idx, (key, color) in enumerate(CMAP.items()):
        fig.add_trace(go.Bar( y=[key], x=[1], orientation='h', marker=dict(color=color), name=color, showlegend=False))
    fig.update_layout(title='Color Palette', yaxis_title='Color Key')
    finalize(fig, fname, top=19, show=show)

    fname = print_init('vlm_results') ###############################################################################
    fig = make_subplots(rows=2, cols=2, shared_yaxes=True, shared_xaxes=True, horizontal_spacing=0.02, vertical_spacing=0.03, column_titles=['Input: Party platform', 'Input: Election compass responses'])
    for c_idx, src in enumerate(['program', 'kommunalomat']):
        for r_idx, (col, cname) in enumerate({'rouge1_f': 'ROUGE-1', 'llm_score': 'LLM score'}.items()):
            # create scatter traces and collect points per subplot
            all_points, traces = [], []
            for idx, (party, name) in enumerate(parties.items()):
                data = vlm_results[(vlm_results['source'] == src) & (vlm_results['party'] == party)]
                assert data.shape[0] == 5, f"Expected five entries for party {party} and source {src}, but got {data.shape[0]}"
                traces.append(go.Scatter(x=data['cosine_similarity'].values, y=data[col].values, mode='markers', marker={'color': LAMARR_COL_SEL[idx]}, name=name, showlegend=c_idx+r_idx==0))
                for x, y in zip(data['cosine_similarity'], data[col]):
                    all_points.append((x, y))
            # add regression line
            x_vals, y_vals = np.array([p[0] for p in all_points]), np.array([p[1] for p in all_points])
            m, b = np.polyfit(x_vals, y_vals, 1)
            x0, x1 = x_vals.min()-0.05, x_vals.max()+0.05
            traces.insert(0, go.Scatter(x=[x0, x1], y=[m*x0+b, m*x1+b], mode='lines', line=dict(color='rgba(0,0,0,0.4)', dash='dash'), name='Regression', showlegend=r_idx+c_idx==0))
            for tr in traces:
                fig.add_trace(tr, row=1+r_idx, col=1+c_idx)
            fig.add_annotation(yref="paper", yanchor="bottom", y=m*(x0+0.1)+b, text=f"r={np.corrcoef(x_vals, y_vals)[0,1]:.2f}", xref="paper", xanchor="center", x=x0+0.03, showarrow=False, font=dict(size=14), row=1+r_idx, col=1+c_idx)
            if c_idx == 0:
                fig.update_yaxes(title=cname, row=1+r_idx, col=1+c_idx)
                fig.update_xaxes(title='Cosine similarity', row=2, col=1+r_idx)
    finalize(fig, fname, top=19, show=show)

    fname = print_init('correct_visualization') ###############################################################################
    # TEST A: are differences > 0 ? (t-test with effect size and confidence intervals)
    acc_stats = df[[f"descr_{i}" for i in range(1,4)] + ["which_descr"] + list(GROUP_VARS.keys())].dropna()
    acc_stats["mean_true"] = acc_stats[["descr_1", "descr_3"]].mean(axis=1)
    acc_stats["diff"] = acc_stats["mean_true"] - acc_stats["descr_2"]
    t_stat, p_val = ttest_1samp(acc_stats["diff"], popmean=0)
    mean_diff, std_diff = acc_stats["diff"].mean(), acc_stats["diff"].std(ddof=1)
    cohen_dz = mean_diff / std_diff
    n = acc_stats.shape[0]
    se_mean = std_diff / np.sqrt(n)
    ci_low = mean_diff + t.ppf(ALPHA/2, n-1)*se_mean
    ci_high = mean_diff + t.ppf(1-ALPHA/2, n-1)*se_mean
    print('TASK A: Score match of given prompt with three images => are differences to the false image > 0?')
    print(f"T-test: t={t_stat:.3f}, p={p_val:.4E} Mean discrimination score: {mean_diff:.3f}, Cohen's dz={cohen_dz:.3f} 95% CI for mean difference: [{ci_low:.3f}, {ci_high:.3f}]")
    print('Wilcoxon signed-rank test:', wilcoxon(acc_stats["diff"]))
    # TEST B: Is the amount of "none" votes significantly above chance level? (Binomial test)
    acc_choices = {'Non': 'None of them', '"Re': 'P3 (Wrong)', '"So': 'P2 (Wrong)', '"Af': 'P1 (Correct)'}
    corr_choice = [ch for ch in acc_choices.values() if 'Correct' in ch][0]
    non_choice = [ch for ch in acc_choices.values() if 'Non' in ch][0]
    acc_stats['correct_choice'] = acc_stats['which_descr'].map(lambda e: int(acc_choices[e[:3]]==corr_choice))
    acc_stats['non_choice'] = acc_stats['which_descr'].map(lambda e: int(acc_choices[e[:3]]==non_choice))
    print('TASK B: Identify the correct prompt for the given image => significantly more "none" or "correct" votes than by chance?')
    p_chance, n = 0.25, acc_stats.shape[0]
    for col, name in zip(['correct_choice', 'non_choice'], ['Significantly correct choices?', 'Significantly NONE choices?']):
        k = sum(acc_stats[col])
        bintest = binomtest(k, n, p=p_chance, alternative='greater')
        ci_lo, ci_hi = proportion_confint(k, n, method='wilson', alpha=ALPHA)
        print(f'{name:<30} | {bintest} with 95% Wilson CI: {ci_lo:.2f} -- {ci_hi:.2f}')
    # perform model-based statistical testing of group differences (SMF OLS & logistic regression)
    acc_gr_stats = []
    for gvar in GROUP_VARS.keys():
        for col, hyp in zip(['correct_choice', 'non_choice'], ['B: Corr choice?', 'B: None choice?']):
            model = smf.logit(formula=f"{col} ~ C({gvar})", data=acc_stats).fit(disp=False)
            cgroup = [p for p in model.params.index if p.startswith(f"C({gvar})")][0]
            acc_gr_stats.append( {"hyp": hyp, "gv": gvar, "cg": cgroup.split(')[T.')[1][:-1], "coef": model.params[cgroup], "p_val": model.pvalues[cgroup]} )
        model = smf.ols(formula=f"diff ~ C({gvar})", data=acc_stats).fit(disp=False)
        cgroup = [p for p in model.params.index if p.startswith(f"C({gvar})")][0]
        acc_gr_stats.append( {"hyp": 'A: Distinguish?', "gv": gvar, "cg": cgroup.split(')[T.')[1][:-1], "coef": model.params[cgroup], "p_val": model.pvalues[cgroup]} )
    acc_gr_stats = pd.DataFrame(acc_gr_stats)
    acc_gr_stats["odds_ratio"] = np.exp(acc_gr_stats["coef"])
    acc_gr_stats['ci_lo'] = np.exp(acc_gr_stats['coef'] - 1.96 * acc_gr_stats['coef'].std()) # 1.96 factor comes from the Gaussian distribution
    acc_gr_stats['ci_hi'] = np.exp(acc_gr_stats['coef'] + 1.96 * acc_gr_stats['coef'].std()) # 1.96 factor comes from the Gaussian distribution
    for _, res in acc_gr_stats.iterrows():
        print(f"H: {res['hyp'][:40]:<40} | G: {res['gv']:<15} (comparing {res['cg'][:10]:<10} against rest) | Coef: {res['coef']:6.3f} | OR: {res['odds_ratio']:5.3f} | p-value: {res['p_val']:6.4f}")
    # visualize
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.25, horizontal_spacing=0.13, subplot_titles=["Task A: How well do the three images?", "Task B: Which prompt best matches the image?", "", ""])
    # boxplot distributions (1/4)
    x, y = [], []
    for col, descr in zip(['descr_2', 'descr_3', 'descr_1'], ['I3 (Wrong)', 'I2 (Correct)', 'I1 (Correct)']):
        x.extend(acc_stats[col].dropna())
        y.extend([descr]*acc_stats[col].dropna().shape[0])
    fig.add_trace(go.Box(x=x, y=y, marker_color=CMAP[OVERALL], orientation="h", showlegend=False), row=1, col=1)
    fig.add_shape(type="line", x0=2, x1=3, y0=0, y1=1, row=1, col=1)
    fig.add_shape(type="line", x0=2, x1=3, y0=0, y1=2, row=1, col=1)
    fig.add_annotation(x=4, y=0.4, showarrow=False, text=f'*signif. diff. (p={p_val:.1E})', row=1, col=1)
    # accuracy bars (2/4)
    count_descr = {a[:3]: acc_stats[acc_stats['which_descr'] == a].shape[0] for a in pd.unique(acc_stats['which_descr']) if isinstance(a, str)}
    bar_x, bar_y, bar_text = zip(*[(t, count_descr[k], f' {count_descr[k]/acc_stats.shape[0]*100:.1f}%') for k, t in acc_choices.items()])
    fig.add_trace(go.Bar(y=bar_x, x=bar_y, text=bar_text, orientation='h', marker_color=CMAP[OVERALL], name=OVERALL, textposition='outside'), row=1, col=2)
    for idx, acc in enumerate(bar_y[1:]):
        fig.add_shape(type="line", x0=bar_y[0]/2, x1=acc/2, y0=0, y1=idx+1, row=1, col=2)
    fig.add_annotation(x=bar_y[0]/2-5, y=0.65, showarrow=False, xanchor='left', text=f'*signif. diff. (p={bintest.pvalue:.1E})', row=1, col=2)
    # group effect scatter plot (3/4)
    for g_idx, group_var in enumerate(GROUP_VARS.keys()):
        for g_idx2, (group, g_data) in enumerate(df.groupby(group_var)):
            color = CMAP[group]
            s_stats = acc_stats.loc[[idx for idx in g_data.index if idx in acc_stats.index]]
            x, sd_x = s_stats["diff"].mean(), s_stats["diff"].std(ddof=1)
            error_x = dict(type='data', array=[sd_x], visible=True)
            n, k = g_data["which_descr"].dropna().shape[0], sum([acc_choices[e[:3]]==corr_choice for e in g_data["which_descr"].dropna()])
            p = k/n
            sd_y = 1.96*np.sqrt((p*(1-p)/n))
            error_y = dict(type='data', array=[sd_y*100], visible=True)
            fig.add_trace(go.Scatter(x=[x], y=[p*100], error_x=error_x, error_y=error_y, name=group, marker_color=color, marker_size=200*sd_x*sd_y), row=2, col=1)
        # forest plot for group effects (4/4)
        sub_stats = acc_gr_stats[(acc_gr_stats['gv'] == group_var)]
        error_x = dict(type='data', symmetric=False, array=sub_stats['ci_hi']-sub_stats['odds_ratio'], arrayminus=sub_stats['odds_ratio']-sub_stats['ci_lo'])
        fig.add_trace(go.Scatter(x=sub_stats['odds_ratio'], y=sub_stats['hyp'], error_x=error_x, mode='markers', showlegend=False,
                                 marker={'size': 8, 'color': CMAP[group_var]}), row=2, col=2)
        for _, row in sub_stats.iterrows(): # annotate significant p values
            if row['p_val'] < 0.05:
                fig.add_annotation(y=row['a'], text=f"p={row['p_val']:.3f}", x=np.log10(row['odds_ratio']), ax=40, ay=-5, font=dict(size=10), row=r_idx+1, col=2)
        fig.add_vline(x=1, line_dash="dash", line_color="gray", row=r_idx+1, col=2)
    fig.update_xaxes(title='Likert scoring', tickvals=[1, 2, 3, 4, 5], range=[0.65, 5.35], ticktext=['Mismatching', '', '', '', 'Matching'], row=1,col=1)
    fig.update_xaxes(title="Number of participants", range=[0, 69], row=1, col=2)
    fig.update_xaxes(title="A: Mean discrimination", range=[0.6, 0.82], row=2, col=1)
    fig.update_xaxes(title='Odds ratio (log scale)', type="log", row=2, col=2)
    fig.update_yaxes(title="B: Accuracy [%]", range=[6, 22], row=2, col=1)
    fig.update_layout(legend=dict(title='Group info', yanchor="top", y=-0.2, xanchor="center", x=0.5, orientation='h'))    
    finalize(fig, fname, hs=1.2, top=19, show=show)

    fname = print_init('statement_support') ###############################################################################
    # statistical testing: paired Wilcoxon signed-rank tests for each statement (pre vs post) & group differences
    pre_post_wilcox_mw = {}
    for col1, col2 in zip(*SM_CMAP):
        sub_df = df[[col1,col2]].dropna()
        pre_row, post_row = sub_df[col1].values, sub_df[col2].values
        # Wilcoxon + Bootstap CI + effect size
        stat, p_val = wilcoxon(pre_row, post_row)
        N=sub_df.shape[0]
        r_effect=np.abs(z_approx(stat, N) / np.sqrt(N))
        ci_low, ci_high = bootstrap_ci_r(pre_row, post_row, n_boot=NBOOT)
        pre_post_wilcox_mw[(col1,col2)] = {"p_val": p_val, "r": r_effect, "ci_lo": ci_low, "ci_hi": ci_high}
        # Mann-Whitney comparisons (by AI skill) for pre, post, and change
        for gr_var in GROUP_VARS.keys():
            mw_pre = mann_whitney_grouped(df, col1, gr_var, alpha=ALPHA, n_boot=NBOOT, seed=42)
            mw_post = mann_whitney_grouped(df, col2, gr_var, alpha=ALPHA, n_boot=NBOOT, seed=42)
            mw_change = mann_whitney_on_change(df, col1, col2, gr_var, alpha=ALPHA, n_boot=NBOOT, seed=42)
            for col, res in zip([col1, col2, 'change'], [mw_pre, mw_post, mw_change]):
                for key, val in res.items():
                    pre_post_wilcox_mw[(col1,col2)][f'{gr_var}_{col}_{key}'] = val
    # FDR correction on p-values
    pre_post_wilcox_mw = pd.DataFrame(pre_post_wilcox_mw).transpose()
    for col in pre_post_wilcox_mw.columns:
        if 'p_val' in col:
            pre_post_wilcox_mw[f'{col}_fdr'] = multipletests(pre_post_wilcox_mw[col], method="fdr_bh")[1]
    # plotting
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04)
    # column 1: distributions
    for q_idx, (cols) in enumerate(zip(*SM_CMAP)):
        for idx, (col, annot) in enumerate(zip(reversed(cols), ['end', 'start'])):
            fig.add_trace(go.Box(x=df[col].tolist(), y=[QMAP[col].split('[')[1][:-1].strip()] * df.shape[0],
                                 name=f'Answered at {annot}', marker_color=CMAP[annot], orientation='h', showlegend=q_idx==0), row=1, col=1)
            stats = pre_post_wilcox_mw.loc[(cols)]
            if stats['p_val_fdr'] < 0.05:
                x0, x1 = df[cols[0]].median()-0.1, df[cols[1]].median()-0.1
                fig.add_shape(type="line", x0=x0, x1=x1, y0=q_idx*2+1, y1=q_idx*2, row=1, col=1)
                fig.add_annotation(x=x0+(x1-x0)/2+0.05+(x1-x0)/5, y=q_idx*2+0.35, showarrow=False, xanchor='left', text='*', row=1, col=1)
        fig.update_xaxes(title='Distributions for statement support', range=[0.8, 5.5], tickvals=[1, 2, 3, 4, 5], ticktext=['Disagree', '', '', '', 'Agree'], row=1, col=1)
    # column 2: group effects
    for gr_var, gr_name in GROUP_VARS.items():
        p_y_r = []
        for q_idx, (cols) in enumerate(zip(*SM_CMAP)):
            for idx, (col, annot) in enumerate(zip(reversed(cols), ['end', 'start'])):
                p = pre_post_wilcox_mw[f'{gr_var}_{col}_p_val'].dropna().iloc[0]
                r = pre_post_wilcox_mw[f'{gr_var}_{col}_r'].dropna().iloc[0]
                p_y_r.append((p, QMAP[col].split('[')[1][:-1].strip(), r))
        p, y, r = zip(*p_y_r)
        for p_, y_, r_ in p_y_r:
            if p_ < 0.05:
                fig.add_annotation(x=r_+0.01, y=y_, showarrow=False, xanchor='left', text='*', row=1, col=2)
        fig.add_trace(go.Scatter(x=r, y=y, marker={'color': CMAP[gr_var], 'size': np.log(p)*-2}, name=gr_name), row=1, col=2)
        fig.update_xaxes(title='Effect size r (Mann-Whitney U)', row=1, col=2)
    fig.add_annotation(x=0.17, y=8.5, showarrow=False, xanchor='left', text='*signif. diff.', row=1, col=2)
    fig.update_layout(legend=dict(yanchor="top", y=-0.3, xanchor="center", x=0.5, orientation='h'))
    finalize(fig, fname, show=show)

    fname = print_init('ai_for_politics') ###############################################################################
    questions = [
        "Which AI-generated political content have you encountered?",
        "For which political AI application do you see high usefulness?",
        "Have you used AI applications for political education?"
    ]
    # prepare the relevant columns for unified binary information on all three questions
    df_quest_cols = {q: {} for q in questions}
    for col in df.columns:
        for col_start, func, q in zip(['enc_', 'usef_'], [lambda e: int(e == 'Yes'), lambda e: int(e > 3)], questions):
            if col.startswith(col_start): # answer columns to question 1 or 2
                df[f'bin_{col}'] = df[col].map(lambda e: func(e))
                if df[f'bin_{col}'].sum() > 0:
                    df_quest_cols[q][f'bin_{col}'] = QMAP[col].split('[')[1][:-1]
    for key in pd.unique(df['would_use']): # answer values for question 3
        if not pd.isna(key):
            df[f'bin_{key}'] = df['would_use'].map(lambda e: int(e==key))
            df_quest_cols[questions[2]][f'bin_{key}'] = map_would_use_rev[key]
    # perform model-based statistical testing of group differences (SMF logistic regression)
    ai4p_stat_results = []
    for question, cols in df_quest_cols.items():
        for col in cols.keys():
            for gvar in GROUP_VARS.keys():
                model = smf.logit(formula=f"{col} ~ C({gvar})", data=df).fit(disp=False)
                cgroup = [p for p in model.params.index if p.startswith(f"C({gvar})")][0]
                ai4p_stat_results.append( {"q": question, "a": df_quest_cols[question][col], "gv": gvar, "cg": cgroup.split(')[T.')[1][:-1], "coef": model.params[cgroup], "p_val": model.pvalues[cgroup]} )
    ai4p_stat_results = pd.DataFrame(ai4p_stat_results)
    ai4p_stat_results["odds_ratio"] = np.exp(ai4p_stat_results["coef"])
    ai4p_stat_results['ci_lo'] = np.exp(ai4p_stat_results['coef'] - 1.96 * ai4p_stat_results['coef'].std()) # 1.96 factor comes from Gaussian distribution
    ai4p_stat_results['ci_hi'] = np.exp(ai4p_stat_results['coef'] + 1.96 * ai4p_stat_results['coef'].std()) # 1.96 factor comes from Gaussian distribution
    for _, res in ai4p_stat_results.iterrows():
        print(f"Q: {res['q'][:40]:<40} | A: {res['a'][:15]:<15} | G: {res['gv']:<15} (comparing {res['cg'][:10]:<10} against rest) | Coef: {res['coef']:6.3f} | OR: {res['odds_ratio']:5.3f} | p-value: {res['p_val']:6.4f}")
    # plot group statistics
    # fig = go.Figure()
    # for gvar, subset in ai4p_stat_results.groupby('gv'):
    #     fig.add_trace(go.Scatter(
    #         x=subset['odds_ratio'], y=subset['a'], mode='markers', name=gvar, marker=dict(size=10), customdata=subset[['p_val']],
    #         error_x=dict(type='data', symmetric=False, array=subset['ci_hi']-subset['odds_ratio'], arrayminus=subset['odds_ratio'] - subset['ci_lo']),
    #         hovertemplate=(f"<b>{gvar}</b><br>" + "Question: %{y}<br>" + "Odds Ratio: %{x:.2f}<br>" + "p-value: %{customdata:.4f}"),
    #     ))
    # fig.add_vline(x=1, line_dash="dash", line_color="gray") # Add reference line at OR=1 (no effect)
    # fig.update_layout(title="Forest Plot of Group Effects on AI-related Responses", xaxis_title="Odds Ratio (log scale)", yaxis_title="Response Variable", xaxis_type="log", template="plotly_white", legend_title_text="Group Variable", height=800)
    # fig.show()
    # perform additional statistical significance tests for group differences (chi2)
    stat_pvalues = {}
    for split_col in GROUP_VARS.keys():
        gr_counts = {}
        for r_idx, df_cols in df_quest_cols.items():
            for col in df_cols.keys():
                results = {}
                for split, s_data in df.groupby(split_col):
                    results[split] = s_data[col].sum()
                gr_counts[col] = results
        gr_counts = pd.DataFrame(gr_counts).transpose()
        stat_pvalues[split_col] = [chi2_contingency(gr_counts.loc[list(cols.keys())])[1] for cols in df_quest_cols.values()] # calc p-value for each question with respective cols
    for q_idx, ques in enumerate(questions):
        print(f'{ques:<80}' + ' - '.join([f"{split}: p={pvals[q_idx]:.2f}" for split, pvals in stat_pvalues.items()]))
    # create plot
    fig = make_subplots(rows=3, cols=2, shared_yaxes=True, vertical_spacing=0.08, horizontal_spacing=0.03, row_heights=[0.3, 0.4, 0.3])
    for r_idx, (question, df_cols) in enumerate(df_quest_cols.items()):
        # column 2: forst plot of group effects
        for g_idx, (gvar, gname) in enumerate(GROUP_VARS.items()):
            sub_stats = ai4p_stat_results[(ai4p_stat_results['gv'] == gvar) & (ai4p_stat_results['q'] == question)]
            error_x = dict(type='data', symmetric=False, array=sub_stats['ci_hi']-sub_stats['odds_ratio'], arrayminus=sub_stats['odds_ratio']-sub_stats['ci_lo'])
            fig.add_trace(go.Scatter(x=sub_stats['odds_ratio'], y=sub_stats['a'], error_x=error_x, mode='markers', name=gname,
                                     marker={'size': 8, 'color': CMAP[gvar]}, showlegend=r_idx==0), row=r_idx+1, col=2)
            for _, row in sub_stats.iterrows(): # annotate significant p values
                if row['p_val'] < 0.05:
                    fig.add_annotation(y=row['a'], text=f"p={row['p_val']:.3f}", x=np.log10(row['odds_ratio']), ax=40, ay=-5, font=dict(size=10), row=r_idx+1, col=2)
        fig.add_vline(x=1, line_dash="dash", line_color="gray", row=r_idx+1, col=2)
        # column 1: amount of (grouped) participants that encountered / see usefulness / would use AI for politics
        for g_idx, (group, g_data) in enumerate(df.groupby('bin_age')):
            y, x = zip(*[(y_descr, g_data[col].sum()) for col, y_descr in df_cols.items()])
            t = [''] * len(x)
            if g_idx == 1:
                t = [f"{(df[col].sum()/df.shape[0] * 100):.1f}%" for col in df_cols.keys()] # only add the relative numbers for the N/A plot (counts for all genders)
            fig.add_trace(go.Bar(y=y, x=x, text=t, textposition='outside', orientation='h', name=group, marker_color=CMAP[group], showlegend=r_idx==0), row=r_idx+1, col=1)
        fig.update_yaxes(range=[-0.5, len(df_cols)-0.5], row=1+r_idx, col=1)
        # set x axes
        if r_idx == 2:
            fig.update_xaxes(title='Number of participants', tickvals=[0, 20, 40, 60, 80], ticktext=['0', '20', '40', '60', '80'], range=[0, df.shape[0]], row=3, col=1)
            fig.update_xaxes(title='Odds ratio (log scale)', tickvals=[0.1, 0.2, 0.5, 1, 2, 5, 10], ticktext=([0.1, 0.2, 0.5, 1, 2, 5, 10]), range=[np.log10(0.09), np.log10(18)], type="log", row=3, col=2)
        else:
            fig.update_xaxes(range=[0, df.shape[0]], tickvals=[0, 20, 40, 60, 80], ticktext=['', '', '', '', ''], row=r_idx+1, col=1)
            fig.update_xaxes(range=[np.log10(0.09), np.log10(18)], tickvals=[0.1, 0.2, 0.5, 1, 2, 5, 10], ticktext=['', '', '', '', '', '', ''], type="log", row=r_idx+1, col=2)
    # add question annotations
    for t, y in zip(questions, [1, 0.66, 0.24]):
        fig.add_annotation(yref="paper", yanchor="bottom", y=y, text=t, xref="paper", xanchor="center", x=0.5, showarrow=False, arrowhead=1, font=dict(size=16))
    fig.update_layout(barmode='stack', legend=dict(title='Group info', yanchor="top", y=-0.15, xanchor="center", x=0.4, orientation='h'))
    finalize(fig, fname, hs=1.39, top=19, show=show)

    fname = print_init('survey_participants') ###############################################################################
    xaxes_titles = ['Level of AI skill', 'Highest qualification', 'Political leaning']
    fig = make_subplots(1, 3, shared_yaxes=True, horizontal_spacing=0.02, column_widths=[0.28, 0.39, 0.28])
    for (g, gdf) in df.groupby('gender'):
        for c_idx, (x, map) in enumerate(zip(['ai_skill', 'qualif_cat', 'pol_pos'], [AISKILL, QUALIF, POLPOS])):
            x_gfd = gdf[[x,'age']].dropna()
            fig.add_trace(go.Scatter(x=x_gfd[x], y=x_gfd['age'], mode='markers', marker={'color': CMAP[g], 'size': 5},
                                     name=g, legendgroup=g, showlegend=c_idx==0), row=1, col=c_idx+1)
            if pd.api.types.is_numeric_dtype(gdf[x]):
                fig.update_xaxes(title=xaxes_titles[c_idx], tickvals=list(map.keys()), ticktext=list(map.values()), row=1, col=c_idx+1)
            else:
                fig.update_xaxes(title=xaxes_titles[c_idx], categoryorder='array', categoryarray=list(map.values()), row=1, col=c_idx+1)
            if g == bin_gender[0]:
                fig.add_hline(y=df['age'].median(), line_dash='dash', line_color='rgba(0,0,0,0.2)', row=1, col=c_idx+1)
                try:
                    x_med = df[x].median()
                except Exception: # for qualif
                    x_med = QUALIF[df[x.replace('_cat', '')].median()]
                fig.add_vline(x=x_med, line_dash='dash', line_color='rgba(0,0,0,0.2)', row=1, col=c_idx+1)
    fig.add_trace(go.Scatter(x=[5.4], y=[50], mode='markers', marker={'color': 'rgba(0,0,0,0)'}, showlegend=False), row=1, col=3) # add hidden point
    fig.add_trace(go.Scatter(x=[0.6], y=[50], mode='markers', marker={'color': 'rgba(0,0,0,0)'}, showlegend=False), row=1, col=3) # add hidden point
    fig.add_trace(go.Scatter(x=[0, 0], y=[1, 1], mode='lines', line=dict(color='rgba(0,0,0,0.2)', dash='dash'), name='Median'))
    fig.update_yaxes(title='Age', range=[17, 80], row=1, col=1)
    fig.update_layout(legend=dict(yanchor="top", y=1, xanchor="center", x=0.5, orientation="h"))
    finalize(fig, fname, show=show)

    print('Qualitative results:')
    for feedback in df['qualitative'].dropna():
        print(f'/n{feedback}')
