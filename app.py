import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)

# ── Confederation colours ────────────────────────────────────────────────────
CONF_COLORS = {
    'UEFA':     '#003399',
    'CONMEBOL': '#009B3A',
    'CAF':      '#CC0001',
    'AFC':      '#FF6600',
    'CONCACAF': '#CC0099',
    'OFC':      '#00AACC',
    'TBD':      '#888888',
}

TEAM_CONF = {
    'Argentina': 'CONMEBOL', 'Brazil': 'CONMEBOL', 'Colombia': 'CONMEBOL',
    'Ecuador': 'CONMEBOL', 'Uruguay': 'CONMEBOL', 'Paraguay': 'CONMEBOL',
    'Spain': 'UEFA', 'England': 'UEFA', 'France': 'UEFA', 'Germany': 'UEFA',
    'Portugal': 'UEFA', 'Netherlands': 'UEFA', 'Norway': 'UEFA',
    'Switzerland': 'UEFA', 'Croatia': 'UEFA', 'Belgium': 'UEFA',
    'Scotland': 'UEFA', 'Austria': 'UEFA', 'Sweden': 'UEFA',
    'Bosnia and Herzegovina': 'UEFA', 'Czech Republic': 'UEFA',
    'Turkey': 'UEFA',
    'Morocco': 'CAF', 'Senegal': 'CAF', 'Egypt': 'CAF', 'Ivory Coast': 'CAF',
    'Algeria': 'CAF', 'South Africa': 'CAF', 'Tunisia': 'CAF',
    'DR Congo': 'CAF', 'Ghana': 'CAF', 'Cape Verde': 'CAF',
    'Japan': 'AFC', 'South Korea': 'AFC', 'Australia': 'AFC', 'Iran': 'AFC',
    'Saudi Arabia': 'AFC', 'Qatar': 'AFC', 'Jordan': 'AFC',
    'Uzbekistan': 'AFC', 'Iraq': 'AFC',
    'United States': 'CONCACAF', 'Mexico': 'CONCACAF', 'Canada': 'CONCACAF',
    'Panama': 'CONCACAF', 'Haiti': 'CONCACAF', 'Curacao': 'CONCACAF',
    'New Zealand': 'OFC',
}

FLAG_EMOJI = {
    'Argentina': '🇦🇷', 'Brazil': '🇧🇷', 'Colombia': '🇨🇴', 'Ecuador': '🇪🇨',
    'Uruguay': '🇺🇾', 'Paraguay': '🇵🇾', 'Spain': '🇪🇸', 'England': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
    'France': '🇫🇷', 'Germany': '🇩🇪', 'Portugal': '🇵🇹', 'Netherlands': '🇳🇱',
    'Norway': '🇳🇴', 'Switzerland': '🇨🇭', 'Croatia': '🇭🇷', 'Belgium': '🇧🇪',
    'Scotland': '🏴󠁧󠁢󠁳󠁣󠁴󠁿', 'Austria': '🇦🇹', 'Sweden': '🇸🇪',
    'Bosnia and Herzegovina': '🇧🇦', 'Czech Republic': '🇨🇿', 'Turkey': '🇹🇷',
    'Morocco': '🇲🇦', 'Senegal': '🇸🇳', 'Egypt': '🇪🇬', 'Ivory Coast': '🇨🇮',
    'Algeria': '🇩🇿', 'South Africa': '🇿🇦', 'Tunisia': '🇹🇳',
    'DR Congo': '🇨🇩', 'Ghana': '🇬🇭', 'Cape Verde': '🇨🇻',
    'Japan': '🇯🇵', 'South Korea': '🇰🇷', 'Australia': '🇦🇺', 'Iran': '🇮🇷',
    'Saudi Arabia': '🇸🇦', 'Qatar': '🇶🇦', 'Jordan': '🇯🇴',
    'Uzbekistan': '🇺🇿', 'Iraq': '🇮🇶',
    'United States': '🇺🇸', 'Mexico': '🇲🇽', 'Canada': '🇨🇦',
    'Panama': '🇵🇦', 'Haiti': '🇭🇹', 'Curacao': '🇨🇼',
    'New Zealand': '🇳🇿',
}

GROUPS = {
    'A': ['Mexico', 'South Korea', 'South Africa', 'Czech Republic'],
    'B': ['Canada', 'Switzerland', 'Qatar', 'Bosnia and Herzegovina'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curacao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Tunisia', 'Sweden'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Norway', 'Iraq'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'Uzbekistan', 'Colombia', 'DR Congo'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

# ── Bracket constants ────────────────────────────────────────────────────────
DATASET_NAME = {'Curacaio': 'Curacao'}  # enriched name → group name

R32_BRACKET_SLOTS = [
    ('1A', '2B'), ('1C', '2D'), ('1E', '2F'), ('1G', '2H'),
    ('1I', '2J'), ('1K', '2L'), ('1B', '2A'), ('1D', '2C'),
    ('1F', '2E'), ('1H', '2G'), ('1J', '2I'), ('1L', '2K'),
]

# ── Bracket simulation ────────────────────────────────────────────────────────
def get_predicted_bracket():
    # Use full DC strengths (attack + defense) — same model as Monte Carlo
    import os
    strengths_path = 'wc2026_strengths.csv'
    if os.path.exists(strengths_path):
        s_df = pd.read_csv(strengths_path).set_index('team')
        attack_s  = s_df['attack'].to_dict()
        defense_s = s_df['defense'].to_dict()
    else:
        # Fallback to composite_strength if not yet generated
        _enriched = load_enriched()
        attack_s = {}
        for _, row in _enriched.iterrows():
            gname = DATASET_NAME.get(row['team_name'], row['team_name'])
            attack_s[gname] = (float(row.get('composite_strength') or 50) - 50) / 50
        defense_s = {t: 0.0 for t in attack_s}

    def win_prob(ta, tb):
        # Full DC model: xg = exp(atk_a - def_b), higher defense = harder to score on
        xg_a = np.exp(attack_s.get(ta, 0) - defense_s.get(tb, 0))
        xg_b = np.exp(attack_s.get(tb, 0) - defense_s.get(ta, 0))
        return xg_a / (xg_a + xg_b)

    # Simulate group stage (expected value — no randomness)
    group_standings = {}
    for grp, teams in GROUPS.items():
        scores = {t: 0.0 for t in teams}
        for i in range(4):
            for j in range(i + 1, 4):
                ta, tb = teams[i], teams[j]
                p = win_prob(ta, tb)
                scores[ta] += p * 3
                scores[tb] += (1 - p) * 3
        group_standings[grp] = sorted(teams, key=lambda t: scores[t], reverse=True)

    slot_map = {}
    for grp, ranked in group_standings.items():
        slot_map[f'1{grp}'] = ranked[0]
        slot_map[f'2{grp}'] = ranked[1]

    thirds = sorted(
        [(group_standings[g][2], attack_s.get(group_standings[g][2], 0)) for g in GROUPS],
        key=lambda x: -x[1]
    )
    best_thirds = [t[0] for t in thirds[:8]]

    r32_raw = [(slot_map[a], slot_map[b]) for a, b in R32_BRACKET_SLOTS]
    for i in range(0, 8, 2):
        r32_raw.append((best_thirds[i], best_thirds[i + 1]))

    bracket = {}

    def sim_round(matches, name):
        results, winners = [], []
        for ta, tb in matches:
            p = win_prob(ta, tb)
            w = ta if p >= 0.5 else tb
            results.append({'team_a': ta, 'team_b': tb, 'prob_a': round(p, 3), 'winner': w})
            winners.append(w)
        bracket[name] = results
        return winners

    w32 = sim_round(r32_raw, 'R32')
    w16 = sim_round([(w32[i], w32[i + 1]) for i in range(0, 16, 2)], 'R16')
    wqf = sim_round([(w16[i], w16[i + 1]) for i in range(0, 8, 2)], 'QF')
    wsf = sim_round([(wqf[i], wqf[i + 1]) for i in range(0, 4, 2)], 'SF')
    sim_round([(wsf[0], wsf[1])], 'Final')

    return bracket, group_standings


def match_card_html(match):
    ta, tb = match['team_a'], match['team_b']
    pa = match['prob_a']
    winner = match['winner']

    def team_row(team, prob, is_winner):
        conf  = TEAM_CONF.get(team, 'TBD')
        color = CONF_COLORS.get(conf, '#888888')
        bg    = color + '28' if is_winner else 'transparent'
        fw    = '700' if is_winner else '400'
        tc    = 'white' if is_winner else '#999'
        bar   = f'<div style="background:{color};height:3px;width:{prob*100:.0f}%;border-radius:2px;margin-top:3px;"></div>'
        return (
            f'<div style="background:{bg};border-radius:5px;padding:0.35rem 0.5rem;margin-bottom:2px;">'
            f'<span style="color:{tc};font-weight:{fw}">{FLAG_EMOJI.get(team,"🏳️")} {team}</span>'
            f'<span style="float:right;color:{color};font-size:0.85rem">{prob*100:.0f}%</span>'
            f'{bar}</div>'
        )

    adv = f'<div style="color:#FFD700;font-size:0.7rem;text-align:right;margin-top:4px">🏆 {winner} advances</div>'
    return (
        '<div style="background:#1e1e2e;border-radius:8px;padding:0.6rem;'
        'margin-bottom:0.5rem;border:1px solid #2d2d3d;">'
        + team_row(ta, pa, ta == winner)
        + '<div style="border-top:1px solid #333;margin:3px 0;"></div>'
        + team_row(tb, 1 - pa, tb == winner)
        + adv + '</div>'
    )


def draw_bracket_fig(bracket):
    ROUNDS    = ['R32', 'R16', 'QF', 'SF', 'Final']
    ROUND_X   = [0, 6.2, 11.8, 16.2, 19.8]
    BOX_W     = 5.5
    BOX_HALF  = 0.42   # half-height of one team row

    def yc(r, i):
        # Centers spaced 2 units in R32, doubling each round
        return (2 ** (r + 1)) * (i + 0.5)

    fig = go.Figure()

    # Round labels
    labels = ['Round of 32', 'Round of 16', 'Quarter-Finals', 'Semi-Finals', 'Final']
    for label, rx in zip(labels, ROUND_X):
        fig.add_annotation(x=rx + BOX_W / 2, y=-0.8, text=f'<b>{label}</b>',
                           showarrow=False, font=dict(color='#718096', size=11),
                           xanchor='center')

    for r_idx, rname in enumerate(ROUNDS):
        if rname not in bracket:
            continue
        rx = ROUND_X[r_idx]
        for m_idx, match in enumerate(bracket[rname]):
            y_c  = yc(r_idx, m_idx)
            ta, tb   = match['team_a'], match['team_b']
            pa       = match['prob_a']
            winner   = match['winner']

            for team, prob, is_top in [(ta, pa, True), (tb, 1 - pa, False)]:
                row_top = y_c - 2 * BOX_HALF if is_top else y_c
                row_bot = y_c if is_top else y_c + 2 * BOX_HALF
                row_mid = (row_top + row_bot) / 2

                conf  = TEAM_CONF.get(team, 'TBD')
                color = CONF_COLORS.get(conf, '#888888')
                won   = team == winner
                # Convert hex to rgba for fill
                hx = color.lstrip('#')
                r_, g_, b_ = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
                fill = f'rgba({r_},{g_},{b_},0.18)' if won else '#161b22'
                border_c = color if won else '#30363d'
                border_w = 1.5 if won else 0.5

                fig.add_shape(type='rect', x0=rx, x1=rx + BOX_W,
                              y0=row_top, y1=row_bot,
                              fillcolor=fill,
                              line=dict(color=border_c, width=border_w),
                              layer='below')
                fig.add_annotation(
                    x=rx + 0.25, y=row_mid,
                    text=f'{FLAG_EMOJI.get(team,"🏳️")} {team}',
                    xanchor='left', yanchor='middle',
                    font=dict(size=9, color='white' if won else '#8b949e'),
                    showarrow=False)
                fig.add_annotation(
                    x=rx + BOX_W - 0.15, y=row_mid,
                    text=f'{prob * 100:.0f}%',
                    xanchor='right', yanchor='middle',
                    font=dict(size=8, color=color),
                    showarrow=False)

            # Divider between the two teams
            fig.add_shape(type='line', x0=rx, x1=rx + BOX_W, y0=y_c, y1=y_c,
                          line=dict(color='#30363d', width=0.5))

            # Elbow connector to next round
            if r_idx < len(ROUNDS) - 1:
                next_rx  = ROUND_X[r_idx + 1]
                next_yc  = yc(r_idx + 1, m_idx // 2)
                win_mid  = y_c - BOX_HALF if winner == ta else y_c + BOX_HALF
                mid_x    = rx + BOX_W + (next_rx - rx - BOX_W) * 0.5
                lc = '#3a3a4a'
                fig.add_shape(type='line', x0=rx + BOX_W, x1=mid_x,
                              y0=win_mid, y1=win_mid, line=dict(color=lc, width=1))
                fig.add_shape(type='line', x0=mid_x, x1=mid_x,
                              y0=win_mid, y1=next_yc, line=dict(color=lc, width=1))
                fig.add_shape(type='line', x0=mid_x, x1=next_rx,
                              y0=next_yc, y1=next_yc, line=dict(color=lc, width=1))

    # Champion badge
    champ = bracket['Final'][0]['winner']
    champ_yc = yc(4, 0)
    fig.add_annotation(
        x=ROUND_X[-1] + BOX_W + 0.4, y=champ_yc,
        text=f'🏆 {FLAG_EMOJI.get(champ, "")} {champ}',
        xanchor='left', yanchor='middle',
        font=dict(size=13, color='#FFD700'),
        showarrow=False,
        bgcolor='#1a1a2e', bordercolor='#FFD700', borderpad=6,
    )

    fig.update_layout(
        height=1900,
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        xaxis=dict(visible=False, range=[-0.5, 29]),
        yaxis=dict(visible=False, range=[34, -1.5]),
        margin=dict(l=10, r=20, t=20, b=20),
        showlegend=False,
    )
    return fig


# ── Load data ────────────────────────────────────────────────────────────────
def load_predictions():
    df = pd.read_csv('wc2026_predictions.csv')
    df['confederation'] = df['team'].map(TEAM_CONF).fillna('TBD')
    df['flag'] = df['team'].map(FLAG_EMOJI).fillna('🏳️')
    df['display'] = df['flag'] + '  ' + df['team']
    df['conf_color'] = df['confederation'].map(CONF_COLORS)
    return df

@st.cache_data
def load_enriched():
    return pd.read_excel('fifa_wc2026_enriched.xlsx')

df = load_predictions()
enriched = load_enriched()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 2rem 2rem 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;">
    <h1 style="color: white; font-size: 2.4rem; margin: 0; font-weight: 800; letter-spacing: -1px;">
        ⚽ FIFA World Cup 2026 Predictor
    </h1>
    <p style="color: #a0aec0; margin: 0.4rem 0 0 0; font-size: 1rem;">
        Dixon-Coles Poisson model · 50,000 Monte Carlo simulations · 49,000 historical matches
    </p>
</div>
""", unsafe_allow_html=True)

# ── Top-level KPI bar ────────────────────────────────────────────────────────
top3 = df.nlargest(3, 'p_win_%')
k1, k2, k3, k4 = st.columns(4)
for col, (_, row) in zip([k1, k2, k3], top3.iterrows()):
    conf_c = CONF_COLORS.get(row['confederation'], '#888')
    col.markdown(f"""
    <div style="background:#1e1e2e; border-left: 4px solid {conf_c};
                padding:1rem 1.2rem; border-radius:8px;">
        <div style="font-size:1.6rem">{row['flag']}</div>
        <div style="font-size:1.1rem; font-weight:700; color:white">{row['team']}</div>
        <div style="font-size:2rem; font-weight:800; color:{conf_c}">{row['p_win_%']}%</div>
        <div style="color:#a0aec0; font-size:0.8rem">Win probability</div>
    </div>""", unsafe_allow_html=True)
k4.markdown(f"""
<div style="background:#1e1e2e; border-left: 4px solid #f6c90e;
            padding:1rem 1.2rem; border-radius:8px;">
    <div style="font-size:1.6rem">🎲</div>
    <div style="font-size:1.1rem; font-weight:700; color:white">Simulations</div>
    <div style="font-size:2rem; font-weight:800; color:#f6c90e">50,000</div>
    <div style="color:#a0aec0; font-size:0.8rem">Monte Carlo runs</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Win Probabilities", "🗂️ Groups", "🔍 Team Deep Dive", "📈 Head-to-Head", "🏆 Bracket"])

# ════════════════════════════════════════════════════════
# TAB 1 — Win Probabilities
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("Tournament Win Probability — All 48 Teams")

    filter_conf = st.multiselect(
        "Filter by confederation",
        options=sorted(df['confederation'].unique()),
        default=sorted(df['confederation'].unique()),
    )
    view_df = df[df['confederation'].isin(filter_conf)].copy()

    # Horizontal bar chart
    view_df_sorted = view_df.sort_values('p_win_%', ascending=True)
    fig = go.Figure()
    for conf in view_df_sorted['confederation'].unique():
        sub = view_df_sorted[view_df_sorted['confederation'] == conf]
        fig.add_trace(go.Bar(
            y=sub['display'],
            x=sub['p_win_%'],
            orientation='h',
            name=conf,
            marker_color=CONF_COLORS.get(conf, '#888'),
            text=sub['p_win_%'].apply(lambda x: f"{x}%" if x >= 0.5 else ""),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Win: %{x}%<extra></extra>',
        ))
    fig.update_layout(
        height=max(500, len(view_df_sorted) * 22),
        margin=dict(l=10, r=60, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Win Probability (%)', gridcolor='#2d2d2d', color='white'),
        yaxis=dict(color='white', tickfont=dict(size=11)),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
        font=dict(color='white'),
        barmode='stack',
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader("Stage-by-Stage Probability Funnel — Top 16 Teams")

    top16 = df.nlargest(16, 'p_win_%')
    stages = ['p_advance_%', 'p_r16_%', 'p_qf_%', 'p_sf_%', 'p_final_%', 'p_win_%']
    stage_labels = ['Group Stage', 'Round of 32', 'Quarter-final', 'Semi-final', 'Final', 'Winner']

    fig2 = go.Figure()
    for _, row in top16.iterrows():
        fig2.add_trace(go.Scatter(
            x=stage_labels,
            y=[row[s] for s in stages],
            mode='lines+markers',
            name=f"{row['flag']} {row['team']}",
            line=dict(color=CONF_COLORS.get(row['confederation'], '#888'), width=2),
            hovertemplate='<b>%{fullData.name}</b><br>%{x}: %{y}%<extra></extra>',
        ))
    fig2.update_layout(
        height=480,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='#2d2d2d', color='white'),
        yaxis=dict(title='Probability (%)', gridcolor='#2d2d2d', color='white'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white', size=10),
                    orientation='v', x=1.01),
        font=dict(color='white'),
        margin=dict(l=10, r=160, t=10, b=10),
    )
    st.plotly_chart(fig2, width='stretch')

# ════════════════════════════════════════════════════════
# TAB 2 — Groups
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("Group Stage Overview")

    cols = st.columns(3)
    for i, (grp, teams) in enumerate(GROUPS.items()):
        col = cols[i % 3]
        grp_df = df[df['team'].isin(teams)].sort_values('p_advance_%', ascending=False)
        with col:
            st.markdown(f"#### Group {grp}")
            for _, row in grp_df.iterrows():
                bar_w = int(row['p_advance_%'])
                conf_c = CONF_COLORS.get(row['confederation'], '#888')
                st.markdown(f"""
                <div style="background:#1e1e2e; border-radius:6px; padding:0.5rem 0.8rem;
                            margin-bottom:0.4rem; border-left:3px solid {conf_c};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="color:white; font-weight:600">{row['flag']} {row['team']}</span>
                        <span style="color:{conf_c}; font-weight:700">{row['p_advance_%']}%</span>
                    </div>
                    <div style="background:#2d2d2d; border-radius:3px; height:4px; margin-top:4px;">
                        <div style="background:{conf_c}; width:{bar_w}%; height:4px; border-radius:3px;"></div>
                    </div>
                    <div style="color:#a0aec0; font-size:0.72rem; margin-top:2px;">
                        Win tournament: {row['p_win_%']}%
                    </div>
                </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 3 — Team Deep Dive
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("Team Deep Dive")
    selected = st.selectbox(
        "Select a team",
        options=df.sort_values('p_win_%', ascending=False)['team'].tolist(),
        format_func=lambda t: f"{FLAG_EMOJI.get(t, '')}  {t}",
    )

    pred_row = df[df['team'] == selected].iloc[0]
    enr_row  = enriched[enriched['team_name'].str.lower() == selected.lower()]
    if enr_row.empty:
        # fuzzy match for name variants
        enr_row = enriched[enriched['team_name'].str.contains(selected.split()[0], case=False, na=False)]
    enr_row = enr_row.iloc[0] if not enr_row.empty else None

    conf_c = CONF_COLORS.get(pred_row['confederation'], '#888')

    # Probability cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, label, key in zip(
        [c1, c2, c3, c4, c5, c6],
        ['Advance', 'Round of 32', 'Quarter-final', 'Semi-final', 'Final', 'Win 🏆'],
        ['p_advance_%', 'p_r16_%', 'p_qf_%', 'p_sf_%', 'p_final_%', 'p_win_%']
    ):
        col.markdown(f"""
        <div style="background:#1e1e2e; border-top:3px solid {conf_c};
                    padding:0.8rem; border-radius:6px; text-align:center;">
            <div style="color:#a0aec0; font-size:0.75rem">{label}</div>
            <div style="color:white; font-size:1.6rem; font-weight:800">{pred_row[key]}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if enr_row is not None:
        left, right = st.columns(2)

        with left:
            st.markdown("##### Recent Form (2024–2026)")
            form_data = {
                'Win Rate':       enr_row.get('form_win_rate', np.nan),
                'Draw Rate':      enr_row.get('form_draw_rate', np.nan),
                'Goals For/Game': enr_row.get('form_goals_for_pg', np.nan),
                'Goals Ag/Game':  enr_row.get('form_goals_against_pg', np.nan),
                'Clean Sheet %':  enr_row.get('form_clean_sheet_pct', np.nan),
                'GD/Game':        enr_row.get('form_gd_pg', np.nan),
            }
            for label, val in form_data.items():
                if pd.notna(val):
                    st.metric(label, f"{val:.3f}" if val < 1 else f"{val:.2f}")

        with right:
            st.markdown("##### WC 2026 Qualifying")
            qual_data = {
                'Games Played':    enr_row.get('qual_games', np.nan),
                'Points/Game':     enr_row.get('qual_points_pg', np.nan),
                'Goals For/Game':  enr_row.get('qual_goals_for_pg', np.nan),
                'Goals Ag/Game':   enr_row.get('qual_goals_against_pg', np.nan),
                'Clean Sheet %':   enr_row.get('qual_clean_sheet_pct', np.nan),
                'GD/Game':         enr_row.get('qual_gd_pg', np.nan),
            }
            for label, val in qual_data.items():
                if pd.notna(val):
                    st.metric(label, f"{val:.0f}" if label == 'Games Played' else
                              (f"{val:.3f}" if val < 1 else f"{val:.2f}"))

        st.markdown("<br>", unsafe_allow_html=True)

        # Radar chart — squad attributes
        radar_cols = ['star_player_rating', 'goalkeeper_rating', 'squad_depth_score',
                      'h2h_vs_top10_winrate', 'knockout_stage_reach_rate']
        radar_labels = ['Star Player', 'Goalkeeper', 'Squad Depth', 'vs Top 10', 'KO Rate']
        vals = [enr_row.get(c, 0) for c in radar_cols]

        # Normalise to 0-10 scale where needed
        norm_vals = []
        for i, (c, v) in enumerate(zip(radar_cols, vals)):
            if pd.isna(v):
                norm_vals.append(0)
            elif v <= 1:       # rate/ratio → scale to 10
                norm_vals.append(float(v) * 10)
            else:
                norm_vals.append(float(v))

        # Convert hex to rgba for fill
        hex_c = conf_c.lstrip('#')
        r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
        fill_color = f'rgba({r},{g},{b},0.2)'

        fig3 = go.Figure(go.Scatterpolar(
            r=norm_vals + [norm_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill='toself',
            fillcolor=fill_color,
            line=dict(color=conf_c, width=2),
            marker=dict(size=6, color=conf_c),
        ))
        fig3.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 10], gridcolor='#2d2d2d',
                                tickcolor='white', color='white'),
                angularaxis=dict(color='white'),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=350,
            margin=dict(l=40, r=40, t=30, b=30),
            showlegend=False,
        )
        st.markdown(f"##### {FLAG_EMOJI.get(selected, '')} {selected} — Squad Profile")
        st.plotly_chart(fig3, width='stretch')

# ════════════════════════════════════════════════════════
# TAB 4 — Head to Head
# ════════════════════════════════════════════════════════
with tab4:
    st.subheader("Head-to-Head Matchup Simulator")
    st.caption("Uses Dixon-Coles expected goals to compute match win probabilities")

    col_a, col_b = st.columns(2)
    team_a = col_a.selectbox("Team A", df.sort_values('p_win_%', ascending=False)['team'].tolist(),
                              format_func=lambda t: f"{FLAG_EMOJI.get(t, '')}  {t}", index=0)
    team_b = col_b.selectbox("Team B", df.sort_values('p_win_%', ascending=False)['team'].tolist(),
                              format_func=lambda t: f"{FLAG_EMOJI.get(t, '')}  {t}", index=1)

    if team_a != team_b:
        # Simulate 10,000 matches quickly
        @st.cache_data
        def h2h_sim(ta, tb, n=10_000):
            s_df = pd.read_csv('wc2026_strengths.csv').set_index('team')
            atk  = s_df['attack'].to_dict()
            dfs  = s_df['defense'].to_dict()
            global_atk = np.mean(list(atk.values()))
            global_dfs = np.mean(list(dfs.values()))
            xg_a = np.exp(atk.get(ta, global_atk) - dfs.get(tb, global_dfs))
            xg_b = np.exp(atk.get(tb, global_atk) - dfs.get(ta, global_dfs))
            ga = np.random.poisson(xg_a, n)
            gb = np.random.poisson(xg_b, n)
            wins_a  = (ga > gb).mean()
            draws   = (ga == gb).mean()
            wins_b  = (ga < gb).mean()
            avg_ga  = ga.mean()
            avg_gb  = gb.mean()
            return wins_a, draws, wins_b, avg_ga, avg_gb, xg_a, xg_b

        with st.spinner("Simulating 10,000 matches..."):
            w_a, d, w_b, avg_a, avg_b, xg_a, xg_b = h2h_sim(team_a, team_b)

        conf_a = CONF_COLORS.get(TEAM_CONF.get(team_a, ''), '#3b82f6')
        conf_b = CONF_COLORS.get(TEAM_CONF.get(team_b, ''), '#ef4444')

        mc1, mc2, mc3 = st.columns(3)
        mc1.markdown(f"""
        <div style="background:#1e1e2e; border-top:4px solid {conf_a}; padding:1.2rem;
                    border-radius:8px; text-align:center;">
            <div style="font-size:2rem">{FLAG_EMOJI.get(team_a, '')}</div>
            <div style="color:white; font-weight:700">{team_a}</div>
            <div style="font-size:2.2rem; font-weight:800; color:{conf_a}">{w_a*100:.1f}%</div>
            <div style="color:#a0aec0; font-size:0.8rem">Win probability</div>
            <div style="color:#a0aec0; font-size:0.8rem">xG: {xg_a:.2f}</div>
        </div>""", unsafe_allow_html=True)
        mc2.markdown(f"""
        <div style="background:#1e1e2e; border-top:4px solid #a0aec0; padding:1.2rem;
                    border-radius:8px; text-align:center;">
            <div style="font-size:2rem">🤝</div>
            <div style="color:white; font-weight:700">Draw</div>
            <div style="font-size:2.2rem; font-weight:800; color:#a0aec0">{d*100:.1f}%</div>
            <div style="color:#a0aec0; font-size:0.8rem">Probability</div>
            <div style="color:#a0aec0; font-size:0.8rem">Avg: {avg_a:.1f} – {avg_b:.1f}</div>
        </div>""", unsafe_allow_html=True)
        mc3.markdown(f"""
        <div style="background:#1e1e2e; border-top:4px solid {conf_b}; padding:1.2rem;
                    border-radius:8px; text-align:center;">
            <div style="font-size:2rem">{FLAG_EMOJI.get(team_b, '')}</div>
            <div style="color:white; font-weight:700">{team_b}</div>
            <div style="font-size:2.2rem; font-weight:800; color:{conf_b}">{w_b*100:.1f}%</div>
            <div style="color:#a0aec0; font-size:0.8rem">Win probability</div>
            <div style="color:#a0aec0; font-size:0.8rem">xG: {xg_b:.2f}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Score probability heatmap
        st.markdown("##### Scoreline Probability Heatmap")
        max_goals = 6
        score_matrix = np.zeros((max_goals, max_goals))
        from scipy.stats import poisson
        for i in range(max_goals):
            for j in range(max_goals):
                score_matrix[i][j] = poisson.pmf(i, xg_a) * poisson.pmf(j, xg_b) * 100

        fig4 = go.Figure(go.Heatmap(
            z=score_matrix,
            x=[str(i) for i in range(max_goals)],
            y=[str(i) for i in range(max_goals)],
            colorscale='Blues',
            text=[[f"{score_matrix[i][j]:.1f}%" for j in range(max_goals)] for i in range(max_goals)],
            texttemplate="%{text}",
            hovertemplate=f"{team_a} %{{y}} – %{{x}} {team_b}<br>Prob: %{{z:.2f}}%<extra></extra>",
            showscale=False,
        ))
        fig4.update_layout(
            height=380,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title=f"{team_b} Goals", color='white', gridcolor='#2d2d2d'),
            yaxis=dict(title=f"{team_a} Goals", color='white', gridcolor='#2d2d2d'),
            font=dict(color='white'),
            margin=dict(l=60, r=20, t=20, b=60),
        )
        st.plotly_chart(fig4, width='stretch')
    else:
        st.warning("Select two different teams.")

# ════════════════════════════════════════════════════════
# TAB 5 — Bracket
# ════════════════════════════════════════════════════════
with tab5:
    st.subheader("🏆 Predicted Tournament Bracket")
    st.caption("Deterministic simulation using composite team strength — always advances the higher-rated team")

    bracket, group_standings = get_predicted_bracket()
    champ = bracket['Final'][0]['winner']

    st.markdown(f"""
    <div style="background:#1a1a2e; border:2px solid #FFD700; border-radius:10px;
                padding:1rem 1.5rem; margin-bottom:1.5rem; text-align:center;">
        <div style="color:#FFD700; font-size:1.6rem; font-weight:800; letter-spacing:-0.5px;">
            🏆 Predicted Champion: {FLAG_EMOJI.get(champ, '')} {champ}
        </div>
        <div style="color:#a0aec0; font-size:0.85rem; margin-top:4px">
            Finalist: {FLAG_EMOJI.get(bracket['Final'][0]['team_a'], '')} {bracket['Final'][0]['team_a']}
            vs {FLAG_EMOJI.get(bracket['Final'][0]['team_b'], '')} {bracket['Final'][0]['team_b']}
            &nbsp;·&nbsp; Win probability: {bracket['Final'][0]['prob_a']*100:.0f}% / {(1-bracket['Final'][0]['prob_a'])*100:.0f}%
        </div>
    </div>""", unsafe_allow_html=True)

    # Round-by-round match cards
    st.markdown("#### Match-by-Match View")
    sel_round = st.selectbox("Select round:",
        ['Round of 32', 'Round of 16', 'Quarter-Finals', 'Semi-Finals', 'Final'],
        index=1)
    round_key = {'Round of 32': 'R32', 'Round of 16': 'R16',
                 'Quarter-Finals': 'QF', 'Semi-Finals': 'SF', 'Final': 'Final'}[sel_round]

    matches = bracket[round_key]
    n_cols = min(4, len(matches))
    cols = st.columns(n_cols)
    for i, match in enumerate(matches):
        with cols[i % n_cols]:
            st.markdown(match_card_html(match), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Full Bracket")
    st.caption("Each box shows both teams with their head-to-head win probability. Winner highlighted. Scroll to explore all rounds.")
    fig_bracket = draw_bracket_fig(bracket)
    st.plotly_chart(fig_bracket, width='stretch')

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='color:#666; text-align:center; font-size:0.8rem;'>"
    "Dixon-Coles Poisson model · Time-decayed weights · Form & qualifying adjustments · "
    "50,000 Monte Carlo simulations · Data: martj42/international_results"
    "</div>",
    unsafe_allow_html=True,
)
