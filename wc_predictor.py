"""
FIFA World Cup 2026 Predictor
Dixon-Coles Poisson model + Monte Carlo tournament simulation
v4 fixes:
  - Sharper time decay — recent results dominate
  - Pre-tournament friendlies (Mar 2026+) treated as competitive
  - UEFA Nations League treated as competitive (not friendly)
  - FORM_BLEND raised so recent form has more impact
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
XI              = 0.006    # Sharper decay — half-life ~116 days (was 0.004)
FRIENDLY_WEIGHT = 0.35     # Standard friendlies weighted down
PRE_WC_FRIENDLY_WEIGHT = 0.90  # March 2026 prep friendlies nearly full weight
NATIONS_LEAGUE_WEIGHT  = 0.90  # UEFA/CONCACAF Nations League = competitive
SQUAD_WEIGHT    = 0.35     # FIFA squad OVA blend (slightly less dominant)
FORM_BLEND      = 0.30     # Raised from 0.20 — recent form matters more
N_SIMS          = 50_000
TRAIN_FROM      = '2022-01-01'
MIN_GAMES       = 12
REF_DATE        = pd.Timestamp('2026-06-11')

# Host nations get a boost from home crowd/travel advantage
HOSTS       = {'United States', 'Canada', 'Mexico'}
HOST_BOOST  = 0.14   # ~14% attack multiplier on log scale

# Confederation difficulty (how hard is it to get GD in this confederation)
CONF_DIFFICULTY = {
    'UEFA':     1.00,
    'CONMEBOL': 0.90,
    'CAF':      0.72,
    'AFC':      0.68,
    'CONCACAF': 0.62,
    'OFC':      0.42,
    'TBD':      0.60,
}

# ── FIFA World Rankings (April 1, 2026) ──────────────────────────────────────
# Lower rank = better team. Used to blend consensus into DC strengths.
FIFA_RANK_WEIGHT = 0.45   # 45% consensus, 55% DC+squad model

FIFA_RANKS = {
    'France':                   1,
    'Spain':                    2,
    'Argentina':                3,
    'England':                  4,
    'Portugal':                 5,
    'Brazil':                   6,
    'Netherlands':              7,
    'Morocco':                  8,
    'Belgium':                  9,
    'Germany':                  10,
    'Croatia':                  11,
    'Colombia':                 13,
    'Senegal':                  14,
    'Mexico':                   15,
    'United States':            16,
    'Uruguay':                  17,
    'Japan':                    18,
    'Switzerland':              19,
    'Iran':                     21,
    'Turkey':                   22,
    'Ecuador':                  23,
    'Austria':                  24,
    'South Korea':              25,
    'Australia':                27,
    'Algeria':                  28,
    'Egypt':                    29,
    'Canada':                   30,
    'Norway':                   31,
    'Panama':                   33,
    'Ivory Coast':              34,
    'Sweden':                   38,
    'Paraguay':                 40,
    'Czech Republic':           41,
    'Scotland':                 43,
    'Tunisia':                  44,
    'DR Congo':                 46,
    'Uzbekistan':               50,
    'Qatar':                    55,
    'Iraq':                     57,
    'South Africa':             60,
    'Saudi Arabia':             61,
    'Jordan':                   63,
    'Bosnia and Herzegovina':   65,
    'Cape Verde':               69,
    'Ghana':                    74,
    'Curacao':                  82,
    'Haiti':                    83,
    'New Zealand':              85,
}

TEAM_CONF = {
    'Argentina': 'CONMEBOL', 'Brazil': 'CONMEBOL', 'Colombia': 'CONMEBOL',
    'Ecuador': 'CONMEBOL', 'Uruguay': 'CONMEBOL', 'Paraguay': 'CONMEBOL',
    'Spain': 'UEFA', 'England': 'UEFA', 'France': 'UEFA', 'Germany': 'UEFA',
    'Portugal': 'UEFA', 'Netherlands': 'UEFA', 'Norway': 'UEFA',
    'Switzerland': 'UEFA', 'Croatia': 'UEFA', 'Belgium': 'UEFA',
    'Scotland': 'UEFA', 'Austria': 'UEFA', 'Sweden': 'UEFA',
    'Bosnia and Herzegovina': 'UEFA', 'Czech Republic': 'UEFA', 'Turkey': 'UEFA',
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

# FIFA_raw_data nationality → our team name
FIFA_NAT_MAP = {
    'Korea Republic':      'South Korea',
    'Bosnia Herzegovina':  'Bosnia and Herzegovina',
    'United States':       'United States',
    'DR Congo':            'DR Congo',
    'Ivory Coast':         'Ivory Coast',
}

# ── 2026 WC Groups ───────────────────────────────────────────────────────────
GROUPS = {
    'A': ['Mexico',        'South Korea',  'South Africa',       'Czech Republic'],
    'B': ['Canada',        'Switzerland',  'Qatar',              'Bosnia and Herzegovina'],
    'C': ['Brazil',        'Morocco',      'Haiti',              'Scotland'],
    'D': ['United States', 'Paraguay',     'Australia',          'Turkey'],
    'E': ['Germany',       'Curacao',      'Ivory Coast',        'Ecuador'],
    'F': ['Netherlands',   'Japan',        'Tunisia',            'Sweden'],
    'G': ['Belgium',       'Egypt',        'Iran',               'New Zealand'],
    'H': ['Spain',         'Cape Verde',   'Saudi Arabia',       'Uruguay'],
    'I': ['France',        'Senegal',      'Norway',             'Iraq'],
    'J': ['Argentina',     'Algeria',      'Austria',            'Jordan'],
    'K': ['Portugal',      'Uzbekistan',   'Colombia',           'DR Congo'],
    'L': ['England',       'Croatia',      'Ghana',              'Panama'],
}

R32_BRACKET_SLOTS = [
    ('1A', '2B'), ('1C', '2D'), ('1E', '2F'), ('1G', '2H'),
    ('1I', '2J'), ('1K', '2L'), ('1B', '2A'), ('1D', '2C'),
    ('1F', '2E'), ('1H', '2G'), ('1J', '2I'), ('1L', '2K'),
]


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_data():
    results = pd.read_csv(
        'international_results.csv', low_memory=False
    )
    results['date'] = pd.to_datetime(results['date'])

    # Normalise encodings
    def fix_team(col):
        results.loc[results[col].str.startswith('Cura', na=False) &
                    results[col].str.endswith('ao', na=False), col] = 'Curacao'
    fix_team('home_team')
    fix_team('away_team')

    teams_df = pd.read_excel('fifa_wc2026_enriched.xlsx')
    return results, teams_df


def compute_squad_ratings():
    """Return dict: team_name → avg OVA of top-23 players (pre-computed CSV)."""
    df = pd.read_csv('squad_ratings.csv')
    squad_ratings = dict(zip(df['team'], df['squad_ova']))
    # Fill missing WC teams with FIFA-ranking-based estimates
    squad_ratings.update({'Curacao': 63.0, 'Iraq': 66.0, 'Jordan': 65.5, 'Qatar': 67.0})
    return squad_ratings


# ── Dixon-Coles Strength Estimation ──────────────────────────────────────────
def estimate_strengths(results, ref_date=REF_DATE):
    df = results[results['date'] >= TRAIN_FROM].copy()
    df = df[df['home_score'].notna() & df['away_score'].notna()]
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)

    days_ago = (ref_date - df['date']).dt.days.clip(lower=0)
    df['w'] = np.exp(-XI * days_ago)

    # Tournament-specific weights
    is_friendly    = df['tournament'] == 'Friendly'
    is_pre_wc      = is_friendly & (df['date'] >= '2026-03-01')
    is_std_friendly = is_friendly & ~is_pre_wc
    is_nations_lg  = df['tournament'].str.contains('Nations League', na=False)

    df.loc[is_std_friendly,  'w'] *= FRIENDLY_WEIGHT
    df.loc[is_pre_wc,        'w'] *= PRE_WC_FRIENDLY_WEIGHT
    df.loc[is_nations_lg,    'w'] *= NATIONS_LEAGUE_WEIGHT

    # Count games per team and only keep teams with >= MIN_GAMES
    game_counts = pd.concat([df['home_team'], df['away_team']]).value_counts()
    qualified = set(game_counts[game_counts >= MIN_GAMES].index)
    df = df[df['home_team'].isin(qualified) & df['away_team'].isin(qualified)]

    # Make sure all 48 WC teams are represented — if a WC team got filtered,
    # lower the bar for them specifically
    wc_teams = {t for teams in GROUPS.values() for t in teams}
    missing = wc_teams - qualified
    if missing:
        for t in missing:
            min_t = game_counts.get(t, 0)
            print(f"  WARNING: {t} only has {min_t} games, including anyway")
        df2 = results[results['date'] >= TRAIN_FROM].copy()
        df2 = df2[df2['home_score'].notna() & df2['away_score'].notna()]
        df2['home_score'] = df2['home_score'].astype(int)
        df2['away_score'] = df2['away_score'].astype(int)
        days_ago2 = (ref_date - df2['date']).dt.days.clip(lower=0)
        df2['w'] = np.exp(-XI * days_ago2)
        is_f2     = df2['tournament'] == 'Friendly'
        is_pre2   = is_f2 & (df2['date'] >= '2026-03-01')
        df2.loc[is_f2 & ~is_pre2, 'w'] *= FRIENDLY_WEIGHT
        df2.loc[is_pre2,          'w'] *= PRE_WC_FRIENDLY_WEIGHT
        df2.loc[df2['tournament'].str.contains('Nations League', na=False), 'w'] *= NATIONS_LEAGUE_WEIGHT
        extra = df2[
            (df2['home_team'].isin(missing) | df2['away_team'].isin(missing)) &
            (~df2.index.isin(df.index))
        ]
        df = pd.concat([df, extra], ignore_index=True)
        qualified = qualified | missing

    all_teams = sorted(qualified)
    idx = {t: i for i, t in enumerate(all_teams)}
    n = len(all_teams)

    hi = df['home_team'].map(idx).dropna().astype(int)
    valid = df['home_team'].isin(idx) & df['away_team'].isin(idx)
    df = df[valid].copy()
    hi = df['home_team'].map(idx).astype(int).values
    ai = df['away_team'].map(idx).astype(int).values
    hg = df['home_score'].values.astype(float)
    ag = df['away_score'].values.astype(float)
    wt = df['w'].values
    nt = df['neutral'].astype(int).values

    def neg_log_likelihood(params):
        atk  = params[:n]
        dfs  = params[n:2*n]
        home = params[2*n]
        lam_h = np.exp(atk[hi] - dfs[ai] + home * (1 - nt))
        lam_a = np.exp(atk[ai] - dfs[hi])
        ll = wt * (
            hg * np.log(lam_h + 1e-9) - lam_h +
            ag * np.log(lam_a + 1e-9) - lam_a
        )
        return -ll.sum()

    x0 = np.zeros(2 * n + 1)
    x0[2*n] = 0.2
    print(f"  Estimating strengths for {n} teams (min {MIN_GAMES} games) from {len(df):,} matches...")
    res = minimize(neg_log_likelihood, x0, method='L-BFGS-B',
                   options={'maxiter': 2000, 'ftol': 1e-9})

    atk_raw = res.x[:n]
    dfs_raw = res.x[n:2*n]
    home_adv = res.x[2*n]
    atk_raw -= atk_raw.mean()
    dfs_raw -= dfs_raw.mean()

    attack  = {t: atk_raw[idx[t]] for t in all_teams}
    defense = {t: dfs_raw[idx[t]] for t in all_teams}
    print(f"  Home advantage: {np.exp(home_adv):.3f}x")
    return attack, defense, home_adv


# ── Squad Rating Blending ────────────────────────────────────────────────────
def blend_squad_ratings(attack, defense, squad_ratings):
    """
    Blend FIFA squad OVA into DC strengths.
    Higher squad OVA → nudge attack up, defense up (harder to score on).
    In this DC formulation: lam = exp(atk_a - dfs_b), so higher dfs = fewer goals conceded.
    """
    wc_teams = {t for teams in GROUPS.values() for t in teams}

    # Normalise squad ratings to the same scale as DC attack (zero-mean)
    sr_vals = [squad_ratings[t] for t in wc_teams if t in squad_ratings]
    sr_mean = np.mean(sr_vals)
    sr_std  = max(np.std(sr_vals), 0.1)

    dc_atk_vals = [attack[t] for t in wc_teams if t in attack]
    dc_std = max(np.std(dc_atk_vals), 0.01)

    atk_blended = dict(attack)
    dfs_blended = dict(defense)

    for team in wc_teams:
        if team not in attack:
            continue
        if team in squad_ratings:
            z_squad = (squad_ratings[team] - sr_mean) / sr_std
            # Scale squad z-score to DC attack scale
            squad_signal = z_squad * dc_std
        else:
            squad_signal = 0.0

        # Weighted blend: (1-w)*dc + w*squad_signal
        w = SQUAD_WEIGHT
        atk_blended[team] = (1 - w) * attack[team] + w * squad_signal
        dfs_blended[team] = (1 - w) * defense[team] + w * (squad_signal * 0.6)

    return atk_blended, dfs_blended


# ── Form Adjustment Layer ────────────────────────────────────────────────────
def apply_form_adjustment(attack, defense, teams_df):
    """
    Nudge strengths using recent form and qualifying, weighted by confederation difficulty.
    """
    form_mean = teams_df['form_gd_pg'].mean()
    form_std  = max(teams_df['form_gd_pg'].std(), 0.01)
    qual_mean = teams_df['qual_gd_pg'].dropna().mean()
    qual_std  = max(teams_df['qual_gd_pg'].dropna().std(), 0.01)

    atk_adj = dict(attack)
    dfs_adj = dict(defense)

    name_map = {'Curacaio': 'Curacao'}

    for _, row in teams_df.iterrows():
        dc_name = name_map.get(row['team_name'], row['team_name'])
        if dc_name not in attack:
            continue

        conf = TEAM_CONF.get(dc_name, 'TBD')
        diff = CONF_DIFFICULTY.get(conf, 0.70)

        z_form = 0.0
        if pd.notna(row.get('form_gd_pg')):
            z_form = (row['form_gd_pg'] - form_mean) / form_std

        z_qual = 0.0
        if pd.notna(row.get('qual_gd_pg')):
            # Scale by confederation difficulty so OFC qualification ≠ UEFA
            raw_z = (row['qual_gd_pg'] - qual_mean) / qual_std
            z_qual = raw_z * diff

        z_combined = 0.55 * z_form + 0.45 * z_qual
        nudge = FORM_BLEND * z_combined * 0.10

        # Host nation boost
        if dc_name in HOSTS:
            nudge += HOST_BOOST * 0.10

        atk_adj[dc_name] = attack[dc_name] + nudge
        dfs_adj[dc_name] = defense[dc_name] - nudge

    return atk_adj, dfs_adj


# ── FIFA Ranking Blend ───────────────────────────────────────────────────────
def apply_fifa_rank_blend(attack, defense, fifa_ranks):
    """
    Blend April 2026 FIFA rankings into DC strengths.
    Lower rank = better team = higher signal. Pulls model toward current consensus.
    FIFA_RANK_WEIGHT controls how much consensus overrides historical data.
    """
    wc_teams = {t for teams in GROUPS.values() for t in teams}

    rank_vals = [-fifa_ranks[t] for t in wc_teams if t in fifa_ranks]
    rank_mean = np.mean(rank_vals)
    rank_std  = max(np.std(rank_vals), 0.1)

    dc_atk_vals = [attack[t] for t in wc_teams if t in attack]
    dc_std = max(np.std(dc_atk_vals), 0.01)

    atk_out = dict(attack)
    dfs_out = dict(defense)

    for team in wc_teams:
        if team not in attack or team not in fifa_ranks:
            continue
        z_rank = (-fifa_ranks[team] - rank_mean) / rank_std
        rank_signal = z_rank * dc_std

        w = FIFA_RANK_WEIGHT
        atk_out[team] = (1 - w) * attack[team] + w * rank_signal
        dfs_out[team] = (1 - w) * defense[team] + w * (rank_signal * 0.6)

    return atk_out, dfs_out


# ── Match Simulation ─────────────────────────────────────────────────────────
def expected_goals(team_a, team_b, attack, defense):
    global_atk = np.mean(list(attack.values()))
    global_dfs = np.mean(list(defense.values()))
    atk_a = attack.get(team_a, global_atk)
    dfs_a = defense.get(team_a, global_dfs)
    atk_b = attack.get(team_b, global_atk)
    dfs_b = defense.get(team_b, global_dfs)
    xg_a = np.exp(atk_a - dfs_b)
    xg_b = np.exp(atk_b - dfs_a)
    return xg_a, xg_b


def simulate_match(team_a, team_b, attack, defense, allow_draw=True):
    xg_a, xg_b = expected_goals(team_a, team_b, attack, defense)
    ga = np.random.poisson(xg_a)
    gb = np.random.poisson(xg_b)
    if not allow_draw and ga == gb:
        ga += np.random.poisson(xg_a * 0.33)
        gb += np.random.poisson(xg_b * 0.33)
        if ga == gb:
            p_a = np.exp(attack.get(team_a, 0)) / (
                np.exp(attack.get(team_a, 0)) + np.exp(attack.get(team_b, 0))
            )
            ga += int(np.random.random() < p_a)
            gb += int(np.random.random() >= p_a)
    return ga, gb


# ── Group Stage ───────────────────────────────────────────────────────────────
def simulate_group_stage(attack, defense):
    group_tables = {}
    for grp, teams in GROUPS.items():
        records = {t: {'pts': 0, 'gd': 0, 'gf': 0} for t in teams}
        pairs = [(teams[i], teams[j]) for i in range(4) for j in range(i+1, 4)]
        for ta, tb in pairs:
            ga, gb = simulate_match(ta, tb, attack, defense, allow_draw=True)
            records[ta]['gf'] += ga; records[tb]['gf'] += gb
            records[ta]['gd'] += ga - gb; records[tb]['gd'] += gb - ga
            if ga > gb:   records[ta]['pts'] += 3
            elif ga == gb: records[ta]['pts'] += 1; records[tb]['pts'] += 1
            else:          records[tb]['pts'] += 3
        table = pd.DataFrame(records).T.reset_index()
        table.columns = ['team', 'pts', 'gd', 'gf']
        table['rand'] = np.random.random(len(table))
        table = table.sort_values(['pts', 'gd', 'gf', 'rand'], ascending=False).reset_index(drop=True)
        table['position'] = range(1, 5)
        table['group'] = grp
        group_tables[grp] = table
    return group_tables


def get_qualifiers(group_tables):
    winners, runners, thirds = [], [], []
    for grp, table in group_tables.items():
        winners.append({'team': table.iloc[0]['team'], 'slot': f'1{grp}',
                        'pts': table.iloc[0]['pts'], 'gd': table.iloc[0]['gd'], 'gf': table.iloc[0]['gf']})
        runners.append({'team': table.iloc[1]['team'], 'slot': f'2{grp}',
                        'pts': table.iloc[1]['pts'], 'gd': table.iloc[1]['gd'], 'gf': table.iloc[1]['gf']})
        thirds.append( {'team': table.iloc[2]['team'], 'slot': f'3{grp}',
                        'pts': table.iloc[2]['pts'], 'gd': table.iloc[2]['gd'], 'gf': table.iloc[2]['gf']})
    thirds_df = pd.DataFrame(thirds)
    thirds_df['rand'] = np.random.random(len(thirds_df))
    thirds_df = thirds_df.sort_values(['pts', 'gd', 'gf', 'rand'], ascending=False)
    best_thirds = thirds_df.head(8)['team'].tolist()
    slot_map = {w['slot']: w['team'] for w in winners}
    slot_map.update({r['slot']: r['team'] for r in runners})
    return slot_map, best_thirds


# ── Knockout Stage ────────────────────────────────────────────────────────────
def simulate_knockout(slot_map, best_thirds, attack, defense):
    r32_teams = [(slot_map.get(a, a), slot_map.get(b, b)) for a, b in R32_BRACKET_SLOTS]
    for i in range(0, 8, 2):
        r32_teams.append((best_thirds[i], best_thirds[i+1]))

    def play_round(matchups):
        winners = []
        for ta, tb in matchups:
            ga, gb = simulate_match(ta, tb, attack, defense, allow_draw=False)
            winners.append(ta if ga > gb else tb)
        return winners

    r16_flat = play_round(r32_teams)
    qf_flat  = play_round([(r16_flat[i], r16_flat[i+1]) for i in range(0, 16, 2)])
    sf_flat  = play_round([(qf_flat[i],  qf_flat[i+1])  for i in range(0, 8, 2)])
    fin_flat = play_round([(sf_flat[i],  sf_flat[i+1])  for i in range(0, 4, 2)])
    winner   = play_round([tuple(fin_flat)])[0]

    return {
        'r16':    set(r16_flat),
        'qf':     set(qf_flat),
        'sf':     set(sf_flat),
        'final':  set(fin_flat),
        'winner': winner,
    }


# ── Monte Carlo Engine ────────────────────────────────────────────────────────
def run_monte_carlo(attack, defense, n_sims=N_SIMS):
    all_teams = [t for teams in GROUPS.values() for t in teams]
    counters = {t: defaultdict(int) for t in all_teams}
    print(f"  Running {n_sims:,} simulations...")
    for i in range(n_sims):
        if (i + 1) % 10_000 == 0:
            print(f"    {i+1:,} / {n_sims:,}")
        group_tables = simulate_group_stage(attack, defense)
        slot_map, best_thirds = get_qualifiers(group_tables)
        qualified = set(slot_map.values()) | set(best_thirds)
        for t in all_teams:
            if t in qualified:
                counters[t]['group_adv'] += 1
        ko = simulate_knockout(slot_map, best_thirds, attack, defense)
        for stage in ['r16', 'qf', 'sf', 'final']:
            for t in ko[stage]:
                counters[t][stage] += 1
        counters[ko['winner']]['winner'] += 1

    rows = []
    for t in all_teams:
        grp = next(g for g, ts in GROUPS.items() if t in ts)
        c = counters[t]
        rows.append({
            'team':        t,
            'group':       grp,
            'p_advance_%': round(100 * c['group_adv'] / n_sims, 1),
            'p_r16_%':     round(100 * c['r16']       / n_sims, 1),
            'p_qf_%':      round(100 * c['qf']        / n_sims, 1),
            'p_sf_%':      round(100 * c['sf']        / n_sims, 1),
            'p_final_%':   round(100 * c['final']     / n_sims, 1),
            'p_win_%':     round(100 * c['winner']    / n_sims, 1),
        })
    return pd.DataFrame(rows).sort_values('p_win_%', ascending=False).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("FIFA World Cup 2026 Predictor v4")
    print("=" * 50)

    print("\n[1/5] Loading data...")
    results, teams_df = load_data()

    print("\n[2/5] Computing FIFA squad ratings...")
    squad_ratings = compute_squad_ratings()
    print(f"  Loaded ratings for {len(squad_ratings)} nations")
    for t in ['France','Spain','Argentina','Brazil','Colombia','United States','Germany','England']:
        print(f"  {t:25} OVA: {squad_ratings.get(t, 0):.1f}")

    print("\n[3/5] Estimating Dixon-Coles team strengths...")
    attack, defense, home_adv = estimate_strengths(results)

    print("\n[4/5] Blending squad ratings + form adjustments + FIFA rankings...")
    attack, defense = blend_squad_ratings(attack, defense, squad_ratings)
    attack, defense = apply_form_adjustment(attack, defense, teams_df)
    attack, defense = apply_fifa_rank_blend(attack, defense, FIFA_RANKS)

    # Save attack strengths for app.py bracket
    wc_teams = [t for teams in GROUPS.values() for t in teams]
    strengths_df = pd.DataFrame([
        {'team': t, 'attack': round(attack.get(t, 0), 4), 'defense': round(defense.get(t, 0), 4)}
        for t in wc_teams
    ])
    strengths_df.to_csv('wc2026_strengths.csv', index=False)

    top10_atk = sorted([(t, attack[t]) for t in wc_teams if t in attack], key=lambda x: -x[1])[:10]
    print("  Top 10 attack (WC teams only):", [(t, round(v,3)) for t, v in top10_atk])

    print("\n[5/5] Running Monte Carlo simulation...")
    np.random.seed(42)
    results_df = run_monte_carlo(attack, defense, n_sims=N_SIMS)

    print("\n" + "=" * 70)
    print("FIFA WORLD CUP 2026 — WIN PROBABILITIES (v2)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    out = 'wc2026_predictions.csv'
    results_df.to_csv(out, index=False)
    print(f"\nSaved to: {out}")
