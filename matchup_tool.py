import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Matchup Priorities", layout="wide")

file_path = "Four Factors by Team and Game.xlsx"
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

# Main stat maps
counterpart_map = {
    'OREB': 'DREB', 'DREB': 'OREB',
    'FTRate': 'OppFTRate', 'OppFTRate': 'FTRate',
    'TOVRate': 'OppTOVRate', 'OppTOVRate': 'TOVRate',
    'oQSQ': 'dQSQ', 'dQSQ': 'oQSQ',

    # Neutral stat flips (updated to match stripped column names)
    '3PARate': 'Opp3PARate',
    'Opp3PARate': '3PARate',
    'AvgOffPace': 'AvgDefPace',
    'AvgDefPace': 'AvgOffPace'
}

readable_labels = {
    'OREB': 'Offensive Rebounding',
    'DREB': 'Defensive Rebounding',
    'FTRate': 'Free Throw Rate',
    'OppFTRate': 'Opponent Free Throw Rate',
    'TOVRate': 'Turnovers',
    'OppTOVRate': 'Opponent Turnovers',
    'oQSQ': 'Offensive Shot Quality',
    'dQSQ': 'Defensive Shot Quality',
    '3PARate': '3PA Rate',
    'Opp3PARate': '3PA Rate Allowed',
    'AvgOffPace': 'Avg Off Pace',
    'AvgDefPace': 'Avg Def Pace'
}

# Key stat categories
positive_stats = ["oQSQ", "DREB", "FTRate", "OREB", "OppTOVRate"]
negative_stats = ["dQSQ", "TOVRate", "OppFTRate"]
neutral_stats = ["3PARate", "Opp3PARate", "AvgOffPace", "AvgDefPace"]

predictors = list(counterpart_map.keys()) + neutral_stats

# Regression-based importance
importance_signed = {}
for team, group in df.groupby("Team"):
    existing_predictors = [col for col in predictors if col in group.columns]
    X = group[existing_predictors].dropna()
    y = group.loc[X.index, 'NETRTG']
    if len(X) < len(predictors):
        continue
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression().fit(X_scaled, y)

    direction_map = {stat: 1 for stat in positive_stats}
    direction_map.update({stat: -1 for stat in negative_stats})
    adjusted_coefs = {
        stat: coef * direction_map.get(stat, 1)
        for stat, coef in zip(existing_predictors, model.coef_)
    }

    importance_signed[team] = pd.Series(adjusted_coefs)

importance_df = pd.DataFrame(importance_signed).T.fillna(0)
existing_predictors = [col for col in predictors if col in df.columns]
variance_df = df.groupby("Team")[existing_predictors].var().fillna(0)
predictors = existing_predictors

priority_product = importance_df * variance_df
priority_weighted = 0.7 * importance_df + 0.3 * variance_df
priority_power = (importance_df * variance_df) ** 1.5

def statwise_scale(df):
    scaled_df = df.copy()
    for col in scaled_df.columns:
        scaler = MinMaxScaler(feature_range=(1, 100))
        scaled_df[col] = scaler.fit_transform(scaled_df[[col]])
    return scaled_df

scaled_product = statwise_scale(priority_product)
scaled_weighted = statwise_scale(priority_weighted)
scaled_power = statwise_scale(priority_power)

# UI
with st.sidebar:
    st.header("How This Works")
    st.markdown("""
    This tool identifies the most important factors for team success based on this season's data.

    **Priority Score = (Importance × 0.7 + Variability × 0.3)**

    This is done for both your team and opponent (counterpart stat).
    Then they are multiplied together and scaled 1–100.

    Neutral stats like Pace or 3PA Rate are treated separately.
    """)

st.title("Matchup-Based Coaching Priorities")

teams = sorted(scaled_weighted.index)
team = st.selectbox("Select Your Team", teams, index=teams.index("CLE") if "CLE" in teams else 0)
opponent_options = ["All Teams", "Top 5 Teams", "Top 10 Teams", "Top 16 Teams"] + [t for t in teams if t != team]
opponent = st.selectbox("Select Opponent", opponent_options)

with st.expander("Advanced Settings: Priority Method", expanded=False):
    method = st.radio("Choose Method for Calculating Priority Scores", ["Product (x)", "Weighted Average", "Powered Product^1.5"], index=1)

if method == "Product (x)":
    selected_priority = scaled_product
elif method == "Weighted Average":
    selected_priority = scaled_weighted
else:
    selected_priority = scaled_power

team_scores = selected_priority.loc[team]

matchup_scores = {}
for stat, counterpart in counterpart_map.items():
    if opponent == "All Teams":
        matchup_scores[stat] = team_scores[stat]
    elif opponent in ["Top 5 Teams", "Top 10 Teams", "Top 16 Teams"]:
        subset_map = {
            "Top 5 Teams": df.groupby("Team")["NETRTG"].mean().sort_values(ascending=False).head(5).index,
            "Top 10 Teams": df.groupby("Team")["NETRTG"].mean().sort_values(ascending=False).head(10).index,
            "Top 16 Teams": df.groupby("Team")["NETRTG"].mean().sort_values(ascending=False).head(16).index,
        }
        opponents = subset_map[opponent]
        avg_counterpart_score = selected_priority.loc[opponents][counterpart].mean()
        matchup_scores[stat] = team_scores[stat] * avg_counterpart_score
    elif stat in team_scores and counterpart in selected_priority.loc[opponent]:
        matchup_scores[stat] = team_scores[stat] * selected_priority.loc[opponent][counterpart]

matchup_series = pd.Series(matchup_scores)
scaler = MinMaxScaler(feature_range=(1, 100))
if not matchup_series.empty:
    scaled = scaler.fit_transform(matchup_series.values.reshape(-1, 1)).flatten()
else:
    scaled = [1] * len(matchup_series)

matchup_df = pd.DataFrame({
    "Variable": matchup_series.index.map(readable_labels),
    "Matchup Priority Score": scaled.round(0).astype(int),
}).sort_values(by="Matchup Priority Score", ascending=False).reset_index(drop=True)
matchup_df.index += 1
matchup_df.index.name = "Rank"

styled_df = matchup_df.style.background_gradient(cmap="Greens", subset=["Matchup Priority Score"])
st.dataframe(styled_df, use_container_width=True)


# Scale neutral stats’ importance across teams (absolute value)
neutral_importance = importance_df[neutral_stats].abs()
scaler = MinMaxScaler(feature_range=(1, 100))
scaled_neutral_importance = pd.DataFrame(
    scaler.fit_transform(neutral_importance),
    index=neutral_importance.index,
    columns=neutral_importance.columns
)

# Build the neutral stat table for the selected team
neutral_data = []
for stat in neutral_stats:
    imp = scaled_neutral_importance.loc[team, stat]
    raw_imp = importance_df.loc[team, stat]
    if stat == "AvgOffPace":
        direction = "Slower" if raw_imp > 0 else "Faster"
        label = "Pace"
    elif stat == "AvgDefPace":
        direction = "Slower" if raw_imp > 0 else "Faster"
        label = "Opp Pace"
    elif stat == "3PA Rate":
        direction = "More" if raw_imp > 0 else "Less"
        label = "Threes"
    elif stat == "3PA Rate Allowed":
        direction = "More" if raw_imp > 0 else "Less"
        label = "Opp Threes"
    else:
        label = readable_labels.get(stat, stat)
    neutral_data.append({"Category": label, "Better": direction, "Importance": round(imp)})

neutral_df = pd.DataFrame(neutral_data).sort_values(by="Importance", ascending=False).reset_index(drop=True)
st.subheader("Neutral Stat Tendencies")
styled_neutral_df = neutral_df.style.background_gradient(cmap="Greens", subset=["Importance"])
st.dataframe(styled_neutral_df, use_container_width=True)

# Stat breakdowns
label_to_stat = {v: k for k, v in readable_labels.items()}
readable_options = list(label_to_stat.keys())
if "selected_stat" not in st.session_state:
    st.session_state["selected_stat"] = readable_options[0]

selected_label = st.selectbox("Select a stat to view team performance tiers", readable_options, index=readable_options.index(st.session_state["selected_stat"]))
st.session_state["selected_stat"] = selected_label
selected_stat = label_to_stat[selected_label]
stat_counterpart = counterpart_map.get(selected_stat, selected_stat)

def stat_by_tier(df, team, stat):
    team_df = df[df["Team"] == team].dropna(subset=["NETRTG", stat])
    team_df = team_df.sort_values("NETRTG", ascending=False)
    n = len(team_df)
    if n < 3:
        return pd.DataFrame(columns=["Game Tier", "Value", "Rank"])
    tiers = {
        "Best Games": team_df.iloc[:n // 3],
        "Average Games": team_df.iloc[n // 3:2 * n // 3],
        "Worst Games": team_df.iloc[2 * n // 3:]
    }
    records = []
    for tier_name, tier_df in tiers.items():
        avg_val = tier_df[stat].mean()
        tier_group = []
        for other_team in df["Team"].unique():
            group = df[df["Team"] == other_team].sort_values("NETRTG", ascending=False)
            m = len(group)
            if m < 3:
                continue
            segment = {
                "Best Games": group.iloc[:m // 3],
                "Average Games": group.iloc[m // 3:2 * m // 3],
                "Worst Games": group.iloc[2 * m // 3:]
            }[tier_name]
            tier_group.append(segment[stat].mean())
        rank = pd.Series(tier_group + [avg_val]).rank(ascending=False, method="min").iloc[-1]
        def ordinal(n): return "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1)*(n % 10 < 4)*n % 10::4])
        rank_str = ordinal(int(rank))
        if stat in ["oQSQ", "dQSQ", "AvgOffPace", "AvgDefPace"]:
            value_str = f"{avg_val:.1f}"
        else:
            value_str = f"{avg_val * 100:.1f}%"
        records.append({"Game Tier": tier_name, "Value": value_str, "Rank": rank_str})
    return pd.DataFrame(records)

# Display tables
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"{team} — {readable_labels.get(selected_stat, selected_stat)}")
    df_team_stat = stat_by_tier(df, team, selected_stat)
    st.dataframe(df_team_stat.set_index("Game Tier"), use_container_width=True)

with col2:
    if opponent in ["Top 5 Teams", "Top 10 Teams", "Top 16 Teams"]:
        subset_map = {
            "Top 5 Teams": df.groupby("Team")["NETRTG"].mean().sort_values(ascending=False).head(5).index.tolist(),
            "Top 10 Teams": df.groupby("Team")["NETRTG"].mean().sort_values(ascending=False).head(10).index.tolist(),
            "Top 16 Teams": df.groupby("Team")["NETRTG"].mean().sort_values(ascending=False).head(16).index.tolist(),
        }
        subset_teams = subset_map[opponent]
        st.subheader(f"{opponent} Avg — {readable_labels.get(stat_counterpart, stat_counterpart)}")
        avg_df = pd.concat([stat_by_tier(df, opp, stat_counterpart) for opp in subset_teams])
        avg_df["Value"] = avg_df["Value"].astype(str).str.replace('%', '').astype(float)
        tier_means = avg_df.groupby("Game Tier").agg({"Value": "mean"}).reset_index()
        tier_means["Value"] = tier_means["Value"].apply(lambda x: f"{x:.1f}%" if selected_stat not in ["oQSQ", "dQSQ", "AvgOffPace", "AvgDefPace"] else f"{x:.1f}")
        tier_means["Rank"] = "–"
        st.dataframe(tier_means, use_container_width=True)
    else:
        st.subheader(f"{opponent} — {readable_labels.get(stat_counterpart, stat_counterpart)}")
        df_opp_stat = stat_by_tier(df, opponent, stat_counterpart)
        st.dataframe(df_opp_stat.set_index("Game Tier"), use_container_width=True)
