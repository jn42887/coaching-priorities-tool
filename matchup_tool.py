import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Matchup Priorities", layout="wide")

file_path = "Four Factors by Team and Game.xlsx"
df = pd.read_excel(file_path)
df.columns = df.columns.str.replace(" ", "")  # Remove all spaces

# Full stat map (8 core + 4 neutral)
counterpart_map = {
    'OREB': 'DREB', 'DREB': 'OREB',
    'FTRate': 'OppFTRate', 'OppFTRate': 'FTRate',
    'TOVRate': 'OppTOVRate', 'OppTOVRate': 'TOVRate',
    'oQSQ': 'dQSQ', 'dQSQ': 'oQSQ',
    '3PARate': '3PARateAllowed', '3PARateAllowed': '3PARate',
    'AvgOffPace': 'AvgDefPace', 'AvgDefPace': 'AvgOffPace'
}

# Labels
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
    '3PARateAllowed': '3PA Rate Allowed',
    'AvgOffPace': 'Avg Off Pace',
    'AvgDefPace': 'Avg Def Pace'
}

positive_stats = ["oQSQ", "DREB", "FTRate", "OREB", "OppTOVRate"]
negative_stats = ["dQSQ", "TOVRate", "OppFTRate"]
neutral_stats = ["3PARate", "3PARateAllowed", "AvgOffPace", "AvgDefPace"]
predictors = list(counterpart_map.keys())

stat_type_map = {
    'OREB': 'Offense', 'FTRate': 'Offense', 'TOVRate': 'Offense', 'oQSQ': 'Offense',
    'DREB': 'Defense', 'OppFTRate': 'Defense', 'OppTOVRate': 'Defense', 'dQSQ': 'Defense',
    '3PARate': 'Offense', '3PARateAllowed': 'Defense', 'AvgOffPace': 'Offense', 'AvgDefPace': 'Defense'
}

# Importance calculation
importance_signed = {}
for team, group in df.groupby("Team"):
    X = group[predictors + neutral_stats].dropna()
    y = group.loc[X.index, 'NETRTG']
    X_scaled = StandardScaler().fit_transform(X)
    model = LinearRegression().fit(X_scaled, y)
    direction_map = {stat: 1 for stat in positive_stats}
    direction_map.update({stat: -1 for stat in negative_stats})
    adjusted_coefs = {
        stat: coef * direction_map.get(stat, 1)
        for stat, coef in zip(predictors + neutral_stats, model.coef_)
    }
    importance_signed[team] = pd.Series(adjusted_coefs)

importance_df = pd.DataFrame(importance_signed).T.fillna(0)
variance_df = df.groupby("Team")[predictors + neutral_stats].var().fillna(0)

priority_weighted = 0.7 * importance_df + 0.3 * variance_df
scaled_weighted = priority_weighted.copy()
for col in scaled_weighted.columns:
    scaled_weighted[col] = MinMaxScaler((1, 100)).fit_transform(scaled_weighted[[col]])

# UI
with st.sidebar:
    st.header("How This Works")
    st.markdown("""
    **Priority Score = (Importance × 0.7 + Variability × 0.3)**  
    All stats (including pace and 3PA rate) are included.  
    Scores are multiplied against opponent tendencies.

    🟩 Green = Offense | 🟦 Blue = Defense | 🟪 Purple = Stylistic
    """)

st.title("Matchup-Based Coaching Priorities")

teams = sorted(scaled_weighted.index)
team = st.selectbox("Select Your Team", teams, index=teams.index("CLE") if "CLE" in teams else 0)
opponent_options = ["All Teams", "Top 5 Teams", "Top 10 Teams", "Top 16 Teams"] + [t for t in teams if t != team]
opponent = st.selectbox("Select Opponent", opponent_options)

team_scores = scaled_weighted.loc[team]
matchup_scores = {}
for stat, counterpart in counterpart_map.items():
    if opponent == "All Teams":
        matchup_scores[stat] = team_scores[stat]
    elif opponent in ["Top 5 Teams", "Top 10 Teams", "Top 16 Teams"]:
        subset = df.groupby("Team")["NETRTG"].mean().sort_values(ascending=False).head(
            int(opponent.split()[1])).index
        matchup_scores[stat] = team_scores[stat] * scaled_weighted.loc[subset][counterpart].mean()
    else:
        matchup_scores[stat] = team_scores[stat] * scaled_weighted.loc[opponent][counterpart]

# Four Factors table
core_stats = [s for s in predictors if s not in neutral_stats]
matchup_series = pd.Series({k: v for k, v in matchup_scores.items() if k in core_stats})
matchup_df = pd.DataFrame({
    "Variable": matchup_series.index.map(readable_labels),
    "Matchup Priority Score": MinMaxScaler((1, 100)).fit_transform(matchup_series.values.reshape(-1, 1)).flatten().round(0).astype(int),
    "Type": [stat_type_map[stat] for stat in matchup_series.index]
}).sort_values(by="Matchup Priority Score", ascending=False).reset_index(drop=True)
matchup_df.index += 1
matchup_df.index.name = "Rank"

def highlight(row):
    color = {"Offense": "#c7f0c7", "Defense": "#c7d7f0", "Stylistic": "#e1c7f0"}[row["Type"]]
    return ["background-color: {}".format(color) if col == "Matchup Priority Score" else "" for col in row.index]

styled_scores = matchup_df.style.apply(highlight, axis=1).hide(subset=["Type"], axis="columns")
st.subheader("Priority of Four Factors")
st.dataframe(styled_scores, use_container_width=True)

# Stylistic (neutral) section
neutral_series = pd.Series({k: v for k, v in matchup_scores.items() if k in neutral_stats})
neutral_scaled = MinMaxScaler((1, 100)).fit_transform(neutral_series.values.reshape(-1, 1)).flatten()
neutral_df = pd.DataFrame({
    "Category": [readable_labels[stat] for stat in neutral_series.index],
    "Better": ["Slower" if "Pace" in stat and importance_df.loc[team, stat] > 0 else
               "Faster" if "Pace" in stat else
               "More" if importance_df.loc[team, stat] > 0 else "Less"
               for stat in neutral_series.index],
    "Importance": neutral_scaled.round(0).astype(int),
    "Type": [stat_type_map[stat] for stat in neutral_series.index]
}).sort_values(by="Importance", ascending=False).set_index("Category")

def highlight_neutral(row):
    color = {"Offense": "#c7f0c7", "Defense": "#c7d7f0"}[row["Type"]]
    return ["background-color: {}".format(color) if col == "Importance" else "" for col in row.index]

st.subheader("Priority of Stylistic Tendencies")
st.dataframe(neutral_df.style.apply(highlight_neutral, axis=1).hide(["Type"], axis="columns"), use_container_width=True)

# Tier tables
label_to_stat = {v: k for k, v in readable_labels.items()}
selected_label = st.selectbox("Select a stat to view performance tiers", list(label_to_stat.keys()))
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
        def ordinal(n): return "%d%s" % (n, "tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
        value_str = f"{avg_val:.1f}" if "Pace" in stat or "QSQ" in stat else f"{avg_val*100:.1f}%"
        records.append({"Game Tier": tier_name, "Value": value_str, "Rank": ordinal(int(rank))})
    return pd.DataFrame(records)

col1, col2 = st.columns(2)
with col1:
    st.subheader(f"{team} — {readable_labels.get(selected_stat)}")
    st.dataframe(stat_by_tier(df, team, selected_stat).set_index("Game Tier").style.hide(axis="index"), use_container_width=True)

with col2:
    if opponent in ["Top 5 Teams", "Top 10 Teams", "Top 16 Teams"]:
        subset = df.groupby("Team")["NETRTG"].mean().sort_values(ascending=False).head(
            int(opponent.split()[1])).index
        st.subheader(f"{opponent} Avg — {readable_labels.get(stat_counterpart)}")
        combined = pd.concat([stat_by_tier(df, t, stat_counterpart) for t in subset])
        combined["Value"] = combined["Value"].astype(str).str.replace('%', '').astype(float)
        mean_tiers = combined.groupby("Game Tier")["Value"].mean().reset_index()
        mean_tiers["Value"] = mean_tiers["Value"].apply(lambda x: f"{x:.1f}%" if "Pace" not in selected_stat and "QSQ" not in selected_stat else f"{x:.1f}")
        mean_tiers["Rank"] = "–"
        st.dataframe(mean_tiers.set_index("Game Tier").style.hide(axis="index"), use_container_width=True)
    else:
        st.subheader(f"{opponent} — {readable_labels.get(stat_counterpart)}")
        st.dataframe(stat_by_tier(df, opponent, stat_counterpart).set_index("Game Tier").style.hide(axis="index"), use_container_width=True)