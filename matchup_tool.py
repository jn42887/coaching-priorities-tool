import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

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
    '3PA Rate': '3PA Rate Allowed', '3PA Rate Allowed': '3PA Rate',
    'AvgOffPace': 'AvgDefPace', 'AvgDefPace': 'AvgOffPace'
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
    '3PA Rate': '3PA Rate',
    '3PA Rate Allowed': '3PA Rate Allowed',
    'AvgOffPace': 'Avg Off Pace',
    'AvgDefPace': 'Avg Def Pace'
}

positive_stats = ["oQSQ", "DREB", "FTRate", "OREB", "OppTOVRate"]
negative_stats = ["dQSQ", "TOVRate", "OppFTRate"]
neutral_stats = ["3PA Rate", "3PA Rate Allowed", "AvgOffPace", "AvgDefPace"]
predictors = list(counterpart_map.keys()) + neutral_stats

# Regression-based importance
importance_signed = {}
for team, group in df.groupby("Team"):
    X = group[predictors].dropna()
    y = group.loc[X.index, 'NETRTG']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression().fit(X_scaled, y)
    direction_map = {stat: 1 for stat in positive_stats}
    direction_map.update({stat: -1 for stat in negative_stats})
    adjusted_coefs = {
        stat: coef * direction_map.get(stat, 1)
        for stat, coef in zip(predictors, model.coef_)
    }
    importance_signed[team] = pd.Series(adjusted_coefs)

importance_df = pd.DataFrame(importance_signed).T.fillna(0)
variance_df = df.groupby("Team")[predictors].var().fillna(0)

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
    else:
        matchup_scores[stat] = team_scores[stat] * selected_priority.loc[opponent][counterpart]

matchup_series = pd.Series(matchup_scores)
scaler = MinMaxScaler(feature_range=(1, 100))
scaled = scaler.fit_transform(matchup_series.values.reshape(-1, 1)).flatten()

matchup_df = pd.DataFrame({
    "Variable": matchup_series.index.map(readable_labels),
    "Matchup Priority Score": scaled.round(0).astype(int),
}).sort_values(by="Matchup Priority Score", ascending=False).reset_index(drop=True)
matchup_df.index += 1
matchup_df.index.name = "Rank"

styled_df = matchup_df.style.background_gradient(cmap="Greens", subset=["Matchup Priority Score"])
st.dataframe(styled_df, use_container_width=True)

# Neutral stat tendencies
neutral_importance = importance_df[neutral_stats].abs()
scaler = MinMaxScaler(feature_range=(1, 100))
scaled_neutral_importance = pd.DataFrame(
    scaler.fit_transform(neutral_importance),
    index=neutral_importance.index,
    columns=neutral_importance.columns
)
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
st.dataframe(styled_neutral_df.set_index("Category"), use_container_width=True)
