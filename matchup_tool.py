import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Load the Excel file
file_path = "Four Factors by Team and Game.xlsx"
df = pd.read_excel(file_path)

# Define counterpart relationships and readable labels
counterpart_map = {
    'OREB': 'DREB', 'DREB': 'OREB',
    'FTRate': 'OppFTRate', 'OppFTRate': 'FTRate',
    'TOVRate': 'OppTOVRate', 'OppTOVRate': 'TOVRate',
    'oQSQ': 'dQSQ', 'dQSQ': 'oQSQ'
}

readable_labels = {
    'OREB': 'Offensive Rebounding',
    'DREB': 'Defensive Rebounding',
    'FTRate': 'Free Throw Rate (Rim Attacking)',
    'OppFTRate': 'Opponent Free Throw Rate',
    'TOVRate': 'Turnovers',
    'OppTOVRate': 'Opponent Turnovers',
    'oQSQ': 'Offensive Shot Quality',
    'dQSQ': 'Defensive Shot Quality'
}

# Define relevant columns
predictors = list(counterpart_map.keys())

# Calculate importance (std coefficients from linear regression)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

importance_scores = {}
for team, group in df.groupby("Team"):
    X = group[predictors].dropna()
    y = group.loc[X.index, 'NETRTG']
    if len(X) < len(predictors):
        continue
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression().fit(X_scaled, y)
    importance_scores[team] = pd.Series(abs(model.coef_), index=predictors)

importance_df = pd.DataFrame(importance_scores).T.fillna(0)

# Calculate variance per predictor per team
variance_df = df.groupby("Team")[predictors].var().fillna(0)

# Compute different versions of priority scores
priority_product = importance_df * variance_df
priority_weighted = 0.7 * importance_df + 0.3 * variance_df
priority_power = (importance_df * variance_df) ** 1.5

# Normalize each across teams (statwise)
def statwise_scale(df):
    scaled_df = df.copy()
    for col in scaled_df.columns:
        scaler = MinMaxScaler(feature_range=(1, 100))
        scaled_df[col] = scaler.fit_transform(scaled_df[[col]])
    return scaled_df

scaled_product = statwise_scale(priority_product)
scaled_weighted = statwise_scale(priority_weighted)
scaled_power = statwise_scale(priority_power)

# Streamlit UI
st.title("Matchup-Based Coaching Priorities")

teams = sorted(scaled_weighted.index)
team = st.selectbox("Select Your Team", teams, index=teams.index("CLE") if "CLE" in teams else 0)
opponent_options = ["All Teams"] + [t for t in teams if t != team]
opponent = st.selectbox("Select Opponent", opponent_options)

# Hide advanced options in an expander
with st.expander("Advanced Settings: Priority Method", expanded=False):
    method = st.radio("Choose Method for Calculating Priority Scores", ["Product (x)", "Weighted Average", "Powered Product^1.5"], index=1)

# Select which scaled version to use
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
    elif stat in team_scores and counterpart in selected_priority.loc[opponent]:
        matchup_scores[stat] = team_scores[stat] * selected_priority.loc[opponent][counterpart]

# Normalize the matchup scores
matchup_series = pd.Series(matchup_scores)
scaler = MinMaxScaler(feature_range=(1, 100))
scaled = scaler.fit_transform(matchup_series.values.reshape(-1, 1)).flatten()

matchup_df = pd.DataFrame({
    "Variable": matchup_series.index.map(readable_labels),
    "Matchup Priority Score": scaled.round(0).astype(int)
}).sort_values(by="Matchup Priority Score", ascending=False).reset_index(drop=True)

# Add rank column from 1 to 8
matchup_df.index += 1
matchup_df.index.name = "Rank"

# Styled DataFrame for color gradient
styled_df = matchup_df.style.background_gradient(cmap="Greens", subset=["Matchup Priority Score"])

st.dataframe(styled_df, use_container_width=True)
