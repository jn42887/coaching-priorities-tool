import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# At the top of your script (after imports)
st.set_page_config(page_title="Matchup Priorities", layout="wide")

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
    'FTRate': 'Free Throw Rate',
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

importance_signed = {}
for team, group in df.groupby("Team"):
    X = group[predictors].dropna()
    y = group.loc[X.index, 'NETRTG']
    if len(X) < len(predictors):
        continue
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression().fit(X_scaled, y)
    # Define stat polarity: +1 if more is good, -1 if less is good
    direction_map = {
        "oQSQ": 1,
        "DREB": 1,
        "FTRate": 1,
        "OREB": 1,
        "OppTOVRate": 1,
        "dQSQ": -1,
        "TOVRate": -1,
    "OppFTRate": -1,
    }

    adjusted_coefs = {
        stat: coef * direction_map[stat]
        for stat, coef in zip(predictors, model.coef_)
    }
    importance_signed[team] = pd.Series(adjusted_coefs)

importance_df = pd.DataFrame(importance_signed).T.fillna(0)

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

# Sidebar explanation
with st.sidebar:
    st.header("How This Works")
    st.markdown("""
    This tool identifies the most important factors for team success in a given matchup based on data from this season.

    #### 🧠 How Priority is Calculated:
    For each stat (e.g., Defensive Rebounding), we calculate:

    - **Your Team**: Importance × 0.7 + Variability × 0.3  
    - **Opponent**: Counterpart Importance × 0.7 + Variability × 0.3

    These are then multiplied together to give a **Matchup Priority Score**, which is scaled from 1–100.

    #### 🎯 What This Means:
    If Defensive Rebounding is a high priority, it means:
    - Your team’s success is strongly tied to Defensive Rebounding.
    - The opponent’s success is strongly tied to Offensive Rebounding.
    - And either (or both) of those stats tend to fluctuate a lot game to game.

    These are the areas most worth emphasizing in preparation.
    """)
st.title("Matchup-Based Coaching Priorities")

teams = sorted(scaled_weighted.index)
team = st.selectbox("Select Your Team", teams, index=teams.index("CLE") if "CLE" in teams else 0)
opponent_options = ["All Teams", "Top 5 Teams", "Top 10 Teams", "Top 16 Teams"] + [t for t in teams if t != team]
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

# Normalize the matchup scores
matchup_series = pd.Series(matchup_scores)
scaler = MinMaxScaler(feature_range=(1, 100))
if not matchup_series.empty:
    scaled = scaler.fit_transform(matchup_series.values.reshape(-1, 1)).flatten()
else:
    scaled = [1] * len(matchup_series)

# Determine the direction of impact from original importance_df
impact_sign = importance_df.loc[team].apply(lambda x: "Positive" if x > 0 else "Negative")

matchup_df = pd.DataFrame({
    "Variable": matchup_series.index.map(readable_labels),
    "Matchup Priority Score": scaled.round(0).astype(int),
    "Impact Direction": matchup_series.index.map(impact_sign)
}).sort_values(by="Matchup Priority Score", ascending=False).reset_index(drop=True)

# Add rank column from 1 to 8
matchup_df.index += 1
matchup_df.index.name = "Rank"

# Styled DataFrame for color gradient
def color_impact(val):
    if val == "Positive":
        return "color: green"
    elif val == "Negative":
        return "color: red"
    return ""

styled_df = matchup_df.style\
    .background_gradient(cmap="Greens", subset=["Matchup Priority Score"])\
    .applymap(color_impact, subset=["Impact Direction"])


st.dataframe(styled_df, use_container_width=True)

# Stat breakdown selector
label_to_stat = {v: k for k, v in readable_labels.items()}
readable_options = list(label_to_stat.keys())

# Make sure the session state is initialized before using it
if "selected_stat" not in st.session_state:
    st.session_state["selected_stat"] = readable_options[0]

selected_label = st.selectbox(
    "Select a stat to view team performance tiers",
    readable_options,
    index=readable_options.index(st.session_state["selected_stat"])
)

# Update session state and get internal stat key
st.session_state["selected_stat"] = selected_label
selected_stat = label_to_stat[selected_label]
stat_counterpart = counterpart_map[selected_stat]

# Function to get team stat averages and ranks by tier
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
        # Get ranks vs other teams in same tier
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
        def ordinal(n):
            return "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1)*(n % 10 < 4)*n % 10::4])
        rank_num = int(rank)
        rank_str = ordinal(rank_num)
        value_str = f"{avg_val * 100:.1f}%"
        records.append({"Game Tier": tier_name, "Value": value_str, "Rank": rank_str})
    return pd.DataFrame(records)

# Show side-by-side tables
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"{team} — {readable_labels.get(selected_stat, selected_stat)}")
    df_team_stat = stat_by_tier(df, team, selected_stat)
    st.dataframe(df_team_stat.set_index("Game Tier"), use_container_width=True)

with col2:
    if opponent in ["Top 5 Teams", "Top 10 Teams", "Top 16 Teams"]:
        # use average of all teams in opponent group
        subset_map = {
            "Top 5 Teams": df.groupby("Team")["NETRTG"].mean().sort_values(ascending=False).head(5).index.tolist(),
            "Top 10 Teams": df.groupby("Team")["NETRTG"].mean().sort_values(ascending=False).head(10).index.tolist(),
            "Top 16 Teams": df.groupby("Team")["NETRTG"].mean().sort_values(ascending=False).head(16).index.tolist(),
        }
        subset_teams = subset_map[opponent]
        st.subheader(f"{opponent} Avg — {readable_labels.get(stat_counterpart, stat_counterpart)}")
        avg_df = pd.concat([stat_by_tier(df, opp, stat_counterpart) for opp in subset_teams])
        tier_means = avg_df.groupby("Game Tier").agg({"Value": "mean"}).reset_index()
        tier_means["Rank"] = "–"
        st.dataframe(tier_means, use_container_width=True)
    else:
        st.subheader(f"{opponent} — {readable_labels.get(stat_counterpart, stat_counterpart)}")
        df_opp_stat = stat_by_tier(df, opponent, stat_counterpart)
        st.dataframe(df_opp_stat.set_index("Game Tier"), use_container_width=True)

# Show which teams are in the selected subset
if opponent in ["Top 5 Teams", "Top 10 Teams", "Top 16 Teams"]:
    subset_map = {
        "Top 5 Teams": df.groupby("Team")["NETRTG"].mean().sort_values(ascending=False).head(5).index.tolist(),
        "Top 10 Teams": df.groupby("Team")["NETRTG"].mean().sort_values(ascending=False).head(10).index.tolist(),
        "Top 16 Teams": df.groupby("Team")["NETRTG"].mean().sort_values(ascending=False).head(16).index.tolist(),
    }
    subset_teams = subset_map.get(opponent, [])
    if subset_teams:
        with st.expander(f"View teams in {opponent}"):
            st.markdown(', '.join(subset_teams))


