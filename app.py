import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="IPL Analytics Dashboard", page_icon="🏏", layout="wide")

# Load data
df = pd.read_csv("data/matches.csv")

# Title
st.title("🏏 IPL Analytics & Win Prediction Dashboard")
st.markdown("**IPL Data 2008 - 2025**")

# Row 1 - Stats
col1, col2, col3 = st.columns(3)
col1.metric("Total Matches", len(df))
col2.metric("Total Seasons", df['season'].nunique())
col3.metric("Total Teams", df['team1'].nunique())

st.divider()

# Top 10 Winning Teams Chart
st.subheader("🏆 Top 10 Winning Teams")
top_teams = df['winner'].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(top_teams.index, top_teams.values, color='blue')
ax.set_xlabel("Teams")
ax.set_ylabel("Wins")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)

st.divider()

# Season Winners
st.subheader("🥇 IPL Season Winners")
finals = df[df['match_type'] == 'Final'][['season', 'winner']]
st.dataframe(finals.reset_index(drop=True), use_container_width=True)

st.divider()

# Toss Analysis
st.subheader("🎯 Toss Decision Analysis")
toss = df['toss_decision'].value_counts()
fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.pie(toss.values, labels=toss.index, autopct='%1.1f%%', colors=['orange', 'green'])
ax2.set_title('Toss Decision')
st.pyplot(fig2)

st.divider()

# Head to Head
st.subheader("⚔️ Head to Head Comparison")
teams = sorted(df['team1'].dropna().unique().tolist())
col1, col2 = st.columns(2)
team1 = col1.selectbox("Select Team 1", teams)
team2 = col2.selectbox("Select Team 2", teams, index=1)

h2h = df[((df['team1'] == team1) & (df['team2'] == team2)) |
          ((df['team1'] == team2) & (df['team2'] == team1))]

t1_wins = len(h2h[h2h['winner'] == team1])
t2_wins = len(h2h[h2h['winner'] == team2])

col1.metric(f"{team1} Wins", t1_wins)
col2.metric(f"{team2} Wins", t2_wins)
st.divider()

# Win Predictor
st.subheader("🎯 Win Probability Predictor")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Train model
df_model = df[['team1', 'team2', 'toss_winner', 'toss_decision', 'venue', 'winner']].dropna()

le_dict = {}
df_encoded = df_model.copy()
for col in df_encoded.columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    le_dict[col] = le

X = df_encoded[['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']]
y = df_encoded['winner']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# User inputs
teams_list = sorted(df['team1'].dropna().unique().tolist())
venues_list = sorted(df['venue'].dropna().unique().tolist())

col1, col2 = st.columns(2)
p_team1 = col1.selectbox("Team 1", teams_list, key="p1")
p_team2 = col2.selectbox("Team 2", teams_list, index=1, key="p2")
p_toss_winner = st.selectbox("Toss Winner", [p_team1, p_team2])
p_toss_decision = st.selectbox("Toss Decision", ["bat", "field"])
p_venue = st.selectbox("Venue", venues_list)

if st.button("🏏 Predict Winner!"):
    try:
        input_data = {
            'team1': le_dict['team1'].transform([p_team1])[0],
            'team2': le_dict['team2'].transform([p_team2])[0],
            'toss_winner': le_dict['toss_winner'].transform([p_toss_winner])[0],
            'toss_decision': le_dict['toss_decision'].transform([p_toss_decision])[0],
            'venue': le_dict['venue'].transform([p_venue])[0],
        }
        prediction = model.predict([list(input_data.values())])[0]
        predicted_team = le_dict['winner'].inverse_transform([prediction])[0]
        st.success(f"🏆 Predicted Winner: **{predicted_team}**")
    except:
        st.error("Please select valid teams and venue!")