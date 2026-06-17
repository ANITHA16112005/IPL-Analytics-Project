import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page config
st.set_page_config(
    page_title="IPL Analytics Dashboard",
    page_icon="🏏",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0f1117; }
    .stMetric {
        background: linear-gradient(135deg, #1e3a5f, #0d47a1);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00e5ff;
    }
    .stMetric label { color: #90caf9 !important; font-size: 14px !important; }
    .stMetric [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 28px !important; }
    h1 { color: #00e5ff !important; }
    h2, h3 { color: #90caf9 !important; }
    .stTabs [data-baseweb="tab"] { color: #90caf9; }
    .stTabs [aria-selected="true"] { color: #00e5ff !important; border-bottom: 2px solid #00e5ff; }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/matches.csv")
    return df

df = load_data()

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/8/8a/IPL_Logo.png/220px-IPL_Logo.png", width=150)
st.sidebar.title("🏏 IPL Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio("📌 Navigate To", [
    "🏠 Home Dashboard",
    "📊 Team Statistics",
    "🌟 Player Statistics",
    "⚔️ Head to Head",
    "🏟️ Venue Analysis",
    "🎯 Win Predictor",
    "🤖 Model Accuracy"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**📅 Data:** IPL 2008 - 2026")
st.sidebar.markdown(f"**🗂️ Matches:** {len(df)}")
st.sidebar.markdown("**🏆 Seasons:** 19")

# ============================================================
# PAGE 1 - HOME DASHBOARD
# ============================================================
if page == "🏠 Home Dashboard":
    st.title("🏏 IPL Analytics & Win Prediction Dashboard")
    st.markdown("### 📅 IPL Data 2008 - 2026")
    st.markdown("---")

    seasons = ["All Seasons"] + sorted(df['season'].unique().tolist())
    selected_season = st.selectbox("🔍 Filter by Season", seasons)

    if selected_season != "All Seasons":
        filtered_df = df[df['season'] == selected_season]
    else:
        filtered_df = df

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏏 Total Matches", len(filtered_df))
    col2.metric("📅 Total Seasons", filtered_df['season'].nunique())
    col3.metric("🏟️ Total Teams", filtered_df['team1'].nunique())
    col4.metric("🏆 Total Venues", filtered_df['venue'].nunique())

    st.markdown("---")

    st.subheader("🏆 Top 10 Winning Teams")
    top_teams = filtered_df['winner'].value_counts().head(10)
    colors = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7',
              '#DDA0DD','#98D8C8','#F7DC6F','#82E0AA','#85C1E9']
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1a2e')
    bars = ax.bar(top_teams.index, top_teams.values, color=colors)
    ax.set_xlabel("Teams", color='white')
    ax.set_ylabel("Wins", color='white')
    ax.tick_params(colors='white')
    for bar, val in zip(bars, top_teams.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', color='white', fontsize=10)
    plt.xticks(rotation=45, ha='right', color='white')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🥇 IPL Season Winners")
        finals = filtered_df[filtered_df['match_type'] == 'Final'][['season', 'winner']]
        st.dataframe(finals.reset_index(drop=True), use_container_width=True, height=400)

    with col2:
        st.subheader("🎯 Toss Decision Analysis")
        toss = filtered_df['toss_decision'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        fig2.patch.set_facecolor('#0f1117')
        ax2.set_facecolor('#1a1a2e')
        ax2.pie(toss.values, labels=toss.index, autopct='%1.1f%%',
                colors=['#FF6B6B', '#4ECDC4'],
                textprops={'color': 'white'})
        ax2.set_title('Toss Decision', color='white')
        st.pyplot(fig2)

# ============================================================
# PAGE 2 - TEAM STATISTICS
# ============================================================
elif page == "📊 Team Statistics":
    st.title("📊 Team Statistics")
    st.markdown("---")

    total_matches = pd.concat([df['team1'], df['team2']]).value_counts()
    total_wins = df['winner'].value_counts()
    win_pct = (total_wins / total_matches * 100).round(1)
    finals = df[df['match_type'] == 'Final']
    titles = finals['winner'].value_counts()

    team_stats = pd.DataFrame({
        'Team': total_matches.index,
        'Matches': total_matches.values,
        'Wins': [total_wins.get(t, 0) for t in total_matches.index],
        'Win %': [win_pct.get(t, 0) for t in total_matches.index],
        'Titles': [titles.get(t, 0) for t in total_matches.index]
    }).sort_values('Win %', ascending=False).reset_index(drop=True)

    selected_team = st.selectbox("🔍 Select Team", sorted(df['team1'].dropna().unique().tolist()))
    team_row = team_stats[team_stats['Team'] == selected_team]
    if not team_row.empty:
        r = team_row.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏏 Matches", int(r['Matches']))
        col2.metric("🏆 Wins", int(r['Wins']))
        col3.metric("📈 Win %", f"{r['Win %']}%")
        col4.metric("🥇 Titles", int(r['Titles']))

    st.markdown("---")
    st.subheader("📋 All Teams Statistics")
    st.dataframe(team_stats, use_container_width=True, height=500)

    st.markdown("---")
    st.subheader("📈 Team Win Percentage Chart")
    top_stat = team_stats.head(10)
    colors = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7',
              '#DDA0DD','#98D8C8','#F7DC6F','#82E0AA','#85C1E9']
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1a2e')
    bars = ax.bar(top_stat['Team'], top_stat['Win %'], color=colors)
    ax.axhline(y=50, color='red', linestyle='--', label='50% line')
    ax.set_xlabel("Teams", color='white')
    ax.set_ylabel("Win %", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    for bar, val in zip(bars, top_stat['Win %']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val}%", ha='center', color='white', fontsize=9)
    plt.xticks(rotation=45, ha='right', color='white')
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================
# PAGE 3 - PLAYER STATISTICS
# ============================================================
elif page == "🌟 Player Statistics":
    st.title("🌟 Player Statistics")
    st.markdown("---")

    st.subheader("🏆 Most Player of the Match Awards")
    top_players = df['player_of_match'].value_counts().head(15)

    col1, col2 = st.columns([2, 1])
    with col1:
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(top_players)))
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1a1a2e')
        bars = ax.bar(top_players.index, top_players.values, color=colors)
        ax.set_xlabel("Players", color='white')
        ax.set_ylabel("Awards", color='white')
        ax.tick_params(colors='white')
        for bar, val in zip(bars, top_players.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(val), ha='center', color='white', fontsize=9)
        plt.xticks(rotation=45, ha='right', color='white')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.dataframe(
            top_players.reset_index().rename(
                columns={'player_of_match': 'Player', 'count': 'Awards'}),
            use_container_width=True, height=400)

    st.markdown("---")
    st.subheader("📈 Team Performance Season by Season")
    all_teams = sorted(df['winner'].dropna().unique().tolist())
    selected_team = st.selectbox("Select Team", all_teams)
    team_trend = df[df['winner'] == selected_team].groupby('season').size()
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    fig4.patch.set_facecolor('#0f1117')
    ax4.set_facecolor('#1a1a2e')
    ax4.plot(team_trend.index, team_trend.values, marker='o', color='#00e5ff', linewidth=2)
    ax4.fill_between(team_trend.index, team_trend.values, alpha=0.3, color='#00e5ff')
    ax4.set_xlabel("Season", color='white')
    ax4.set_ylabel("Wins", color='white')
    ax4.set_title(f"{selected_team} - Wins Per Season", color='white')
    ax4.tick_params(colors='white')
    plt.tight_layout()
    st.pyplot(fig4)

# ============================================================
# PAGE 4 - HEAD TO HEAD
# ============================================================
elif page == "⚔️ Head to Head":
    st.title("⚔️ Head to Head Comparison")
    st.markdown("---")

    teams = sorted(df['team1'].dropna().unique().tolist())
    col1, col2 = st.columns(2)
    team1 = col1.selectbox("Select Team 1", teams)
    team2 = col2.selectbox("Select Team 2", teams, index=1)

    h2h = df[((df['team1'] == team1) & (df['team2'] == team2)) |
              ((df['team1'] == team2) & (df['team2'] == team1))]

    t1_wins = len(h2h[h2h['winner'] == team1])
    t2_wins = len(h2h[h2h['winner'] == team2])
    total = len(h2h)

    col1, col2, col3 = st.columns(3)
    col1.metric(f"🏆 {team1} Wins", t1_wins)
    col2.metric("🏏 Total Matches", total)
    col3.metric(f"🏆 {team2} Wins", t2_wins)

    st.markdown("---")

    if total > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Win % Comparison")
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.patch.set_facecolor('#0f1117')
            ax.pie([t1_wins, t2_wins],
                   labels=[team1, team2],
                   autopct='%1.1f%%',
                   colors=['#FF6B6B', '#4ECDC4'],
                   textprops={'color': 'white'})
            st.pyplot(fig)

        with col2:
            st.subheader("Recent 5 Matches")
            recent = h2h.tail(5)[['match_date', 'team1', 'team2', 'winner']].reset_index(drop=True)
            st.dataframe(recent, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 All Matches History")
    st.dataframe(h2h[['match_date', 'venue', 'toss_winner',
                       'toss_decision', 'winner']].reset_index(drop=True),
                 use_container_width=True)

# ============================================================
# PAGE 5 - VENUE ANALYSIS
# ============================================================
elif page == "🏟️ Venue Analysis":
    st.title("🏟️ Venue Analysis")
    st.markdown("---")

    top_venues = df['venue'].value_counts().head(10)
    st.subheader("🏟️ Top 10 IPL Venues")
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_venues)))
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1a2e')
    bars = ax.bar(top_venues.index, top_venues.values, color=colors)
    ax.set_xlabel("Venue", color='white')
    ax.set_ylabel("Matches", color='white')
    ax.tick_params(colors='white')
    for bar, val in zip(bars, top_venues.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha='center', color='white', fontsize=9)
    plt.xticks(rotation=45, ha='right', color='white')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    selected_venue = st.selectbox("🔍 Select Venue for Details",
                                   sorted(df['venue'].dropna().unique().tolist()))
    venue_df = df[df['venue'] == selected_venue]

    col1, col2, col3 = st.columns(3)
    col1.metric("🏏 Total Matches", len(venue_df))
    most_wins_team = venue_df['winner'].value_counts().idxmax() if len(venue_df) > 0 else "N/A"
    col2.metric("🏆 Most Wins", most_wins_team)
    col3.metric("📅 Seasons", venue_df['season'].nunique())

    st.subheader(f"Team Wins at {selected_venue}")
    venue_wins = venue_df['winner'].value_counts().head(8)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    fig2.patch.set_facecolor('#0f1117')
    ax2.set_facecolor('#1a1a2e')
    ax2.bar(venue_wins.index, venue_wins.values, color='#4ECDC4')
    ax2.set_xlabel("Teams", color='white')
    ax2.set_ylabel("Wins", color='white')
    ax2.tick_params(colors='white')
    plt.xticks(rotation=45, ha='right', color='white')
    plt.tight_layout()
    st.pyplot(fig2)

# ============================================================
# PAGE 6 - WIN PREDICTOR
# ============================================================
elif page == "🎯 Win Predictor":
    st.title("🎯 Win Probability Predictor")
    st.markdown("---")

    @st.cache_resource
    def train_model(df):
        df_model = df[['team1', 'team2', 'toss_winner', 'toss_decision', 'venue', 'winner']].dropna()
        le_dict = {}
        df_encoded = df_model.copy()
        for col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            le_dict[col] = le
        X = df_encoded[['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']]
        y = df_encoded['winner']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        return model, le_dict, accuracy

    model, le_dict, accuracy = train_model(df)
    st.info(f"🤖 Model Accuracy: **{round(accuracy * 100, 1)}%**")
    st.markdown("---")

    teams_list = sorted(df['team1'].dropna().unique().tolist())
    venues_list = sorted(df['venue'].dropna().unique().tolist())

    col1, col2 = st.columns(2)
    p_team1 = col1.selectbox("🏏 Team 1", teams_list, key="p1")
    p_team2 = col2.selectbox("🏏 Team 2", teams_list, index=1, key="p2")
    p_toss_winner = st.selectbox("🪙 Toss Winner", [p_team1, p_team2])
    p_toss_decision = st.selectbox("🎯 Toss Decision", ["bat", "field"])
    p_venue = st.selectbox("🏟️ Venue", venues_list)

    if st.button("🏏 Predict Winner!", use_container_width=True):
        try:
            input_data = {
                'team1': le_dict['team1'].transform([p_team1])[0],
                'team2': le_dict['team2'].transform([p_team2])[0],
                'toss_winner': le_dict['toss_winner'].transform([p_toss_winner])[0],
                'toss_decision': le_dict['toss_decision'].transform([p_toss_decision])[0],
                'venue': le_dict['venue'].transform([p_venue])[0],
            }
            prediction = model.predict([list(input_data.values())])[0]
            probabilities = model.predict_proba([list(input_data.values())])[0]
            classes = le_dict['winner'].inverse_transform(model.classes_)

            t1_prob = t2_prob = 0
            for i, team in enumerate(classes):
                if team == p_team1: t1_prob = probabilities[i]
                if team == p_team2: t2_prob = probabilities[i]

            total = t1_prob + t2_prob
            if total > 0:
                t1_prob = round((t1_prob / total) * 100, 1)
                t2_prob = round((t2_prob / total) * 100, 1)
            else:
                t1_prob = t2_prob = 50.0

            predicted_team = le_dict['winner'].inverse_transform([prediction])[0]
            st.success(f"🏆 Predicted Winner: **{predicted_team}**")

            col1, col2 = st.columns(2)
            col1.metric(f"{p_team1} Win Probability", f"{t1_prob}%")
            col2.metric(f"{p_team2} Win Probability", f"{t2_prob}%")

            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0f1117')
            ax.set_facecolor('#1a1a2e')
            bars = ax.bar([p_team1, p_team2], [t1_prob, t2_prob],
                          color=['#FF6B6B', '#4ECDC4'])
            ax.set_ylabel("Win Probability %", color='white')
            ax.tick_params(colors='white')
            for bar, val in zip(bars, [t1_prob, t2_prob]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val}%", ha='center', color='white', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================================
# PAGE 7 - MODEL ACCURACY
# ============================================================
elif page == "🤖 Model Accuracy":
    st.title("🤖 Model Accuracy & Evaluation")
    st.markdown("---")

    @st.cache_resource
    def evaluate_model(df):
        df_model = df[['team1', 'team2', 'toss_winner', 'toss_decision', 'venue', 'winner']].dropna()
        le_dict = {}
        df_encoded = df_model.copy()
        for col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            le_dict[col] = le
        X = df_encoded[['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']]
        y = df_encoded['winner']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        return accuracy, train_accuracy, X_train, X_test, model

    accuracy, train_accuracy, X_train, X_test, model = evaluate_model(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Test Accuracy", f"{round(accuracy * 100, 1)}%")
    col2.metric("📈 Train Accuracy", f"{round(train_accuracy * 100, 1)}%")
    col3.metric("🏏 Training Samples", len(X_train))

    st.markdown("---")
    st.subheader("📊 Feature Importance")
    features = ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']
    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1a2e')
    colors = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7']
    bars = ax.bar(features, importances, color=colors)
    ax.set_xlabel("Features", color='white')
    ax.set_ylabel("Importance", color='white')
    ax.set_title("Which factors matter most for prediction?", color='white')
    ax.tick_params(colors='white')
    for bar, val in zip(bars, importances):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.3f}", ha='center', color='white', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("ℹ️ About the Model")
    st.markdown("""
    | Detail | Value |
    |--------|-------|
    | Algorithm | Random Forest Classifier |
    | Trees | 100 |
    | Train/Test Split | 80% / 20% |
    | Features Used | Team1, Team2, Toss Winner, Toss Decision, Venue |
    | Dataset | IPL 2008-2026 (1208 matches) |
    """)
