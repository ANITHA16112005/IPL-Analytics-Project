import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/matches.csv")

# Season wise IPL Winners (Final match only)
print("=== IPL Season Winners ===")
finals = df[df['match_type'] == 'Final']
print(finals[['season', 'winner']].to_string(index=False))
# Chart - Season wise winners
winner_counts = finals['winner'].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(winner_counts.index, winner_counts.values, color='gold')
plt.title('IPL Title Winners 2008-2025')
plt.xlabel('Teams')
plt.ylabel('Number of Titles')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('ipl_title_winners.png')
plt.show()
print("Chart saved!")
# Head to Head - CSK vs MI
team1 = "Chennai Super Kings"
team2 = "Mumbai Indians"

# Get all matches between these two teams
h2h = df[((df['team1'] == team1) & (df['team2'] == team2)) |
          ((df['team1'] == team2) & (df['team2'] == team1))]

csk_wins = len(h2h[h2h['winner'] == team1])
mi_wins = len(h2h[h2h['winner'] == team2])

print(f"\n=== Head to Head: CSK vs MI ===")
print(f"Total Matches: {len(h2h)}")
print(f"CSK Wins: {csk_wins}")
print(f"MI Wins: {mi_wins}")

# Chart
plt.figure(figsize=(6, 6))
plt.pie([csk_wins, mi_wins],
        labels=['CSK', 'MI'],
        autopct='%1.1f%%',
        colors=['yellow', 'blue'])
plt.title('Head to Head: CSK vs MI')
plt.savefig('h2h_csk_mi.png')
plt.show()
print("Head to Head chart saved!")
# Top 10 Player of the Match winners
print("\n=== Top 10 Players of the Match ===")
top_players = df['player_of_match'].value_counts().head(10)
print(top_players)

# Chart
plt.figure(figsize=(12, 6))
plt.bar(top_players.index, top_players.values, color='green')
plt.title('Top 10 Players of the Match in IPL 2008-2025')
plt.xlabel('Players')
plt.ylabel('Number of Awards')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_players.png')
plt.show()
print("Top Players chart saved!")