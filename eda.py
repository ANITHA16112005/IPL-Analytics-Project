import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/matches.csv")

# Chart 1 - Top 10 Winning Teams
top_teams = df['winner'].value_counts().head(10)
plt.figure(figsize=(12, 6))
plt.bar(top_teams.index, top_teams.values, color='blue')
plt.title('Top 10 Winning Teams in IPL 2008-2025')
plt.xlabel('Teams')
plt.ylabel('Number of Wins')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_teams.png')
plt.show()
print("Chart 1 saved!")

# Chart 2 - Toss Decision
toss = df['toss_decision'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(toss.values, labels=toss.index, autopct='%1.1f%%', colors=['orange', 'green'])
plt.title('Toss Decision Analysis')
plt.savefig('toss_decision.png')
plt.show()
print("Chart 2 saved!")

# Chart 3 - Matches Per Season
season = df['season'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
plt.plot(season.index, season.values, marker='o', color='red')
plt.title('Matches Per Season')
plt.xlabel('Season')
plt.ylabel('Number of Matches')
plt.grid(True)
plt.savefig('matches_per_season.png')
plt.show()
print("Chart 3 saved!")