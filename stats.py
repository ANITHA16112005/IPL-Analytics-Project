import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/matches.csv")

# Team Win Percentage
print("=== Team Win Percentage ===")
total_matches = pd.concat([df['team1'], df['team2']]).value_counts()
total_wins = df['winner'].value_counts()

win_percentage = (total_wins / total_matches * 100).round(2)
win_percentage = win_percentage.dropna().sort_values(ascending=False)
print(win_percentage)
# Chart - Win Percentage
plt.figure(figsize=(14, 6))
plt.bar(win_percentage.index, win_percentage.values, color='purple')
plt.title('IPL Team Win Percentage 2008-2025')
plt.xlabel('Teams')
plt.ylabel('Win Percentage %')
plt.xticks(rotation=45, ha='right')
plt.axhline(y=50, color='red', linestyle='--', label='50% line')
plt.legend()
plt.tight_layout()
plt.savefig('win_percentage.png')
plt.show()
print("Win Percentage chart saved!")

# Venue wise matches
print("\n=== Top 10 Venues ===")
top_venues = df['venue'].value_counts().head(10)
print(top_venues)

# Chart - Top Venues
plt.figure(figsize=(14, 6))
plt.bar(top_venues.index, top_venues.values, color='orange')
plt.title('Top 10 IPL Venues 2008-2025')
plt.xlabel('Venue')
plt.ylabel('Number of Matches')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_venues.png')
plt.show()
print("Venues chart saved!")