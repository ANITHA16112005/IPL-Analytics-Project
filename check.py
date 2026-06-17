import pandas as pd
df = pd.read_csv("data/matches.csv")

# Check season 19 specifically
season19 = df[df['season'] == 19]
print("Match types in Season 19:")
print(season19['match_type'].value_counts())
print()
print("Winner column for Season 19:")
print(season19[['match_type', 'winner']])