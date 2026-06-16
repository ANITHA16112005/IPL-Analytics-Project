import pandas as pd

# Load both datasets
old_df = pd.read_csv("data/matches.csv")
new_df = pd.read_csv("data/ipl2026.csv")

# Fix team short names to full names
team_names = {
    'CSK': 'Chennai Super Kings',
    'MI': 'Mumbai Indians',
    'RCB': 'Royal Challengers Bengaluru',
    'KKR': 'Kolkata Knight Riders',
    'RR': 'Rajasthan Royals',
    'SRH': 'Sunrisers Hyderabad',
    'DC': 'Delhi Capitals',
    'PBKS': 'Punjab Kings',
    'GT': 'Gujarat Titans',
    'LSG': 'Lucknow Super Giants'
}

# Rename columns to match old dataset
new_df = new_df.rename(columns={
    'date': 'match_date',
    'match_winner': 'winner',
    'player_of_the_match': 'player_of_match',
    'stage': 'match_type'
})

# Replace short names with full names
for col in ['team1', 'team2', 'winner', 'toss_winner']:
    new_df[col] = new_df[col].map(team_names)

# Add season column
new_df['season'] = 19

# Keep only matching columns
common_cols = ['team1', 'team2', 'match_date', 'toss_winner',
               'toss_decision', 'winner', 'player_of_match',
               'venue', 'season', 'match_type']

new_df = new_df[common_cols]
old_df = old_df[common_cols]

# Combine both datasets
combined = pd.concat([old_df, new_df], ignore_index=True)

# Save combined dataset
combined.to_csv("data/matches.csv", index=False)

print("✅ Done! Combined dataset saved!")
print("Total matches:", len(combined))
print("Total seasons:", combined['season'].nunique())
print("Seasons list:", sorted(combined['season'].unique().tolist()))