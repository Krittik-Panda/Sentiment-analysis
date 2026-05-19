import pandas as pd

df = pd.read_csv("data/training_1600000_processed_noemoticon.csv")

# keep header automatically, remove first 526420 data rows
df = df.iloc[526420:]

df.to_csv("data/training_1600000_processed_noemoticon.csv", index=False)

print("Rows deleted successfully.")