import pandas as pd

df = pd.read_csv("./data/users.csv")

df["Age"].apply(lambda x: "Young" if x < 30 else "Adult")

df.to_csv("./data/processed_users.csv", index=False)