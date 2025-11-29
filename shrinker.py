import pandas as pd

df = pd.read_csv("./AI Tools Adoption.csv")

# df_small = df.sample(n=4000, random_state=42)

df_small = df.head()

df_small.to_csv("AI Tools Adoption small.csv")