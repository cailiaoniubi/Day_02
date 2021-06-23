import pandas as pd


# Create DataFrame
d = {"column1": 1, "column2": "string"}
df = pd.DataFrame.from_dict(d, orient="index")

pd.DataFrame()

df.head()