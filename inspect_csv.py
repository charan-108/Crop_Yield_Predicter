import pandas as pd

df = pd.read_csv(r'C:\Users\ramu6\crop yield predictor\crop_yield.csv')
print(df.head().to_string(index=False))
print('---')
print(df.columns.tolist())
print('SHAPE:', df.shape)
