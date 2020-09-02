import pandas as pd
import numpy as np
import sys

df = pd.read_csv(f'data/df{"_renamed" if "--renamed" in sys.argv else ""}.csv', index_col=0)
df_prices = pd.DataFrame({'Price': None}, index=[])

for x in df.frm.unique():
    for y in df[df.frm == x].to.unique():
        df_prices.loc[f'{x} -> {y}', 'Price'] = np.random.random() * 200

df_prices.Price = df_prices.Price.astype(float)

df_prices.to_csv(f'data/prices{"_renamed" if "--renamed" in sys.argv else ""}.csv')
