import pandas as pd

df = pd.read_excel('data/для практики (контейнеры).xlsx')
df = df.iloc[:, [2, 4]]
df.columns = ['frm', 'to']
df.frm = df.frm.apply(lambda x: x[1:])
df.to = df.to.apply(lambda x: x[1:])
df = df.astype(int)

df.to_csv('data/df.csv')
