import pandas as pd
from sklearn.preprocessing import LabelEncoder

df_tmp = pd.read_excel('data/справочник станций по контейнерам.xlsx')
df_tmp.columns = ['code', '_', 'name']
df_tmp.index = df_tmp.code
df_tmp = df_tmp.drop(['code', '_'], axis=1)
df_tmp.name = list(map(lambda x: f'Узел: {x}', LabelEncoder().fit_transform(df_tmp.name)))

df = pd.read_csv('data/df.csv', index_col=0)
df.frm = df.frm.apply(lambda x: df_tmp.loc[x, 'name'] if x in df_tmp.index else x)
df.to = df.to.apply(lambda x: df_tmp.loc[x, 'name'] if x in df_tmp.index else x)
df = df.astype(str)
df = df[df.frm != df.to]

df.to_csv('data/df_renamed.csv')
