import pandas as pd



pos = pd.read_csv(r'C:\\Studia\\Progranmy\\Adam\\positive_clean.csv', low_memory=False)
neg = pd.read_csv(r'C:\\Studia\\Progranmy\\Adam\\negative_clean.csv', low_memory=False)


def remove_dashes(df):
    return df.apply(lambda col: col.str.replace('-', '', regex=False) if col.dtype == 'object' else col)

df = remove_dashes(pos)
df1 = remove_dashes(neg)

df.to_csv('C:\\Studia\\Progranmy\\Adam\\positive_clean2.csv', index=False)
df1.to_csv('C:\\Studia\\Progranmy\\Adam\\negative_clean2.csv', index=False)







