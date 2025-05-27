import pandas as pd


pos = pd.read_csv(r'C:\\Studia\\Progranmy\\Adam\\positive_clean2.csv', low_memory=False)
neg = pd.read_csv(r'C:\\Studia\\Progranmy\\Adam\\negative_clean2.csv', low_memory=False)


def joiin(df_pos, df_neg):
    combined = pd.concat([df_pos, df_neg], ignore_index=True)    
    return combined

equal = joiin(pos, neg)

print(equal.head())

# Zapisz wynik (opcjonalnie)
equal.to_csv(r'C:\\Studia\\Progranmy\\Adam\\sum.csv', index=False)










