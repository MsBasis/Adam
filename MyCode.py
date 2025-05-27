import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader 
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 


#przygotowanie csv z poprawnymi aminokwasami
'''
df = pd.read_csv(r'C:\\Studia\\Progranmy\\Adam\\sum.csv', low_memory=False)
aminokwasy = set('ACDEFGHIKLMNPQRSTVWY')

def czy_valid(seq): #sprawdzanie poprawnych
    return isinstance(seq, str) and all(aa in aminokwasy for aa in seq) and len(seq) <= 30

df_valid = df[df['Epitope - Name'].apply(czy_valid)]

df_valid.to_csv(r'C:\\Studia\\Progranmy\\Adam\\valid_sequences.csv', index=False)

print(f"Zapisano {len(df_valid)} poprawnych sekwencji do pliku 'valid_sequences.csv'")
'''



























