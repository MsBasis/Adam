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

#print(f"Zapisano {len(df_valid)} poprawnych sekwencji do pliku 'valid_sequences.csv'")
'''
#cieply_encode
aminokwasy2 = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_index = {aa: i for i, aa in enumerate(aminokwasy2)}

def wektorowanie(seq, max_len=30):
    encoded = np.zeros((max_len, 20), dtype=int)

    for i, aa in enumerate(seq):
        encoded[i, aa_to_index[aa]] = 1  

    return encoded.flatten()  

#przygotowanie czytalnego datasetu dla pytorcha
class PepDataset(Dataset):
    def __init__(self,df,max_len=30):
        self.max_len = max_len
        self.sequences = df['Epitope - Name'].tolist()
        self.labels = df['assay'].astype(np.float32).tolist()
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]
        encode = wektorowanie(sequence, self.max_len)
        x = torch.tensor(encode, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.float32)
        
        return x, y

class Bodygoals(nn.Module):
    def __init__(self,input_size=600):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)    
        
    

    





