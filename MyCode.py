import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader 
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#praise Lana cause without her this shi wouldnt work

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

#stworzenie modelu (ktory zadziala za 1 razem)
class Bodygoals(nn.Module):
    def __init__(self,input_size=600):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128), #normalizacja, ktora wspiera stabilizacje i przyspiszenia uczenie - kinda cool ngl
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)    
        
#budowanie nadzorcy dla goata    
def locked_in(model, dataset, batch_size=32, epochs=20, lr=0.001):
    dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=True)
    crit = nn.BCEWithLogitsLoss() #dziala szybciej (nie musi byc sigmoid juz liczony w modelu)
    poprawiacz = optim.Adam(model.parameters(),lr=lr)
    model.to(device)
    
    print('OKAYYYY LETS GO')
    for epoch in range(epochs):
        model.train()
        biegnaca_strata = 0.0
        
        for i, (x_batch,y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            poprawiacz.zero_grad()
            outputs = model(x_batch)
            outputs = outputs.view(-1)
            loss = crit(outputs, y_batch)
            loss.backward()
            poprawiacz.step()
            biegnaca_strata += loss.item()
        avg_loss = biegnaca_strata / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "modelImmu.pt")
    print("Model zapisany")

#budowanie egzaminatora
def evaluate_model(model, dataset, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size)  
    model.eval()     
    model.to(device)
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            outputs = torch.sigmoid(model(x_batch)).view(-1) #handluje ten nowy zrobiony loss, zeby dalej bylo [0,1]

            predicted = (outputs >= 0.5).float()
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("Ewaluacja modelu:")
    print(f"Accuracy :  {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall   :  {rec:.4f}")
    print(f"F1 Score :  {f1:.4f}")


#przygotowanie danych do treningu
df = pd.read_csv(r'C:\\Studia\\Progranmy\\Adam\\valid_sequences.csv')
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = PepDataset(df_train)
test_dataset = PepDataset(df_test)


#2 year time skip type shi
model = Bodygoals(input_size=600)
locked_in(model, train_dataset, batch_size=32, epochs=20, lr=0.001)
'''
#Sabody arc
model = Bodygoals(input_size=600)
model.load_state_dict(torch.load("C:\\Studia\\Progranmy\\Adam\\modelImmu.pt", map_location=device, weights_only=True))
model.to(device)
'''

#matura
#evaluate_model(model, test_dataset)

#we can finnaly use this fker
def Venator(model, sequence, max_len=30):
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    if not isinstance(sequence, str) or any(aa not in valid_aa for aa in sequence):
        print("Błędna sekwencja: zawiera niepoprawne znaki.")
        return

    if len(sequence) > max_len:
        print(f"Błędna długość: sekwencja ma {len(sequence)}, a dopuszczalne to {max_len}")
        return

    encoded = wektorowanie(sequence, max_len)
    x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(x).item()

    print(f"Sekwencja: {sequence}")
    print(f"Probability: {output:.4f}")
    if output >= 0.5:
        print("Model przewiduje: AKTYWNA (1)")
    else:
        print("Model przewiduje: NIEAKTYWNA (0)")

#Venator(model, "GIDSSHPDLKVAGGA")



#to do:
#1 comment everything
#2 relearn venator