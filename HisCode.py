#Import bibliotek (w tym tych przydatnych potem):
import torch
import torch.nn as nn #Do wczytania gotowej implementacji MLP
import torch.optim as optim #Do wykorzystania optymalizatora
from torch.utils.data import Dataset, DataLoader #Klasy obsługujące dataset
import pandas as pd #wstępna obróbka datasetu
import numpy as np
from sklearn.model_selection import train_test_split #Przyda się do dzielenia zbioru na treningowy i testowy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #Gotowe funkcje do wyznaczania klasycznych metryk przy klasyfikacji

# 2) Wczytanie naszego datasetu
df = pd.read_csv("nasz dataset")

#Przefiltrowanie datasetu tak, żeby zostało tylko 20 aminokwasów:
aminokwasy = set(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
def czy_nasza_sekwencja_ma_tylko_kodujace_aminokwasy(sekwencja):
    return all(aa in aminokwasy for aa in sekwencja)

df = df[df["sequence"]].apply(czy_nasza_sekwencja_ma_tylko_kodujace_aminokwasy)]

#Wyrzucenie wszystkich sekwencji dłuższych od 30
df = df.loc[df.str.len() <=30]


#3) Zdefiniowanie jakiejś funkcji zaplikującej one-hot-encoding
def one_hot_encode(sekwencja, max_len=30):
    #-jakiś kod-
    return #1D_wektor_o_dlugosci_600_z_naszym_encodingiem
'''
4)
Tutaj dochodzimy do dosyć kluczowej sprawy, czyli przygotowania naszego zbioru danych tak, aby mógł być wykorzystany przez PyTorch. Aby to zrobić musimy zdefiniować klasę, dziedziczącą po PyTorchowej klasie "Dataset" zapewniającej interfejs dostępu do naszych danych. W tym wypadku musimy tak naprawdę zdefiniować tylko dwie metody __len__ i __getitem__
W naszym przypadku może to wyglądać następująco:
'''

class PeptideDataset(Dataset):
    def __init__(self, df, max_len=30): 
        #Metoda inicjująca
        self.sequences = [one_hot_encode(seq, max_len) for seq in df['sequence']]
        self.labels = df['activity'].astype(np.float32).values

    def __len__(self):
        #Metoda zwracająca wielkość zbioru
        return len(self.sequences) 

    def __getitem__(self, idx):
        #Metoda zwracająca pare sekwencja - label dla indeksu o numerze idx
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx])
'''
5)
Na tym etapie możemy zdefiniować nasz model sieci neuronowej. Załóżmy, że chcemy zrobić model którego warstwa wejściowa będzie miała 600 neuronów które łączą się z warstwą 128 neuronów i przechodzą przez funkcje aktywacji ReLU, następnie łączą się z warstwą 64 neuronów i przechodzą przez funkcje aktywacji ReLU, po czym przechodzą do jednego neuronu z funkcją aktywacji sigmoid (ten neuron odpowiada za predykcje, jeżeli chcielibyśmy przewidywać wiele klas, moglibyśmy użyć funkcji aktywacji softmax, tutaj jednak sigmoid powinien być odpowiedni).
W praktyce będzie to wyglądało następująco, definiujemy klasą dziedziczącą po pytorchowej klasie nn.Module i definiujemy model warstwa po warstwie:
'''

class MLPClassifier(nn.Module):
    def __init__(self, input_size=600):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

#W tej samej klasie definiujemy również metodę propagacji wprzód:
    def forward(self, x):
        return self.model(x)


#6) To jest chyba najtrudniejsza część zadania - napisanie funkcji trenującej.

def funkcja_trenujaca(df):
# Zdefiniowanie datasetu
    dataset = PeptideDataset(df, max_len=30) 
# Wykorzystanie pytorchowego dataloadera - narzędzia do wyciągania z naszego datasetu danych w "paczkach" (batchach o określonej przez nas wielkości, parametr shuffle=True sprawia, że paczki będą wyciągane w losowej kolejności) i ładującego ich do naszego modelu.
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#Zdefiniowanie niezbędnych narzędzi
    input_size=600
    model = MLPClassifier(input_size) # zdefiniowanie naszego modelu jako tego który będzie trenowany
    criterion = nn.BCELoss()  # Zdefiniowanie w jaki sposób będzie obliczana funkcja straty, tutaj Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Zdefiniowanie algorytmu optymalizującego wagi, w tym wypadku Adam ze współczynnikiem uczenia 0.001

#Sama pętla treningowa jest w gruncie rzeczy zagnieżdżoną pętlą - trening odbywa się w cyklach nazwanych epokami, gdzie każda epoka (w tym wypadku jest ich 10) to po prostu przejście naszego algorytmu trenującego przez cały dataset, podzielony na batche

    for epoch in range(epochs=10):
        model.train() #wchodzimy w tryb treningu
        running_loss = 0.0 #Inicjalizacja zmiennej akumulującej wynik funkcji straty przez każdy mini-batch, potem będzie to uśrednione na każdą epoke.
        for x_batch, y_batch in dataloader: 
            optimizer.zero_grad() #WYZEROWANIE GRADIENTÓW, PYTORCH NIE ROBI TEGO AUTOMATYCZNIE
            outputs = model(x_batch) #propagacja w przód, obliczenie wyników takiego działania
            loss = criterion(outputs.view(-1), y_batch) #Obliczenie funkcji straty (błędu między obliczoną wartością w poprzednim kroku a spodziewanym wynikiem), .view(-1) zamienia outputs w 1D wektor pasujący kształtem do y_batch
            loss.backward() #Propagacja wsteczna, obliczenie gradientu
            optimizer.step() #Aktualizacja parametrów modelu z wykorzystaniem gradientów obliczonych w poprzednim kroku
            running_loss += loss.item() # aktualizacja błędu
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}") # Po zakończeniu obliczania wszystkich batchy, obliczamy średni wynik funkcji straty.

    torch.save(model.state_dict(), model_path="Nasz_model.pt") #Zapisanie wag naszego modelu,
'''
Teraz można taką funkcje wywołać i zacząć trening. Jeżeli chcemy potem wyznaczyć metryki  accuracy_score, precision_score, recall_score, f1_score służące do oceny skuteczności modelu, trzeba najpierw podzielić dataset na zbiór treningowy i testowy.
'''

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42) #train_test split jest funkcją ze scikit_learn
train_model(df_train, epochs=10) #Zaczynamy trenowanie



#7) następnie można napisać funkcje która wczytuje nasz zapisany model razem z wagami, wczytuje od nas sekwencje i wykonuje predykcje

def evaluate_model(model_path="Nasz_model.pt", max_len=30):
    input_size = max_len * len(aminokwasy) #600
    model = MLPClassifier(input_size) #Wczytanie zdefiniowanego przez nas modelu
    model.load_state_dict(torch.load(model_path)) #Wczytanie wytrenowanych przez nas wag
    model.eval() #wchodzimy w tryb ewaluacji

    seq = input("Enter amino acid sequence: ").strip().upper() #pobieramy od użytkownika sekwencje
    encoded = one_hot_encode(seq, max_len) #Aplikujemy do tego sekwencji one-hot-encoding
    x = torch.tensor(encoded).unsqueeze(0) #Zamieniamy zakodowaną sekwencje na tensor pytorchowy

    with torch.no_grad():
        output = model(x).item() #Obliczamy wynik przejścia propagacji w przód dla naszego tensora x (czyli tak naprawdę to co kryje się za ostatnim neuronem i funkcją sigmoid)
        prediction = 1 if output >= 0.5 else 0 #Jeżeli wynik takiego przejścia jest większy od 0.5 to peptyd jest immunogenny jak mniejszy od 0.5 to nie jest
        print(f"Predicted activity: {prediction} (probability: {output:.4f})")