import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Carregar os dados
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Verificar a consist�ncia dos dados
data = data_dict['data']
labels = data_dict['labels']

# Verificar os comprimentos dos elementos em data
lengths = [len(item) for item in data]
max_length = max(lengths)

# Padronizar os dados para que todos os elementos tenham o mesmo comprimento
standardized_data = []
for item in data:
    if len(item) < max_length:
        item = np.pad(item, (0, max_length - len(item)), mode='constant')
    standardized_data.append(item)

# Converter para arrays NumPy
data = np.asarray(standardized_data)
labels = np.asarray(labels)

print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# Dividir os dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Inicializar o modelo
model = RandomForestClassifier()

# Treinar o modelo
model.fit(x_train, y_train)

# Fazer previs�es
y_train_predict = model.predict(x_train)
y_test_predict = model.predict(x_test)

# Calcular acur�cia
train_score = accuracy_score(y_train_predict, y_train)
test_score = accuracy_score(y_test_predict, y_test)

print(f'Train Accuracy: {train_score * 100:.2f}%')
print(f'Test Accuracy: {test_score * 100:.2f}%')

# Salvar o modelo
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
