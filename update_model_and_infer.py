import os
import cv2
import shutil
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mediapipe as mp

# Configurar MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = len([name for name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, name))])
dataset_size = 100

cap = cv2.VideoCapture(0)

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def capture_and_save_new_class(class_dir):
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for new class {}'.format(class_dir))

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

def load_data():
    data = []
    labels = []
    for dir_ in os.listdir(DATA_DIR):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(int(dir_))  # Convertendo diretório para int (classe)

    return np.asarray(data), np.asarray(labels)

def train_model():
    global number_of_classes

    # Carregar os dados existentes
    data, labels = load_data()

    # Dividir os dados em conjuntos de treinamento e teste
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    # Inicializar o modelo
    model = RandomForestClassifier()

    # Treinar o modelo
    model.fit(x_train, y_train)

    # Fazer previsões
    y_train_predict = model.predict(x_train)
    y_test_predict = model.predict(x_test)

    # Calcular acurácia
    train_score = accuracy_score(y_train_predict, y_train)
    test_score = accuracy_score(y_test_predict, y_test)

    print(f'Train Accuracy: {train_score * 100:.2f}%')
    print(f'Test Accuracy: {test_score * 100:.2f}%')

    # Salvar o modelo
    with open('model.p', 'wb') as f:
        pickle.dump({'model': model}, f)

    number_of_classes += 1

    # Atualizar o dicionário de labels
    labels_dict = {i: chr(65 + i) for i in range(number_of_classes)}

    return labels_dict, model

def inference(labels_dict, model):
    # Configurar MediaPipe para detecção de mãos
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)

    # Carregar o vídeo
    video_path = 'path/to/your/video.mp4'  # Substitua pelo caminho do seu vídeo
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Obter dimensões do frame
        H, W, _ = frame.shape

        # Converter o frame para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processar o frame para detecção de mãos
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                # Desenhar as landmarks da mão no frame
                mp_drawing.draw_landmarks(
                    frame,  # Imagem para desenhar
                    hand_landmarks,  # Saída do modelo
                    mp_hands.HAND_CONNECTIONS,  # Conexões da mão
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Coletar coordenadas das landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalizar coordenadas
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Definir retângulo ao redor da mão detectada
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Fazer a previsão da letra
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Desenhar retângulo e texto no frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Mostrar o frame
        cv2.imshow('frame', frame)
        
        # Fechar a janela se 'q' for pressionado
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar captura e fechar janelas
    cap.release()
    cv2.destroyAllWindows()

def main():
    global number_of_classes

    # Capturar uma nova classe se solicitado
    new_class_dir = os.path.join(DATA_DIR, str(number_of_classes))
    capture_and_save_new_class(new_class_dir)
    number_of_classes += 1

    # Treinar o modelo com a nova classe
    labels_dict, model = train_model()

    # Executar inferência com o modelo atualizado
    inference(labels_dict, model)

if __name__ == "__main__":
    main()
