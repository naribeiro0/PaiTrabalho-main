import pickle
import cv2
import mediapipe as mp
import numpy as np

# Carregar o modelo treinado
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Configurar MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)

# Dicionário de labels
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Extra'}

# Caminho para o vídeo
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
