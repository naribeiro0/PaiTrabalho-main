import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Carregar o modelo treinado
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Iniciar captura de vídeo
cap = cv2.VideoCapture(0)

# Aguardar 2 segundos antes de iniciar a captura (opcional)
time.sleep(2)

# Verificar se a câmera está aberta corretamente
if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

# Configurar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)

# Dicionário de labels
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

recognized_letters = []  # Lista para armazenar as letras reconhecidas
previous_letter = ''  # Variável para armazenar a letra anterior

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)

                # Verificar se data_aux tem o comprimento esperado (42)
                if len(data_aux) != 42:
                    print(f"Erro: data_aux tem comprimento {len(data_aux)}, esperado 42. Pulando esta amostra.")
                    continue

                prediction = model.predict([data_aux])

                predicted_character = labels_dict[int(prediction[0])]
                
                # Adicionar a letra reconhecida à lista, evitando repetições consecutivas
                if not recognized_letters or recognized_letters[-1] != predicted_character:
                    recognized_letters.append(predicted_character)
                    previous_letter = recognized_letters[-2] if len(recognized_letters) > 1 else ''  # Atualiza a letra anterior

                cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())

        # Exibir a letra anterior reconhecida na janela do vídeo
        if previous_letter:
            cv2.putText(frame, "Letra anterior:", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, previous_letter, (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):  # Pressione 'q' para sair do loop
            break

    except Exception as e:
        print(f"Erro: {e}")

cap.release()
cv2.destroyAllWindows()

# Escrever as letras reconhecidas em um arquivo de texto
output_file = "recognized_letters.txt"
with open(output_file, 'w') as f:
    f.write("Letras reconhecidas durante a execução:\n")
    for letter in recognized_letters:
        f.write(f"{letter}")

print(f"Legenda das letras reconhecidas foi salva em '{output_file}'.")
