import os
import cv2
import shutil

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 25
dataset_size = 100

cap = cv2.VideoCapture(0)

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Pronto? Pressione "Q" para iniciar, "R" para reiniciar a classe', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        if key == ord('q'):
            break
        elif key == ord('r'):
            clear_directory(class_dir)
            print('Diretório limpo. Pronto para coletar dados para a classe {}'.format(j))

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
