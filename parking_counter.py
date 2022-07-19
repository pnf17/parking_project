#import library
import cv2
import pickle

#import file video
cap = cv2.VideoCapture('parking_lot_1.mp4')

#import file hasil running "parking_space_picker.py"
with open('park_positions', 'rb') as f:
    park_positions = pickle.load(f)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

#Parameter objek yang akan dideteksi
width, height = 40, 19
full = width * height
empty = 0.09

#metode penghitungan tempat parkir tersedia
def parking_space_counter(img_processed):

    global counter

    counter = 0


    for position in park_positions:
        x, y = position

        img_crop = img_processed[y:y + height, x:x + width]
        count = cv2.countNonZero(img_crop)

        ratio = count / full

        if ratio < empty:
            color = (0, 255, 0)
            counter += 1
        else:
            color = (0, 0, 255)

        cv2.rectangle(overlay, position, (position[0] + width, position[1] + height), color, -1)
        cv2.putText(overlay, "{:.2f}".format(ratio), (x + 4, y + height - 4), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


while True:

    # Video looping
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    _, frame = cap.read()
    overlay = frame.copy()

    #Convert ke grayscale
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Convert ke filter gaussian
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
    
    #Objek citra dilakukan thresholding
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)

    parking_space_counter(img_thresh)

    alpha = 0.7
    frame_new = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    #metode untuk menampilkan counter pada frame
    w, h = 220, 60
    cv2.rectangle(frame_new, (0, 0), (w, h), (255, 0, 255), -1)
    cv2.putText(frame_new, f"Kosong : {counter}/{len(park_positions)}", (int(w / 10), int(h * 3 / 4)), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow('frame', frame_new)
    # cv2.imshow('image_blur', img_blur)
    # cv2.imshow('image_thresh', img_thresh)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
