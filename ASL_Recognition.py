import numpy as np
import cv2
import keras

model = keras.models.load_model('./asl_5.h5')
alphabet_dict = {}
for i in range(0,26):
  alphabet_dict[i] = chr(65 + i)

alphabet_dict.pop(9)
alphabet_dict.pop(25)
def hand_area(img):
    hand = img[100:324, 100:324]
    hand = cv2.resize(hand, (28,28))
    return hand
cap = cv2.VideoCapture(0)
p = True
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    # get the hand area on the video capture screen
    cv2.rectangle(frame, (100, 100), (324, 324), (20, 34, 255), 2)
    h = hand_area(frame)
    hand = h.copy()
    h = cv2.resize(h,(224,224))
    h = cv2.cvtColor(h,cv2.COLOR_BGR2GRAY)
    hand = cv2.cvtColor(hand,cv2.COLOR_BGR2GRAY)
    hand = hand/255
    rhand = hand.reshape((-1,28,28,1))
    out = model.predict(rhand)
    cv2.putText(frame, alphabet_dict[np.argmax(out)], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
    cv2.imshow('image', frame)
    cv2.imshow('hand', h)
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()