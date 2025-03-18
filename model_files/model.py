import cv2
import numpy as np
from keras.models import load_model


def predict(l_eye, r_eye):

    # loading our model
    model = load_model('./modell.h5')
    rpred = [99]  # for taking class prediction of right eye
    lpred = [99]  # for taking class prediction of left eye

    rpred = model.predict(r_eye)
    rpred = np.argmax(rpred, axis=1)
    lpred = model.predict(l_eye)
    lpred = np.argmax(lpred, axis=1)

    if(rpred[0] == 0 and lpred[0] == 0):  # if both eyes are closed incerement the score
        return True
    else:  # if eyes are not closed decrement the score
        return False
