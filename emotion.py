import cv2
import imutils
import time
import numpy as np
from keras.models import load_model

cap = cv2.VideoCapture(0)

cap.set(3, 480) #set width
cap.set(4, 640) #set height

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)',' (13,16)','(17, 20)','(21, 24)', '(25, 32)','(33,37 )', '(38, 43)', '(48, 53)', '(60, 70)','(75, 85)','(90, 100)']
gender_list = ['Male', 'Female']
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

def initialize_caffe_models():
	
	age_net = cv2.dnn.readNetFromCaffe(
		'data/deploy_age.prototxt', 
		'data/age_net.caffemodel')

	gender_net = cv2.dnn.readNetFromCaffe(
		'data/deploy_gender.prototxt', 
		'data/gender_net.caffemodel')

	return(age_net, gender_net)

def read_from_camera(age_net, gender_net):
	font = cv2.FONT_HERSHEY_SIMPLEX

	while True:

		ret, image = cap.read()

		

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.1, 5)

		if(len(faces)>0):
			print("Found {} faces".format(str(len(faces))))

		for (x, y, w, h )in faces:
			cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)

			# Get Face 
			face_img = image[y:y+h, h:h+w].copy()
			blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

			#Predict Gender
			gender_net.setInput(blob)
			gender_preds = gender_net.forward()
			gender = gender_list[gender_preds[0].argmax()]
			print("Gender : " + gender)

			#Predict Age
			age_net.setInput(blob)
			age_preds = age_net.forward()
			age = age_list[age_preds[0].argmax()]
			print("Age Range: " + age)

			overlay_text = "%s %s" % (gender, age)
			cv2.putText(image, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)


		cv2.imshow('frame', image)
		
                        
	
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == "__main__":
	age_net, gender_net = initialize_caffe_models()

	read_from_camera(age_net, gender_net)


# pre - trinaed xml file for detecting faces
#facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX


# loading saved cnn model
model = load_model('face_reco.h5')

# predicting face emotion using saved model
def get_emo(im):
    im = im[np.newaxis, np.newaxis, :, :]
    res = model.predict_classes(im,verbose=0)
    emo = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    return emo[res[0]]


# reshaping face image
def recognize_face(im):
    im = cv2.resize(im, (48, 48))
    return get_emo(im)

while True:
    _, fr = cap.read()
    flip_fr = cv2.flip(fr,1)
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        fc = fr[y:y+h, x:x+w, :]
        gfc = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
        out = recognize_face(gfc)
        cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
        flip_fr = cv2.flip(fr,1)
        cv2.putText(flip_fr, out, (30, 30), font, 1, (255, 255, 0), 2)
    
    cv2.imshow('frame', flip_fr)

    # press esc to close the window
    k = cv2.waitKey(1) & 0xEFFFFF
    if k==27:   
        break
    elif k==-1:
        continue
    else:
        # print k
        continue

cv2.destroyAllWindows()




