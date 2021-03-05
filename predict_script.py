# import tensorflow
from tensorflow.keras.preprocessing import image
# import matplotlib.pyplot as plt
# import numpy as np



from tensorflow.keras.models import load_model
classifier = load_model('model_cat_dog_grayscale.h5')


def predict_video(frame):
	count_frames = 0
	frame_list = []

	img = image.img_to_array(frame)
	img = img/255
	img = np.expand_dims(img, axis=0)
	
	
	#cv2.imshow('Video',img)

	#print('Dog')
	prediction = classifier.predict(img, batch_size=None,steps=1) #gives all class prob.
	if(prediction[:,:]>0.5):
	    return 'Cat'
	else:
	    return 'Dog'

import cv2,time
import numpy as np

cap = cv2.VideoCapture('dogcat.gif')
while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	time.sleep(0.1)
	frame_pred = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_pred = cv2.resize(frame_pred,(150,150))

	texto = predict_video(frame_pred)
	frame = cv2.resize(frame,(750,500),interpolation=cv2.INTER_AREA)
	frame = cv2.putText(frame,str(texto),(0,130), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 3, cv2.LINE_AA)

	# Display the resulting frame
	cv2.imshow('Video',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()