# realtime face recognition with deep learning in python

The code of a project about using deep-learning to realize the face recognition in my project group(4 people).

The detection of face is using OPENCV.

We need to prepare at least 5 photos of every person in the project (in this example so totally 5*4=20 photos) and then we use baseofimage.py to transform each photo randomly into 100 different photos, so totally 500 photos for each person. Then take the face part of each transformed photo, and apply LBP to them and save the new photos in the folder CNNdata.(You can change the path in the python code (baseofimage.py)as you want.)

Then we have totally 2000 images after LBP for training the model. Use train.py to train the model and the trained model is saved as model.h5 and model.json.

Phototest.py is for testing the face recognition between our 4 people by the photos. We prepared some other photos or the test.

Load.py is consisted by 3 parts. Print the accuracy by the test of random 10% images of CNNdata. Print the result of 5 special photos choosen from CNNData folder.(The fifth photo is by a person out of our group) And the real-time face recognition with a webcamara.

All the third party packages we used is here(with python2): keras,cv2,numpy,skimage,sklearn,tensorflow(ver.1.0.0+)
