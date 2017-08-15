# author: ZHANG wentong
# date: 2017.05.08
# email: wentong.zhang@groupe-esigelec.org
# code for building the data base of our model
# we prepare 5 photos of everyone, change them and apply LBP and face detection

import os
from os import listdir
from os.path import isfile, join

import cv2
import numpy
import numpy as np
from numpy import *
from random import randint

# import the photo of someone and change randomly for the base of images.
# here i take 5 photos for everyone, change them into 100 photos for each photo
# so totally 500 photos for everyone, totally 2000 photos as the input of CNN

def larger(img):
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(0, 24)

    pts1 = np.float32([[num, num], [cols - num, num], [num, rows - num], [cols - num, rows - num]])
    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    return dst

def smaller(img):
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(0, 24)

    pts1 = np.float32([[num, num], [cols - num, num], [num, rows - num], [cols - num, rows - num]])
    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(pts2, pts1)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    return dst

def lighter(img):
    # copy the basic picture, avoid the change of the basic one
    dst = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
    # get the number of rows and cols of picture
    rows, cols = dst.shape[:2]
    # take a random number to use
    num = randint(20,50)

    for xi in xrange(0, cols):
        for xj in xrange(0, rows):
            for i in range(0, 3):
                if dst[xj, xi, i] <= 255 - num:
                    dst[xj, xi, i] = int(dst[xj, xi, i] + num)
                else:
                    dst[xj, xi, i] = 255
    return dst

def darker(img):
    # copy the basic picture, avoid the change of the basic one
    dst = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(20,50)

    for xi in xrange(0, cols):
        for xj in xrange(0, rows):
            for i in range(0, 3):
                if dst[xj, xi, i] >= num:
                    dst[xj, xi, i] = int(dst[xj, xi, i] - num)
                else:
                    dst[xj, xi, i] = 0
    return dst

def moveright(img):
    # get the number of rows and cols of picture
    rows,cols = img.shape[:2]
    # take a random number to use
    num = randint(1, 2)

    M = np.float32([[1,0,num],[0,1,0]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def moveleft(img):
    # get the number of rows and cols of picture
    rows,cols = img.shape[:2]
    # take a random number to use
    num = randint(1, 2)

    M = np.float32([[1,0,-num],[0,1,0]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def movetop(img):
    # get the number of rows and cols of picture
    rows,cols = img.shape[:2]
    # take a random number to use
    num = randint(1, 2)

    M = np.float32([[1,0,0],[0,1,-num]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def movebot(img):
    # get the number of rows and cols of picture
    rows,cols = img.shape[:2]
    # take a random number to use
    num = randint(1, 2)

    M = np.float32([[1,0,0],[0,1,num]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def turnright(img):
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(3,6)

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -num, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def turnleft(img):
    # get the number of rows and cols of picture
    rows, cols = img.shape[:2]
    # take a random number to use
    num = randint(3,6)

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), num, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def changeandsave(name,time,choice,img,i):
    # the new name of changed picture is "changed?.png",? means it's the ?-th picture changed
    name = './changedphoto/' + str(name) + '/' +str(time) + '_changed' + str(i) + '.jpg'
    # do different changes by the choice
    if choice == 1:
        newimg = larger(img)
    elif choice == 2:
        newimg = smaller(img)
    elif choice == 3:
        newimg = lighter(img)
    elif choice == 4:
        newimg = darker(img)
    elif choice == 5:
        newimg = moveright(img)
    elif choice == 6:
        newimg = moveleft(img)
    elif choice == 7:
        newimg = movetop(img)
    elif choice == 8:
        newimg = movebot(img)
    elif choice == 9:
        newimg = turnleft(img)
    elif choice == 10:
        newimg = turnright(img)
    # save the new picture
    cv2.imwrite(name, newimg)


# take fu's 5 photos, change each photo into 100 photos, so totally 500
for j in range(1,6):
  img = cv2.imread('./fu_' + str(j) +'.jpg',1)
  # for cycle to make change randomly 100 times
  # (1,n), n for n-1 photos, (1,10), after change, 9 photos
  for i in range(1,101):
      # take a random number as the choice
      choice = randint(1,10)
      changeandsave('fu',j,choice,img,i)

# then the same for monica's photos
for j in range(1,6):
  img = cv2.imread('./monica_' + str(j) +'.jpg',1)
  # for cycle to make change randomly 100 times
  # (1,n), n for n-1 photos, (1,10), after change, 9 photos
  for i in range(1,101):
      # take a random number as the choice
      choice = randint(1,10)
      changeandsave('monica',j,choice,img,i)



# then the same for boris's photos
for j in range(1,6):
  img = cv2.imread('./boris_' + str(j) +'.jpg',1)
  # for cycle to make change randomly 100 times
  # (1,n), n for n-1 photos, (1,10), after change, 9 photos
  for i in range(1,101):
      # take a random number as the choice
      choice = randint(1,10)
      changeandsave('boris',j,choice,img,i)


# then the same for zhang's photos
for j in range(1,6):
  img = cv2.imread('./zhang_' + str(j) +'.jpg',1)
  # for cycle to make change randomly 100 times
  # (1,n), n for n-1 photos, (1,10), after change, 9 photos
  for i in range(1,101):
      # take a random number as the choice
      choice = randint(1,10)
      changeandsave('zhang',j,choice,img,i)



# and we need to use opencv to apply face detection and LPB of every image
# here we have totally 1500 photos, 500 for everyone

# the file of OPENCV for face detection
#path = '/home/pi/Desktop/opencv-2.4.10/data/haarcascades/'
path = '/home/user/opencv-3.1.0/data/haarcascades/'
face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')

# 2 fonctions for LBP
def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out


def get_pixel_else_0(l, idx, idy, default=0):
    try:
        return l[idx, idy]
    except IndexError:
        return default

# we apply LBP and face detection to all photos and save the new photos
for i in range(4,5):# that means i = 1,2,3,4
	if i == 1:
	    mypath = './changedphoto/fu'
	elif i == 2:
	    mypath = './changedphoto/monica'
	elif i == 3:
	    mypath = './changedphoto/zhang'
	elif i == 4:
	    mypath = './changedphoto/boris'

	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	images = numpy.empty(len(onlyfiles), dtype=object)
	for n in range(0, len(onlyfiles)):
	    images[n] = cv2.imread(join(mypath, onlyfiles[n]))
	    # transform the new image into GRAY
	    newgray = cv2.cvtColor(images[n], cv2.COLOR_BGR2GRAY)
	    # apply face detection to this image
	    faces = face_cascade.detectMultiScale(
		newgray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
                #flags = cv2.cv.CV_HAAR_SCALE_IMAGE#OPENCV version 2.x
		flags=cv2.CASCADE_SCALE_IMAGE #OPENCV version 3.x
	    )

	    for (x, y, w, h) in faces:
		x = x

	    cropped = newgray[y:y + h, x:x + w]
	    # here cropped is still in size of w*h, we need an image in 48*48, so change it
	    result = cv2.resize(cropped, (48, 48), interpolation=cv2.INTER_LINEAR)  # OPENCV 3.x

	    # then use LBP to this image
	    # copy result as transformed_img
	    transformed_img = cv2.copyMakeBorder(result, 0, 0, 0, 0, cv2.BORDER_REPLICATE)

	    for x in range(0, len(result)):
		for y in range(0, len(result[0])):
		    center = result[x, y]
		    top_left = get_pixel_else_0(result, x - 1, y - 1)
		    top_up = get_pixel_else_0(result, x, y - 1)
		    top_right = get_pixel_else_0(result, x + 1, y - 1)
		    right = get_pixel_else_0(result, x + 1, y)
		    left = get_pixel_else_0(result, x - 1, y)
		    bottom_left = get_pixel_else_0(result, x - 1, y + 1)
		    bottom_right = get_pixel_else_0(result, x + 1, y + 1)
		    bottom_down = get_pixel_else_0(result, x, y + 1)

		    values = thresholded(center, [top_left, top_up, top_right, right, bottom_right,
		                                  bottom_down, bottom_left, left])

		    weights = [1, 2, 4, 8, 16, 32, 64, 128]
		    res = 0
		    for a in range(0, len(values)):
		        res += weights[a] * values[a]

		    transformed_img.itemset((x, y), res)

	    # we only use the part (1,1) to (46,46) of the result img.
	    # original img: 0-47, after resize: 1-46
	    lbp = transformed_img[1:47, 1:47]  # here 1 included, 47 not included

	    # save the final image
	    if i == 1:
		name = './CNNdata/1_fu_' + str(n) + '.jpg'
	    elif i == 2:
		name = './CNNdata/2_monica_' + str(n) + '.jpg'
	    elif i == 3:
		name = './CNNdata/3_zhang_' + str(n) + '.jpg'
	    elif i == 4:
		name = './CNNdata/4_boris_' + str(n) + '.jpg'

	    cv2.imwrite(name, lbp)


