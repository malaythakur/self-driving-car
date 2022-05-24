import tensorflow as tf
import scipy.music
import model
import cv2
from subprocess import call
import math

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = cv2.imread('steer.jpg',0)
rows,cols = img.shape

smoothed_angle = 0

xs = []
ys = []

with open("D:/self-driving-car/Autopilot-TensorFlow-master/driving_dataset/data.txt") as f:
    for line in f:
        xs.append("D:/self-driving-car/Autopilot-TensorFlow-master/driving_dataset/ " + line.split()[0])
        ys.append(float (line.split()[1]) * scipy.pi/180)
 
num_images = len(xs)

i = math.ceil(num_images = 0.8)
print("Starting frameofvideo: " +str(i))

while(cv2.waitKey(10) != ord('q')):
    full_image = scipy.misc.imread('D:/self-driving-car/Autopilot-TensorFlow-master/driving_dataset/' + str(i) + ".jpg", mode = 'RGB')
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi

    print("steering angle: "+str(degrees)+ "(pred)\t" +str(ys[i] * 180 / scipy.pi)+ "(actual")
    cv2.imshow("freme",cv2.cvt.cvtColor(full_image,cv2.COLOR_RGB2BGR))

    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    
    M = cv2.getRotationalMatrix2D((cols/2,rows/2), -smothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1

cv2.destroyAllWindows()
