#machine learning
import tensorflow as tf
#image input 
import os
#possible image processing
import cv2
#PSNR calculation 
import numpy as np
#output 
import matplotlib.pyplot as plt

#load file
def load_file(path):
    images = []
    L = os.listdir(path)
    for filename in L:
        images.append(os.path.join(path,filename))
    return images

#return in np format 750 array
def image_slice(slices,img,h,w,overlap=0):
    
    image = cv2.imread(str(img)) #grayscale
    if image is None: return
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for r in range(123,598,w-overlap):
        for c in range(215,395,h-overlap):
            s = img[c:(c+h),r:(r+w)]
            s.astype(dtype="float32")
            slices.append(s.reshape((w*h)))
def image_merge(slices):
    num_r = 8
    num_c = 16
    
    image = [[]]
    for x in range(num_c):
        #print("c ",x)
        temp = np.reshape(slices[x*num_r],(25,30))
        #cv2.imshow("s",temp)
        cv2.waitKey(0)        
        #print(temp.shape)
        for y in range(1,num_r):
            #print(x*8+y)
            temp = np.concatenate((temp,np.reshape(slices[x*8+y],(25,30))),axis = 0)
        if(x == 0):image = temp
        else:image = np.concatenate((image,temp),axis = 1)
    #print(image.shape)
    return image      

#Data
hight = 25
width = 30
image_size = hight*width
x = tf.placeholder(tf.float32, shape=[None,image_size])
t = tf.placeholder(tf.float32, shape=[None,image_size])

#Model
#Size
h1 = 500
h2 = 250
h3 = 125
h4 = h2
h5 = h1
 
#Weight
initializer = tf.variance_scaling_initializer()
bias = 1
w1 = tf.Variable(initializer([image_size,h1]),dtype=tf.float32)
w2 = tf.Variable(initializer([h1,h2]),dtype=tf.float32)
w3 = tf.Variable(initializer([h2,h3]),dtype=tf.float32)
w4 = tf.Variable(initializer([h3,h4]),dtype=tf.float32)
w5 = tf.Variable(initializer([h4,h5]),dtype=tf.float32)
w6 = tf.Variable(initializer([h5,image_size]),dtype=tf.float32)
w10 = tf.Variable(tf.zeros(h1)+bias,dtype=tf.float32)
w20 = tf.Variable(tf.zeros(h2)+bias,dtype=tf.float32)
w30 = tf.Variable(tf.zeros(h3)+bias,dtype=tf.float32)
w40 = tf.Variable(tf.zeros(h4)+bias,dtype=tf.float32)
w50 = tf.Variable(tf.zeros(h5)+bias,dtype=tf.float32)
w60 = tf.Variable(tf.zeros(image_size)+bias,dtype=tf.float32)

#Layer
method = tf.nn.relu#tf.nn.sigmoid#tf.nn.relu
L1 = method(tf.matmul(x,w1)+w10)
L2 = method(tf.matmul(L1,w2)+w20)
L3 = method(tf.matmul(L2,w3)+w30)
L4 = method(tf.matmul(L3,w4)+w40)
L5 = method(tf.matmul(L4,w5)+w50)
y = method(tf.matmul(L5,w6)+w60)

#Train
Lr = 0.001
loss = tf.reduce_mean(tf.square(y-t))
optimizer=tf.train.AdamOptimizer(Lr)
train=optimizer.minimize(loss)

#Validation
PIXEL_MAX = tf.constant(255,dtype=tf.float32)
mse =  tf.reduce_mean(tf.square(y-t))
psnr = tf.multiply(tf.constant(20,dtype=tf.float32),tf.divide(tf.log(PIXEL_MAX/tf.math.square(mse)),tf.log(tf.constant(10,dtype=tf.float32))))



#data
pass_flag = False
train_x = load_file("./x")
size = len(train_x)
train_t = load_file("./t")
train_x_batch = []
train_t_batch = []

test_x = load_file("./tx")
tsize = len(test_x)
test_t = load_file("./tt")
test_x_batch = []
test_t_batch = []

if(len(train_t) != size or tsize != len(test_t)):
    print("Error: invalid train data {} != {}".format(size,len(train_t)))
    print("Error: invalid train data {} != {}".format(tsize,len(test_t)))
else:
    print("Processing....")
    for key in range(size):
        image_slice(train_x_batch,train_x[key],hight,width,5)
        image_slice(train_t_batch,train_t[key],hight,width,5)
        image_slice(test_x_batch,train_x[key],hight,width)
        image_slice(test_t_batch,train_t[key],hight,width)        
    if(len(train_x_batch) == len(train_t_batch) and len(train_x_batch) != 0 ):
        pass_flag = True
    else:
        print(train_x_batch)
np.array(train_x_batch)
np.array(train_t_batch)
#train

#Train 15 image, test 3 image, 1000 time
#print(train_t_batch[0])
#pass_flag = False
batch_flag = 0 #0: all together, 1: each slice, 2: each image
epoch = 1000
batch_size = size
#Result 
sample = []
Loss = []
Loss_Batch = [[0]*len(train_x_batch)]*epoch
#Loss_image = [[0]*(len(train_x_batch)/114)]*epoch
MSE = []
if pass_flag:
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        if(batch_flag == 0):
            for ti in range(epoch):
                sess.run(train,feed_dict={x:train_x_batch,t:train_t_batch})
                train_loss=loss.eval(feed_dict={x:train_x_batch,t:train_t_batch})
                Loss.append(train_loss)
                print("epoch {}: {}".format(ti,train_loss))
        if(batch_flag == 1):
            for ti in range(epoch):
                for batch in range(len(train_x_batch)):
                    sess.run(train,feed_dict={x:np.reshape(train_x_batch[0],(1,750)),t:np.reshape(train_t_batch[batch],(1,750))})
                    train_loss=loss.eval(feed_dict={x:np.reshape(train_x_batch[0],(1,750)),t:np.reshape(train_t_batch[batch],(1,750))})
                    Loss_batch[ti][batch] = train_loss
        #test        
        for b in range(len(test_x_batch)):
            output_sample = y.eval(feed_dict={x:np.reshape(test_x_batch[b],(1,750))})
            expect_sample = sess.run(y,feed_dict={x:np.reshape(test_x_batch[b],(1,750))})
            sample.append(expect_sample)
            result_loss=loss.eval(feed_dict={x:np.reshape(test_x_batch[b],(1,750)),t:np.reshape(test_t_batch[b],(1,750))})
            MSE.append(result_loss)
            
        print(MSE)
            

test_result = sample[:128]
ans = image_merge(test_result)
plt.imshow(ans,cmap='gray')
plt.savefig('ans.png')
plt.close()
#for i in range(1):
    #plt.imshow(np.reshape(train_t_batch[i],(25,30)),cmap='gray')
    #plt.imshow(sample[i],cmap='gray')
    #plt.close()
    
#ploting
plt.plot(list(range(epoch)),Loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('loss.png')
plt.close()

plt.bar(list(range(len(MSE))),MSE)
plt.xlabel('Slice')
plt.ylabel('loss')
plt.savefig('mse.png')
