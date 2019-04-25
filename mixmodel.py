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


#return in np format 100 array
def image_slice(slices,img,h,w,overlap=0):
    
    image = cv2.imread(str(img)) #grayscale
    if image is None: return
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    i = 0
    for r in range(123,603,(w-overlap)):
        for c in range(215,415,(h-overlap)):
            #if(True):cv2.rectangle(img,(r,c), (r+w,c+h), (255,255,255),1)
            #if((603-r) == (415-c)):cv2.rectangle(img,(r,c), (r+w,c+h), (255,255,255),1)
            s = img[c:(c+h),r:(r+w)]
            #print(s.shape)
            s.astype(dtype="float32")
            slices.append(s.reshape((w*h)))
            i+=1
    #print(len(slices))
    #print(r," ",c)
    return img
#20*48
def image_merge(slices,num_r=0,num_c=0):
    num_r = 20#8
    num_c = 48#16
    
    image = [[]]
    for x in range(num_c):
        #print("c ",x)
        temp = np.reshape(slices[x*num_r],(10,10))
        #cv2.imshow("s",temp)
        cv2.waitKey(0)        
        #print(temp.shape)
        for y in range(1,num_r):
            #print(x*8+y)
            temp = np.concatenate((temp,np.reshape(slices[x*num_r+y],(10,10))),axis = 0)
        if(x == 0):image = temp
        else:image = np.concatenate((image,temp),axis = 1)
    #print(image.shape)
    return image    

#Data
hight = 10
width = 10
image_size = hight*width
x = tf.placeholder(tf.float32, shape=[None,image_size])
t = tf.placeholder(tf.float32, shape=[None,image_size])

#Size
h1 = 50
h2 = 25
h3 = h1

bias = 1
#Weight I
initializer = tf.variance_scaling_initializer(seed = 12)
w1I = tf.Variable(initializer([image_size,h1]),dtype=tf.float32)
w2I = tf.Variable(initializer([h1,h2]),dtype=tf.float32)
w3I = tf.Variable(initializer([h2,h3]),dtype=tf.float32)
w4I = tf.Variable(initializer([h3,image_size]),dtype=tf.float32)
bias = 1
w10I = tf.Variable(tf.zeros(h1)+bias,dtype=tf.float32)
w20I = tf.Variable(tf.zeros(h2)+bias,dtype=tf.float32)
w30I = tf.Variable(tf.zeros(h3)+bias,dtype=tf.float32)
w40I = tf.Variable(tf.zeros(image_size),dtype=tf.float32)


#Layer I
method = tf.nn.relu#tf.nn.sigmoid#tf.nn.relu
IL1 = method(tf.matmul(x,w1I)+w10I)
IL2 = method(tf.matmul(IL1,w2I)+w20I)
IL3 = method(tf.matmul(IL2,w3I)+w30I)
Iy = method(tf.matmul(IL3,w4I)+w40I)

#Weight II
#initializer = tf.initializers.random_uniform()
initializer = tf.variance_scaling_initializer(seed = 1)
w1II = tf.Variable(initializer([image_size,h1]),dtype=tf.float32)
w2II = tf.Variable(initializer([h1,h2]),dtype=tf.float32)
w3II = tf.Variable(initializer([h2,h3]),dtype=tf.float32)
w4II = tf.Variable(initializer([h3,image_size]),dtype=tf.float32)
bias = 1
w10II = tf.Variable(tf.zeros(h1)+bias,dtype=tf.float32)
w20II = tf.Variable(tf.zeros(h2)+bias,dtype=tf.float32)
w30II = tf.Variable(tf.zeros(h3)+bias,dtype=tf.float32)
w40II = tf.Variable(tf.zeros(image_size),dtype=tf.float32)


#Layer II
IIL1 = method(tf.matmul(x,w1II)+w10II)
IIL2 = method(tf.matmul(IIL1,w2II)+w20II)
IIL3 = method(tf.matmul(IIL2,w3II)+w30II)
IIy = method(tf.matmul(IIL3,w4II)+w40II)

Finaly = tf.multiply(tf.add(Iy,IIy),0.5)

#Train
Lr = 0.001
lossI = tf.reduce_mean(tf.square(Iy-t))
lossII = tf.reduce_mean(tf.square(IIy-t))
loss = tf.reduce_mean(tf.square(Finaly-t))
optimizer=tf.train.AdamOptimizer(Lr)
trainI=optimizer.minimize(lossI)
trainII=optimizer.minimize(lossII)

#Validation
PIXEL_MAX = tf.constant(255,dtype=tf.float32)
mse =  tf.reduce_mean(tf.square(Finaly-t))
psnr = tf.multiply(tf.constant(20,dtype=tf.float32),tf.divide(tf.log(PIXEL_MAX/tf.math.square(mse)),tf.log(tf.constant(10,dtype=tf.float32))))

#data
#pass_flag = False
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
        image_slice(train_x_batch,train_x[key],hight,width,2)
        image_slice(train_t_batch,train_t[key],hight,width,2)
        image_slice(test_x_batch,train_x[key],hight,width)
        image_slice(test_t_batch,train_t[key],hight,width)        
    if(len(train_x_batch) == len(train_t_batch) and len(train_x_batch) != 0 ):
        pass_flag = True
    else:
        print(train_x_batch)
np.array(train_x_batch)
np.array(train_t_batch)
file = open("loss.txt","w")
#print(train_t_batch[0])
#pass_flag = False
batch_flag = 0 #0: all together, 1: each slice, 2: each image
epoch = 1000
batch_size = size
#Result 
sample = []
LossI = []
LossII = []
Loss_Batch = [[0]*len(train_x_batch)]*epoch
MSE = []
if pass_flag:
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        if(batch_flag == 0):
            for ti in range(epoch):
                sess.run(trainI,feed_dict={x:train_x_batch,t:train_t_batch})
                sess.run(trainII,feed_dict={x:train_x_batch,t:train_t_batch})
                train_loss_I =lossI.eval(feed_dict={x:train_x_batch,t:train_t_batch})
                train_loss_II =lossII.eval(feed_dict={x:train_x_batch,t:train_t_batch})
                LossI.append(train_loss_I)
                LossII.append(train_loss_II)
                print("epoch {}: {} {}".format(ti,train_loss_I,train_loss_II))
                file.write("{} {} {}".format(ti,train_loss_I,train_loss_II))
        if(batch_flag == 1):
            for ti in range(epoch):
                for batch in range(len(train_x_batch)):
                    sess.run(train,feed_dict={x:np.reshape(train_x_batch[0],(1,image_size)),t:np.reshape(train_t_batch[batch],(1,image_size))})
                    train_loss=loss.eval(feed_dict={x:np.reshape(train_x_batch[0],(1,image_size)),t:np.reshape(train_t_batch[batch],(1,image_size))})
                    Loss_batch[ti][batch] = train_loss
        #test        
        for b in range(len(test_x_batch)):
            output_sample = Finaly.eval(feed_dict={x:np.reshape(test_x_batch[b],(1,image_size))})
            expect_sample = sess.run(Finaly,feed_dict={x:np.reshape(test_x_batch[b],(1,image_size))})
            sample.append(expect_sample)
            result_loss=loss.eval(feed_dict={x:np.reshape(test_x_batch[b],(1,image_size)),t:np.reshape(test_t_batch[b],(1,image_size))})
            MSE.append(result_loss)
            
        #print(MSE)
            
file.close()
test_result = sample[:960]
ans = image_merge(test_result)
plt.imshow(ans,cmap='gray')
plt.savefig('output.png')
plt.close()
#for i in range(1):
    #plt.imshow(np.reshape(train_t_batch[i],(25,30)),cmap='gray')
    #plt.imshow(sample[i],cmap='gray')
    #plt.close()
    
#ploting
plt.plot(list(range(epoch)),LossI)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('loss1.png')
plt.close()

plt.plot(list(range(epoch)),LossII)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('loss2.png')
plt.close()

plt.bar(list(range(len(MSE))),MSE)
plt.xlabel('Slice')
plt.ylabel('loss')
plt.savefig('mse.png')
plt.close()
