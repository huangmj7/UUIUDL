#absolute value
#increase layer size 
#normalize

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
      
#PSNR source: https://dsp.stackexchange.com/questions/38065/peak-signal-to-noise-ratio-psnr-in-python-for-an-image?rq=1


#Data
hight = 10
width = 10
image_size = hight*width
x = tf.placeholder(tf.float32, shape=[None,image_size])
t = tf.placeholder(tf.float32, shape=[None,image_size])

#Size
h1 = 300
h2 = 400
h3 = h1

#Weight
uniform = 0.000000000
initializer = tf.variance_scaling_initializer(distribution='normal')
w1 = tf.Variable(initializer([image_size,h1])+uniform,dtype=tf.float32)
w2 = tf.Variable(initializer([h1,h2])+uniform,dtype=tf.float32)
w3 = tf.Variable(initializer([h2,h3])+uniform,dtype=tf.float32)
w4 = tf.Variable(initializer([h3,image_size])+uniform,dtype=tf.float32)
bias = 1
w10 = tf.Variable(tf.zeros(h1)+bias,dtype=tf.float32)
w20 = tf.Variable(tf.zeros(h2)+bias,dtype=tf.float32)
w30 = tf.Variable(tf.zeros(h3)+bias,dtype=tf.float32)
w40 = tf.Variable(tf.zeros(image_size),dtype=tf.float32)

#Layer
method = tf.nn.relu#tf.nn.sigmoid#tf.nn.relu
L1 = method(tf.matmul(x,w1)+w10)
L2 = method(tf.matmul(L1,w2)+w20)
L3 = method(tf.matmul(L2,w3)+w30)
y = method(tf.matmul(L3,w4)+w40)

#Train
Lr = 0.001
loss = tf.reduce_mean(tf.square(y-t))
optimizer=tf.train.AdamOptimizer(Lr)
train=optimizer.minimize(loss)

#Validation
PIXEL_MAX = tf.constant(255,dtype=tf.float32)
mse =  tf.reduce_mean(tf.square(y-t))
psnr = tf.multiply(tf.constant(20,dtype=tf.float32),tf.divide(tf.log(PIXEL_MAX/tf.math.square(mse)),tf.log(tf.constant(10,dtype=tf.float32))))



#Input
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
                print("{} {}".format(ti,train_loss))
        if(batch_flag == 1):
            for ti in range(epoch):
                for batch in range(len(train_x_batch)):
                    sess.run(train,feed_dict={x:np.reshape(train_x_batch[0],(1,image_size)),t:np.reshape(train_t_batch[batch],(1,image_size))})
                    train_loss=loss.eval(feed_dict={x:np.reshape(train_x_batch[0],(1,image_size)),t:np.reshape(train_t_batch[batch],(1,image_size))})
                    Loss_batch[ti][batch] = train_loss
        #test        
        for b in range(len(test_x_batch)):
            output_sample = y.eval(feed_dict={x:np.reshape(test_x_batch[b],(1,image_size))})
            expect_sample = sess.run(y,feed_dict={x:np.reshape(test_x_batch[b],(1,image_size))})
            sample.append(expect_sample)
            result_loss=loss.eval(feed_dict={x:np.reshape(test_x_batch[b],(1,image_size)),t:np.reshape(test_t_batch[b],(1,image_size))})
            MSE.append(result_loss)
            
        #print(MSE)
            

test_result = sample[:960]
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
plt.savefig('test_result.png')
plt.close()


#output_sample = y.eval(feed_dict={x:train_x_batch})
#expect_sample = t.eval(feed_dict={t:train_t_batch})
#sample_mse = mse.eval(feed_dict={x:train_x_batch,t:train_t_batch})
#sample_psnr = psnr.eval(feed_dict={x:train_x_batch,t:train_t_batch})
        

      
      
#Testing 
#Output