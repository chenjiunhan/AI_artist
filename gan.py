import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np

result_dir = "result/"
RESTORE = False
TEST = False

mnist = np.load('images.npy')
#mnist = np.load('cifar.npy')

I_H = 480
I_W = 640

F_I_H = int(I_H / 8)
F_I_W = int(I_W / 8)

print('---------load data successful-----------')
def input_placeholder(img_size,noise_size):
    img=tf.placeholder(dtype=tf.float32,shape=(None,img_size[1],img_size[2],img_size[3]),name='input_image')
    noise=tf.placeholder(dtype=tf.float32,shape=(None,noise_size),name='input_noise')
    return img,noise

def generator(noise_img, output_dim, is_train=True, alpha=0.01):

    with tf.variable_scope("generator") as scope0:
        if not is_train:
            scope0.reuse_variables()

        # 100 x 1 to 4 x 4 x 512
        # 全连接层
        layer1 = tf.layers.dense(noise_img, F_I_H*F_I_W*512)
        layer1 = tf.reshape(layer1, [-1, F_I_H, F_I_W, 512])
        layer1 = tf.layers.batch_normalization(layer1, training=is_train)
        # Leaky ReLU
        layer1 = tf.maximum(alpha * layer1, layer1)
        # dropout
        layer1=tf.nn.dropout(layer1,keep_prob=0.9)


        # 4 x 4 x 512 to 8 x 8 x 256
        layer2 = tf.layers.conv2d_transpose(layer1, 256, 4, strides=2, padding='same')
        layer2 = tf.layers.batch_normalization(layer2, training=is_train)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2=tf.nn.dropout(layer2,keep_prob=0.9)

        # 8 x 8 256 to 16 x 16 x 128
        layer3 = tf.layers.conv2d_transpose(layer2, 128, 3, strides=2, padding='same')
        layer3 = tf.layers.batch_normalization(layer3, training=is_train)
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3=tf.nn.dropout(layer3,keep_prob=0.9)

        # 16 x 16 x 128 to 32 x 32 x 3
        logits = tf.layers.conv2d_transpose(layer3, output_dim, 3, strides=2, padding='same')

        outputs = tf.tanh(logits)

        tf.summary.image('input',outputs,10)

        return outputs



def discriminator(img_or_noise, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator') as scope1:

        if reuse:
            scope1.reuse_variables()

        layer1=tf.layers.conv2d(img_or_noise,128,3,strides=2,padding='same')
        layer1=tf.maximum(alpha*layer1,layer1)
        layer1=tf.nn.dropout(layer1,keep_prob=0.9)


        layer2=tf.layers.conv2d(layer1,256,3,strides=2,padding='same')
        layer2=tf.layers.batch_normalization(layer2,training=True)
        layer2=tf.maximum(alpha*layer2,layer2)
        layer2=tf.nn.dropout(layer2,keep_prob=0.9)

        layer3=tf.layers.conv2d(layer2,512,3,strides=2,padding='same')
        layer3=tf.layers.batch_normalization(layer3,training=True)
        layer3=tf.maximum(alpha*layer3,layer3)
        layer3=tf.nn.dropout(layer3,keep_prob=0.9)

        flatten=tf.reshape(layer3,(-1,F_I_H*F_I_W*512))
        logits=tf.layers.dense(flatten,1)
        outputs=tf.sigmoid(logits)
        return logits,outputs


def inference(real_img,fake_noise,image_depth=3,smooth=0.1):
    g_outputs=generator(fake_noise,image_depth,is_train=True)
    d_logits_real,d_outputs_real=discriminator(real_img)
    d_logits_fake,d_outputs_fake=discriminator(g_outputs,reuse=True)

    g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                        labels=tf.ones_like(d_outputs_fake)*(1-smooth)))

    d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                        labels=tf.ones_like(d_outputs_real)*(1-smooth)))

    d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                        labels=tf.zeros_like(d_outputs_fake)))

    d_loss=tf.add(d_loss_real,d_loss_fake)

    tf.summary.scalar('d_loss_real', d_loss_real)
    tf.summary.scalar('d_loss_fake', d_loss_fake)



    return g_loss,d_loss

def test(fake_placeholder,output_dim=3,num_images=25):
    with tf.Session() as sess:
        if RESTORE:
            saver.restore(sess,tf.train.latest_checkpoint('checkpoints'))
        fake_shape=fake_placeholder.shape.as_list()[-1]

        fake_images=np.random.uniform(-1,1,size=[num_images,fake_shape])

        samples=sess.run(generator(fake_placeholder,output_dim,is_train=False),
                         feed_dict={fake_placeholder:fake_images})

        

        plot_image(samples)

def plot_image(samples):

    samples=(samples+1)/2.
    print(samples.shape)
    fig,axes=plt.subplots(nrows=5,ncols=5,figsize=(7,7))
    
    count_file = 0
    for img,ax in zip(samples,axes.flatten()):        
        count_file += 1
        #ax.imshow(img, cmap='Greys_r')        
        ax.imshow(img)        

        ax.axis('off')

        scipy.misc.imsave(result_dir + "img" + str(count_file) + ".png", img)

    plt.show()


def get_optimizer(g_loss,d_loss,beta=0.4,learning_rate=0.001):
    train_vars=tf.trainable_variables()
    g_vars=[var for var in train_vars if var.name.startswith('generator')]
    d_vars=[var for var in train_vars if var.name.startswith('discriminator')]
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_opt=tf.train.AdamOptimizer(learning_rate,beta1=beta).minimize(g_loss,var_list=g_vars)
        d_opt=tf.train.AdamOptimizer(learning_rate,beta1=beta).minimize(d_loss,var_list=d_vars)

    return g_opt,d_opt

def train(real_placeholder,fake_placeholder,g_train_opt,d_train_opt,epoches,noise_size=100,batch_size=64,n_samples=25):
    global_step_=tf.Variable(0,trainable=False)
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        if RESTORE:
            saver.restore(sess,tf.train.latest_checkpoint('checkpoints'))
        summary_writer = tf.summary.FileWriter('log/', sess.graph)

        for e in range(1,epoches):

            for step in range(len(mnist)//batch_size):
                global_step_=global_step_+1
                images=mnist[step*batch_size:(step+1)*batch_size]

                batch_image=images*2 -1

                batch_noise=np.random.uniform(-1,1,size=(batch_size,noise_size))

                sess.run(g_train_opt,feed_dict={real_placeholder:batch_image,fake_placeholder:batch_noise})


                sess.run(d_train_opt,feed_dict={real_placeholder:batch_image,fake_placeholder:batch_noise})

                summary_str=sess.run(summary,feed_dict={real_placeholder:batch_image,fake_placeholder:batch_noise})

                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                if step %50 ==0:

                    train_loss_d=d_loss.eval({real_placeholder:batch_image,fake_placeholder:batch_noise})

                    train_loss_g=g_loss.eval(feed_dict={fake_placeholder:batch_noise})

                    print('step:{}/Epoch:{}/total Epoch:{}'.format(step,e,epoches),
                          'Discriminator Loss:{:.4f}..'.format(train_loss_d),'Generator Loss:{:.4f}..'.format(train_loss_g))

            saver.save(sess,'./checkpoints/generator.ckpt',global_step=global_step_)


with tf.device('/cpu:0'):
    #with tf.Graph().as_default():

    real_img,fake_img=input_placeholder([-1,I_H,I_W,3],noise_size=20)

    g_loss,d_loss=inference(real_img,fake_img)
    summary = tf.summary.merge_all()
    g_train_opt,d_train_opt=get_optimizer(g_loss,d_loss)

    saver=tf.train.Saver()
    if not TEST:
        train(real_img,fake_img,g_train_opt,d_train_opt,epoches=50,noise_size=20,batch_size=1)
    test(fake_img,num_images=25)
