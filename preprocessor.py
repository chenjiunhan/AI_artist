import os
import scipy.misc
import numpy as np

I_H = 480
I_W = 640

image_dir = "image_data/"
training_dir = "training_data/"
result_dir = "result/"
images = []

for image_file in os.listdir(image_dir):
    image_path = image_dir + image_file
    save_path = training_dir + os.path.splitext(image_file)[0] + ".png"
    image = scipy.misc.imread(image_path)
    image = scipy.misc.imresize(image, (I_H, I_W)) / 255.
    images += [image]
    #print(image)
    #scipy.misc.imshow(image)
    #scipy.misc.imsave(save_path, image)

images = np.array(images)
print(images.shape)
np.save("images.npy", images)

