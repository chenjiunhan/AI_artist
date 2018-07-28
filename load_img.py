import tensorflow as tf

training_dir = "training_data/"

filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(training_dir + "*.png"))

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

images = tf.image.decode_png(value, channels=3)
print(images)

sess = tf.Session()
result = sess.run(images)
