import tensorflow as tf
import numpy as np
from PIL import Image

label_list = ['아메리카노', '카페라떼',
              '카페모카', '카라멜마끼야또',
              '에스프레소', '프라푸치노',
              '핫초코', '플레인요거트스무디']


def read_one_spect_png(image_file):
    # png 파일을 읽어 ndarray로 변환
    img_ndarr = np.array([np.array(Image.open(image_file).convert('RGB'))])  ### color (spectrogram)
    # img_ndarr = np.array([np.array(Image.open(image_file).convert('L'))])	### gray (waveform)

    img_ndarr = abs(img_ndarr / 255 - 1)  # normalization(0~1)

    img_ndarr = img_ndarr.reshape(-1, 100, 100, 3)  ### color (spectrogram)
    # img_ndarr = img_ndarr.reshape(-1, 256, 256, 1)	### gray (waveform)

    return img_ndarr


def predict_spect(uid):
    input_image_name = "./firebase/kDdNvE4mcfQsdXVmUrtJ9TAMUZm2/using/using.png"

    model_path = './firebase/kDdNvE4mcfQsdXVmUrtJ9TAMUZm2/model/'
    model_name = './firebase/kDdNvE4mcfQsdXVmUrtJ9TAMUZm2/model/vamonos_model-500.meta'
    saver = tf.train.import_meta_graph(model_name)

    with tf.Session() as sess:
        # recareate NN
        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        # get default graph
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('X:0')  # placeholder : input image
        Y = graph.get_tensor_by_name('Y:0')  # prediction
        dkeep = graph.get_tensor_by_name('dkeep:0')  # dropout keep rate

        img_ndarr = read_one_spect_png(input_image_name)

        predict_result = sess.run(Y, feed_dict={X: img_ndarr, dkeep: 1.0})

        prediction_list = predict_result[0].tolist()  # predict_result[0] : numpy arr
        # print(prediction_list)
        print('')
        for index, score in enumerate(prediction_list):
            print('%s %s %0.3f' % (index, label_list[index], score))

        max_score_index = prediction_list.index(max(prediction_list))
        print(max_score_index)
        print('-------------------------------------------')
        print('prediction = %s' % (label_list[max_score_index]))

        return label_list[max_score_index]