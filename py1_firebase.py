import os
from os.path import isfile, join
import random
import sys
import threading
import time

from scipy.io import wavfile
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import db

import py4_predict as predict
import py5_template as template
import py2_labelling as label

json_key = 'foruser-86c2a-firebase-adminsdk-o9ghu-8e76c66a0c.json'
db_url = 'https://foruser-86c2a.firebaseio.com',
bk_name = 'foruser-86c2a.appspot.com'

fb_dir = './firebase'

liv_light = 0
main_light = 0
tol_light = 0


def learning_model(user_id):
    label.make_label(user_id)
    print("	>> success learning")


# convert wav file to spectrogram
def convert_wav_file(uid, dname):
    print(uid)
    print(dname)
    os.system('wts.sh firebase/%s/%s' % (uid, dname))

    print("	>> success convert")


# download file from firebase
def download_from_fb(uid, dname):
    dir_path = uid + "/" + dname
    blobs = bucket.list_blobs(prefix=dir_path, delimiter=None)

    for blob in blobs:
        blob = bucket.blob(blob.name)  # HuR20vWoMycfidmj9oMl2OTDx033/01.wav
        fname = blob.name.split('/')[2]  # 01.wav
        fdir = fb_dir + '/' + uid  # ./fiebase/HuR20vWoMycfidmj9oMl2OTDx033

        if not os.path.isdir(fdir):
            os.mkdir(fdir)

        fdir = fb_dir + '/' + uid + '/' + dname  # ./fiebase/HuR20vWoMycfidmj9oMl2OTDx033/[dname]
        if not os.path.isdir(fdir):
            os.mkdir(fdir)

        fpath = fdir + '/' + fname  # ./firebase/HuR20vWoMycfidmj9oMl2OTDx033/[dname]/01.wav
        blob.download_to_filename(fpath)

    print("	>> success download")


# firebase realtime database event listener
def listener(event):
    print('= = = = = = EVENT = = = = = =')
    print("type : " + event.event_type)
    print("path : " + event.path)
    print(event.data)
    print('= = = = = = = = = = = = = = =\n')

    if event.path == '/':
        return

    flag = event.path.split('/')[-1]
    user_gr = event.path.split('/')[1]
    user_id = event.path.split('/')[2]

    if flag == 'learning' and event.data == 'true':
        print('----> Start learning')
        download_from_fb(user_id, flag)
        convert_wav_file(user_id, flag)

        learning_model(user_id)

        upath = "/" + user_gr + "/" + user_id
        u_ref = db.reference(path=upath, url=db_url)
        u_ref.update({'learning': 'false'})

        print("	>> success update\n")

    elif flag == 'using' and event.data == 'true':
        print('----> Start prediction')
        download_from_fb(user_id, flag)
        convert_wav_file(user_id, flag)

        upath = "/" + user_gr + "/" + user_id
        u_ref = db.reference(path=upath, url=db_url)

        mpath = upath + "/mode"
        m_ref = db.reference(path=mpath, url=db_url)
        mode = m_ref.get()
        print("----> Predict type : " + mode)

        if mode == 'deep':
            result = predict.predict_spect(user_id)
        elif mode == 'temp':
            result = template.tmp_predict(user_id)

        ### flag

        print(result)

        u_ref.update({'result': result})
        u_ref.update({'using': 'false'})

        print("	>> success update\n")

    elif flag == 'feedback':
        print('----> Feedback')

        img_path = "./firebase/" + user_id + "/using/using.png"
        save_path = "./firebase/" + user_id + "/learning/" + event.data + ".png"

        feed_img = Image.open(img_path)
        feed_img.save(save_path)

        label.make_label(user_id)

        print(' >> success feedback')


if __name__ == '__main__':
    cred = credentials.Certificate(json_key)

    rdb_app = firebase_admin.initialize_app(cred,
                                            {'databaseURL': 'https://foruser-86c2a.firebaseio.com'},
                                            name='rdb_app')
    ref = db.reference(app=rdb_app)
    str_app = firebase_admin.initialize_app(cred,
                                            {'storageBucket': 'foruser-86c2a.appspot.com'}, name='str_app')
    bucket = storage.bucket(app=str_app)

    print("# # # # # # # # # # # # # # #\n")
    print("\tStart Listen\n")
    print("# # # # # # # # # # # # # # #\n")

    ref.listen(listener)
