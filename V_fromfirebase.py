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

import V_predict as predict
import py5_template as template
import V_labelling as label

json_key = 'foruser-86c2a-firebase-adminsdk-o9ghu-8e76c66a0c.json'
db_url = 'https://foruser-86c2a.firebaseio.com',
bk_name = 'foruser-86c2a.appspot.com'

fb_dir = './firebase'


def learning_model(user_id):
    label.make_label(user_id)


# convert wav file to spectrogram
def convert_wav_file(uid, dname):
    os.system('binbin.sh firebase/%s/%s' % (uid, dname))

    print("--- convert to spectrogram ---")


# download file from firebase
def download_from_fb(uid, dname):
    dir_path = uid + "/" + dname
    blobs = bucket.list_blobs(prefix=dir_path, delimiter=None)

    for blob in blobs:
        blob = bucket.blob(blob.name)  # HuR20vWoMycfidmj9oMl2OTDx033/01.wav
        fname = blob.name.split('/')[1] # 01.wav
 ##       print(fname)

        fdir = fb_dir + '/' + uid  # ./fiebase/HuR20vWoMycfidmj9oMl2OTDx033

        if not os.path.isdir(fdir):
            os.mkdir(fdir)

        fdir = fb_dir + '/' + uid + '/' + dname  # ./fiebase/HuR20vWoMycfidmj9oMl2OTDx033/[dname]
        if not os.path.isdir(fdir):
            os.mkdir(fdir)

        fpath = fdir + '/' + fname  # ./firebase/HuR20vWoMycfidmj9oMl2OTDx033/[dname]/01.wav
    blob.download_to_filename(fpath)
    print("--- download from firebase ---")


# firebase realtime database event listener
def listener(event):
    if event.path == '/':
        return

    flag = event.path.split('/')[-1]
    user_gr = event.path.split('/')[1]
    ##user_id = event.path.split('/')[2]

    print(flag)
    print(event.data)

    if flag == 'using' and event.data == 'true':
        print('--- Start prediction ---')
        download_from_fb('kDdNvE4mcfQsdXVmUrtJ9TAMUZm2', 'using')
        convert_wav_file('kDdNvE4mcfQsdXVmUrtJ9TAMUZm2', 'using')

        result = predict.predict_spect('kDdNvE4mcfQsdXVmUrtJ9TAMUZm2')

        upath = user_gr  # + "/learning" #여기
        u_ref = db.reference(path=upath, url=db_url)
        u_ref.update({'result': result})
        u_ref.update({'result': result})

if __name__ == '__main__':
    cred = credentials.Certificate(json_key)
    firebase_admin.initialize_app(cred)
    ref = db.reference(url=db_url)
    bucket = storage.bucket(bk_name)

    print("-----------------------\n")
    print("\tvamonos\n")
    print("-----------------------\n")

    ref.listen(listener)
