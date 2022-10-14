# -*- coding: utf-8 -*-

#connect to firebase
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import math

from libsvm.commonutil import svm_read_problem
from libsvm.svmutil import *

import scipy
from scipy import signal
from scipy.interpolate import interp1d
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks

# Import packages
import biosppy
from opensignalsreader import OpenSignalsReader



import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import db

import pyrebase
firebaseConfig={
    "apiKey": "AIzaSyBq-8HojfR-eI9HmCWOLskkIxb6hhC4J_M",
    "authDomain": "showsignalonwpipeb.firebaseapp.com",
    "databaseURL": "https://showsignalonweb-default-rtdb.firebaseio.com",
    "projectId": "showsignalonweb",
    "storageBucket": "showsignalonweb.appspot.com",
    "messagingSenderId": "427435107006",
    "appId": "1:427435107006:web:53b9595b7220b94ac2f4ef",
    "measurementId": "G-Y1R5T7X4DH"
}
firebase = pyrebase.initialize_app(firebaseConfig)
RealTimeDatabase = firebase.database()

cred = credentials.Certificate("firebase_sdk.json")
firebase_admin.initialize_app(cred,{
    "databaseURL": "https://showsignalonweb-default-rtdb.firebaseio.com/"
})
model_svm = svm_load_model('model_svm')

from flask import Flask, render_template, request
app = Flask(__name__)
@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")

@app.route('/ShowSignal.html')
def ShowSignal():
    return render_template('ShowSignal.html')

@app.route('/Features.html')
def GetValueSporadically():
    return render_template('Features.html')

class ECGClass:
    index = 0
    # data is dictionary
    data = {
        u'ECG': u'0'
    }

 # data = {
 #      u'12',
 #      u'300'
 #     #u'ECG': u'300'
 # }
# Get a database reference to our posts
ref = db.reference('server/saving-data/fireblog/posts')
db = firestore.client()
# Read the data at the posts reference (this is a blocking operation)
print(ref.get())

### Calculate moving average with 0.75s in both directions, then append do dataset
hrw = 0.75  # One-sided window size, as proportion of the sampling frequency
fs = -1  # The example dataset was recorded at 500Hz
DataLen = 3333
AllData = []
measures = {}
mood = "Detecting..."

def calc_RR(dataset, fs, peaklist):
    RR_list = []
    cnt = 0

    while (cnt < (len(peaklist) - 1)):
        RR_interval = (peaklist[cnt + 1] - peaklist[cnt])
        ms_dist = ((RR_interval / fs) * 1000.0)
        RR_list.append(ms_dist)
        cnt += 1

    RR_diff = []
    RR_sqdiff = []
    cnt = 0
    while (cnt < (len(RR_list) - 1)):
        RR_diff.append(abs(RR_list[cnt] - RR_list[cnt + 1]))
        RR_sqdiff.append(math.pow(RR_list[cnt] - RR_list[cnt + 1], 2))
        cnt += 1
    measures['RR_list'] = RR_list
    measures['RR_diff'] = RR_diff
    measures['RR_sqdiff'] = RR_sqdiff

def calc_ts_measures():
    RR_list = measures['RR_list']
    RR_diff = measures['RR_diff']
    RR_sqdiff = measures['RR_sqdiff']
    measures['bpm'] = 60000 / np.mean(RR_list)
    measures['ibi'] = np.mean(RR_list)
    measures['sdnn'] = np.std(RR_list)
    measures['sdsd'] = np.std(RR_diff)
    measures['rmssd'] = np.sqrt(np.mean(RR_sqdiff))
    NN20 = [x for x in RR_diff if (x > 20)]
    NN50 = [x for x in RR_diff if (x > 50)]
    measures['nn20'] = NN20
    measures['nn50'] = NN50
    measures['pnn20'] = float(len(NN20)) / float(len(RR_diff))
    measures['pnn50'] = float(len(NN50)) / float(len(RR_diff))

def FrequencyDomain(ppi_time, ppi):
    VLF = 0
    LF = 0
    HF = 0
    TP = 0
    NewHRV = []
    # x=ppi[1]:(1/4):ppi(len(ppi)-1);
    # x = np.arange(ppi[0], len(ppi)+1, 0.25)
    x = np.linspace(0, len(ppi), len(ppi))
    InterpFunc = interp1d(x, ppi, kind='cubic')
    xnew = np.linspace(0, len(ppi), len(ppi) * 4)
    NewHRV = InterpFunc(xnew)
    plt.plot(x, ppi, 'o', xnew, NewHRV, '--')
    plt.show()
    NewHRV = np.pad(NewHRV, (0, 1024 - len(NewHRV) % 1024), 'constant')
    np.absolute(NewHRV)
    print(NewHRV)
    freq = scipy.fft.fft(NewHRV)
    plt.plot(freq)
    plt.show()
    return VLF, LF, HF, TP



def FindPeaks():
    ppi = []
    # peaks, amp = find_peaks(AllData, height=0, distance=(fs*0.3))
    peaks, amp = find_peaks(AllData, height=np.mean(AllData), distance=(fs * 0.9))
    print(peaks)
    peaks = peaks/fs;
    #np.diff(peaks)
    for i in range(len(peaks)-1):
        ppi.append(peaks[i+1]-peaks[i])
    print(ppi)
    return ppi

def ComputeSDSD(rri):
    RRInter = []
    for i in range(len(rri)-1):
        RRInter.append(rri[i+1]-rri[i])
    sdsd = np.sqrt(sum(np.power((RRInter-np.mean(RRInter)),2)))
    return sdsd

def ComputeRMSSD(rri):
    sum = 0
    rmssd = 0
    for i in range(len(rri)-1):
        sum += np.power(rri[i+1]-rri[i],2)
    rmssd = np.sqrt(sum/(len(rri)-1))
    return rmssd

def FrequencyDomain(ppi):
    VLF, LF, HF,TP=0
    NewHRV = []
    # x=ppi[1]:(1/4):ppi(len(ppi)-1);
    x = np.arange(ppi[0], len(ppi)+1, 0.25)
    print(x)
    # 做zero padding
    NewHRV = scipy.interpolate.CubicSpline(x, ppi)
    freq = scipy.fft(NewHRV)

    return VLF, LF, HF,TP


def TrainSVM():
    ## Read data in LIBSVM format
    y, x = svm_read_problem('../heart_scale')
    m = svm_train(y[:200], x[:200], '-c 4')
    p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)

@app.route("/Input_Freq", methods = ['POST', "GET"])
def Input_Freq():
    global fs
    fs = request.form['frequency']
    if fs =="":
        fs =-1
    fs = float(fs)
    print(str(fs))
    return render_template("Features.html", Warning="fs="+str(fs))

@app.route("/change_ECG_value", methods = ['POST', "GET"])
def change_ECG_value():
    AllData.clear()
    print("fs: ",fs)
    if fs == -1:
        return render_template("Features.html", Warning="Input Freq!")
    ### Get all data in real-time dataset
    sdnn = 0
    sdsd = 0
    rmssd = 0
    vlf = 0
    lf = 0
    hf = 0
    tp = 0
    global mood
    mood = "Detecting..."
    real_datas = RealTimeDatabase.order_by_key().limit_to_last(DataLen).get()
    #print(real_datas.each())
    if real_datas.each() == None:
        return render_template("Features.html", Warning="No Data!")
    else:
        for data in real_datas.each():  # 對於key順序的遍歷
            if data.val()!="":
                AllData.append(float(data.val()))
    #         #print(AllData)
    if len(AllData)>DataLen-1:
        ### Mark regions of interest
        window = []
        peaklist = []
        listpos = 0  # We use a counter to move over the different data columns
        peaklist, peakheight = find_peaks(AllData, height=np.mean(AllData), distance=(fs * 0.9))
        measures['peaklist'] = peaklist
        measures['peakheight'] = peakheight
        ### Calculate heart rate
        # BPM: 即每分鐘的心跳量
        calc_RR(AllData, fs, peaklist)
        ### Time domain
        # R-peak的位置
        # RR之間的間隔
        # RR對之間所有間隔的差異 ex.RRdiff...
        # RR對所有後續差之間的平方差
        RR_diff = []
        RR_sqdiff = []
        RR_list = measures['RR_list']
        cnt = 1  # Use counter to iterate over RR_list
        while (cnt < (len(RR_list) - 1)):  # Keep going as long as there are R-R intervals
            RR_diff.append(
                abs(RR_list[cnt] - RR_list[cnt + 1]))  # Calculate absolute difference between successive R-R interval
            RR_sqdiff.append(math.pow(RR_list[cnt] - RR_list[cnt + 1], 2))  # Calculate squared difference
            cnt += 1

        print(RR_diff, RR_sqdiff)
        ### Claculate Time Domain
        ibi = np.mean(RR_list)  # Take the mean of RR_list to get the mean Inter Beat Interval
        print("IBI:", ibi)
        sdnn = np.std(RR_list)  # Take standard deviation of all R-R intervals
        print("SDNN:", sdnn)
        sdsd = np.std(RR_diff)  # Take standard deviation of the differences between all subsequent R-R intervals
        print("SDSD:", sdsd)
        rmssd = np.sqrt(np.mean(RR_sqdiff))  # Take root of the mean of the list of squared differences
        print("RMSSD:", rmssd)
        NN20 = [x for x in RR_diff if (x > 20)]  # First create a list of all values over 20, 50
        NN50 = [x for x in RR_diff if (x > 50)]
        pnn20 = float(len(NN20)) / float(
            len(RR_diff))  # Calculate the proportion of NN20, NN50 intervals to all intervals
        pnn50 = float(len(NN50)) / float(
            len(RR_diff))  # Note the use of float(), because we don't want Python to think we want an int() and round the proportion to 0 or 1
        print("pNN20, pNN50:", pnn20, pnn50)
        ### Frequency Domain
        # time sequence 不是從第一個peak開始，而是从第二个R-peak位置开始,因為使用時間間隔
        peaklist = measures['peaklist']  # First retrieve the lists we need
        RR_list = measures['RR_list']
        RR_x = peaklist[1:]  # Remove the first entry, because first interval is assigned to the second beat.
        RR_y = RR_list  # Y-values are equal to interval lengths
        RR_x_new = np.linspace(RR_x[0], RR_x[-1], RR_x[
            -1])  # Create evenly spaced timeline starting at the second peak, its endpoint and length equal to position of last peak
        f = interp1d(RR_x, RR_y, kind='cubic',fill_value="extrapolate")  # Interpolate the signal with cubic spline interpolation
        # # Plot
        # plt.title("Original and Interpolated Signal")
        # plt.plot(RR_x, RR_y, label="Original", color='blue')
        # plt.plot(RR_x_new, f(RR_x_new), label="Interpolated", color='red')
        # plt.legend()
        # plt.show()
        ### FFT Time-domain-->Frequency Domain
        # Set variables
        n = len(AllData)  # Length of the signal
        frq = np.fft.fftfreq(len(AllData), d=((1/fs)))  # divide the bins into frequency categories
        frq = frq[range(n // 2)]  # Get single side of the frequency range

        # Do FFT
        RR_x_new = f(RR_x_new)
        RR_x_new = np.pad(RR_x_new, (0, 8192 - len(RR_x_new) % 8192), 'constant')
        print("RR_x_new")
        print(RR_x_new)
        Y = np.fft.fft(RR_x_new) / n  # Calculate FFT
        Y = Y[range(n // 2)]  # Return one side of the FFT

        vlf = abs(np.trapz(Y, [0, 0.04]))
        print("VLF:", vlf)
        lf = abs(np.trapz(Y, [0.04, 0.15]))
        print("LF:", lf)
        hf = abs(np.trapz(Y, [0.15, 0.4]))
        print("HF:", hf)
        tp = abs(np.trapz(Y, [0, 0.4]))
        print("TP:", tp)
        ampitude = np.array(list(peakheight.values())).mean()
        print("ampitude:", ampitude)
        # vlf = np.trapz(abs(Y[(frq >= 0) & (frq <= 0.04)]))
        # print("VLF:", vlf)
        # lf = np.trapz(abs(Y[(frq >= 0.04) & (
        #         frq <= 0.15)]))  # Slice frequency spectrum where x is between 0.04 and 0.15Hz (LF), and use NumPy's trapezoidal integration function to find the area
        # print("LF:", lf)
        # hf = np.trapz(abs(Y[(frq >= 0.16) & (frq <= 0.5)]))  # Do the same for 0.16-0.5Hz (HF)
        # print("HF:", hf)
        # tp = np.trapz(abs(Y[(frq >= 0) & (frq <= 0.4)]))
        # print("TP:", tp)

        # Store features to database
        ECGClass.index = ECGClass.index + 1
        # doc_ref = db.collection("Signal").document('ECG')
        # ECGClass.data['index'] = ECGClass.index
        ECGClass.data['SDNN'] = sdnn
        ECGClass.data['SDSD'] = sdsd
        ECGClass.data['RMSSD'] = rmssd
        ECGClass.data['VLF'] = vlf
        ECGClass.data['LF'] = lf
        ECGClass.data['HF'] = hf
        ECGClass.data['TP'] = tp

        ## find vally
        valleylist = []
        valleyAmp = []
        ValueIndex = peaklist[0]
        VallyValue = AllData[peaklist[0]]
        for x in range(len(peaklist) - 1):
            VallyValue = AllData[peaklist[x]]
            for xindex in range(peaklist[x], peaklist[x + 1]):
                if AllData[xindex] < VallyValue:
                    VallyValue = AllData[xindex]
                    ValueIndex = xindex
            valleylist.append(ValueIndex)
            valleyAmp.append(VallyValue)
        ## Find Amp
        AmpTotal = 0
        for x in range(len(valleylist)):
            AmpTotal += (peakheight['peak_heights'][x] - valleyAmp[x])
        print(np.mean(AmpTotal))
        peakheight = np.array(list(peakheight.values())).mean()

        # db.collection(u'Signal').document().set(ECGClass.data)
        # ECGClass.data['TEST'] = 300
        db.collection(u'Signal').document(u'ECG{0:09d}F'.format(ECGClass.index)).set(ECGClass.data)
        print(ECGClass.index)
        # doc_ref.set({str(ECGClass.index): 150})
        ## Using SVM predict emotion
        FileWrite = open('PredictFeature.txt', 'w')
        ## write to txt for libsvm
        FileWrite.write("0 1:" + str(ibi))
        FileWrite.write(" 2:" + str(sdnn))
        FileWrite.write(" 3:" + str(sdsd))
        FileWrite.write(" 4:" + str(rmssd))
        FileWrite.write(" 5:" + str(vlf))
        FileWrite.write(" 6:" + str(lf))
        FileWrite.write(" 7:" + str(hf))
        FileWrite.write(" 8:" + str(tp))
        FileWrite.write(" 9:" + str(pnn20))
        FileWrite.write(" 10:" + str(pnn50))
        FileWrite.write(" 11:" + str(peakheight))
        FileWrite.write("\n")
        FileWrite.close()
        yt, xt = svm_read_problem('PredictFeature.txt')  # read test data
        p_label, p_acc, p_val = svm_predict(yt, xt, model_svm)
        if p_label[0] ==1:
            mood = "Normal"
        if p_label[0] ==2:
            mood = "Sad"
        if p_label[0] == 3:
            mood = "Stress"
        if p_label[0] == 4:
            mood = "Happy"
        print("Test p_label: ", p_label)
    # return render_template("Features.html",Warning="", SDNN=ECGClass.data['SDNN'], SDSD=ECGClass.data['SDSD'], RMSSD=ECGClass.data['RMSSD'], VLF=ECGClass.data['VLF'], LF=ECGClass.data['LF'],
    #     HF=ECGClass.data['HF'],
    #     TP=ECGClass.data['TP'])
    return render_template("Features.html", Warning="fs="+str(fs),mood=mood, SDNN=round(sdnn,2), SDSD=round(sdsd,2),
                           RMSSD=round(rmssd,2), VLF=round(vlf,2), LF=round(lf,2),
                           HF=round(hf,2),
                           TP=round(tp,2))

@app.route("/ClearAll", methods = ['POST', "GET"])
def ClearAll():
    RealTimeDatabase.set('/')
    i = 1;
    while(db.collection(u'Signal').document(u'ECG{0:09d}F'.format(i)).get().exists):
        db.collection(u'Signal').document(u'ECG{0:09d}F'.format(i)).delete()
        i= i+1;
        print(i)
    ECGClass.index = 0
    fs = -1
    mood = "Detecting..."
    return render_template("Features.html")


if __name__ == '__main__':
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=5230)
    app.run(debug= True, port=5230, host='0.0.0.0')

#pytorch code
# import torch
# import math
#
#
# dtype = torch.float
# device = torch.device("cpu")
# # device = torch.device("cuda:0") # Uncomment this to run on GPU
#
# # Create random input and output data
# x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
# y = torch.sin(x)
#
# # Randomly initialize weights
# a = torch.randn((), device=device, dtype=dtype)
# b = torch.randn((), device=device, dtype=dtype)
# c = torch.randn((), device=device, dtype=dtype)
# d = torch.randn((), device=device, dtype=dtype)
#
# learning_rate = 1e-6
# for t in range(2000):
#     # Forward pass: compute predicted y
#     y_pred = a + b * x + c * x ** 2 + d * x ** 3
#
#     # Compute and print loss
#     loss = (y_pred - y).pow(2).sum().item()
#     if t % 100 == 99:
#         print(t, loss)
#
#     # Backprop to compute gradients of a, b, c, d with respect to loss
#     grad_y_pred = 2.0 * (y_pred - y)
#     grad_a = grad_y_pred.sum()
#     grad_b = (grad_y_pred * x).sum()
#     grad_c = (grad_y_pred * x ** 2).sum()
#     grad_d = (grad_y_pred * x ** 3).sum()
#
#     # Update weights using gradient descent
#     a -= learning_rate * grad_a
#     b -= learning_rate * grad_b
#     c -= learning_rate * grad_c
#     d -= learning_rate * grad_d
#
#
# print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')