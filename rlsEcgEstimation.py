import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# load patient data into appropriate variables
patientNo = 1
patientStr = 'ECG_' + str(patientNo) + '/' + 'ECG_' + str(patientNo) + '_'
ECG_AVR = sio.loadmat(str(patientStr + 'AVR'))
ECG_II_missing = sio.loadmat(patientStr + 'II_missing')
ECG_II = sio.loadmat(patientStr + 'II')
ECG_V = sio.loadmat(patientStr + 'V')

xT = np.array(ECG_II[str('ECG_'+str(patientNo)+'_II')])
x1 = np.array(ECG_V['ECG_'+str(patientNo)+'_V'])
x2 = np.array(ECG_AVR['ECG_'+str(patientNo)+'_AVR'])
xTMissing = np.array(ECG_II_missing['ECG_'+str(patientNo)+'_II_missing'])
print('line17' + str(xT.shape))
print('line18' + str(x1.shape))
print('line19' + str(x2.shape))
print('21 ECG_II_missing.shape: ' + str(xTMissing.shape))
# model as xTHat = sum over N+1 i a[i] x1[n-i] + sum over M+1 i b[i] x2[n-i]
M = 20
N = 20
p = M + N + 2
n0 = max(M, N)+1

# reshape vectors
xT = xT.reshape(max(xT.shape), 1)
x1 = x1.reshape(max(x1.shape), 1)
x2 = x2.reshape(max(x2.shape), 1)
xTMissing = xTMissing.reshape(max(xTMissing.shape), 1)

plt.plot(xT)
plt.show()
# signals AVR and V are recordings of 10 minutes so find sample period and freq
x1Length = x1.shape[0]
Ts = (60*10)/x1Length
fs = 1/Ts
print('40 Ts = ' + str(Ts) + ', fs = ' + str(fs))

# create time vector tVec
N95min = round(x1Length*0.95)
N10min = x1Length
t10min = np.array(range(N10min))*Ts
t95min = np.array(range(round(N95min)))*Ts
t05min = np.array(range(N95min, N10min))*Ts

print('50 t95min.shape: ' + str(t95min.shape))

# initialize RLS
print('xtShape[0] '+str(xT.shape[0]))
NVectors = N95min-n0
print('line55 NVectors: '+str(NVectors))
P = np.matrix(1*np.identity(p))
cHat = np.zeros(shape=(p, 1))  # theta hat
h = np.zeros(shape=(p, 1))
K = np.zeros(shape=(p, 1))
I = np.identity(p)

lm = 0.999  # lambda in RLS algorithm
hVec = np.matrix(np.zeros(shape=(p, N95min)))
cHatVec = np.matrix(np.zeros(shape=(p, N95min)))
xHatVec = (np.zeros(shape=(N95min)))
KVec = np.matrix(np.zeros(shape=(p, N95min)))
eVec = np.zeros(shape=(N95min))
eNVec = np.zeros(shape=(N95min))

# go do that RLS thingy
for n in range(n0, N95min):
    y = xT[n]
    h[0:p] = np.matrix(np.append(x1[n-N-1:n], x2[n-M-1:n]).reshape(p, 1))
    xHat = np.dot(h.T, cHat)
    e = y - xHat
    Ph = P*h
    K = np.asscalar(1/(lm + h.T*Ph))*Ph  # alternatively put lm**n but can explode
    cHat = cHat + K*e
    P = P-K*Ph.T  # (I - K*h.T)*P, where h.T*P = (P.T*h).T = (P*h).T = Ph.T

    cHatVec[:, n] = cHat
    hVec[:, n] = h
    KVec[:, n] = K
    eVec[n] = e
    xHatVec[n] = xHat

# test on new data
NTest = xTMissing.shape[0]
eTestVec = np.zeros(shape=NTest)
xHatTestVec = np.zeros(shape=NTest)
for n in range(NTest):
    y = xTMissing[n]
    h[0:p] = np.matrix(np.append(x1[n+N95min-N-1:n+N95min],
                                 x2[n+N95min-M-1:n+N95min]).reshape(p, 1))
    xHat = np.dot(h.T, cHat)
    e = y - xHat

    eTestVec[n] = e
    xHatTestVec[n] = xHat

print('xT avg: ' + str(np.average(xT)))
print('xHat avg: ' + str(np.average(xHatVec)))
plt.plot(xHatVec)
plt.title('xHat training')
plt.show()
fig, axs = plt.subplots(2, 2, constrained_layout=True)
eNVec = eVec**2/(np.average(xT)**2)  # normalized square error
# plot x and xhat during training
end = t95min.shape[0]
axs[0, 0].plot(t95min[end-3000:end], xHatVec[end-3000:end], t95min[end-3000:end],
               xT[end-3000:end], label=['xHat Training', 'xT true'])
axs[0, 0].legend(loc='upper right')
axs[0, 0].set_title('Training set')
# plot error during training
axs[1, 0].plot(t95min[end-3000:end], 10*np.log10(eNVec[end-3000:end] +
                                                 0.000000000000001), label='10log10(eVec**2/xT_avg^2')
axs[1, 0].plot([t95min[end-3000], t95min[end-1]], [np.average(10*np.log10(eNVec[end-3000:end]+0.000000000000001)),
                                                   np.average(10*np.log10(eNVec[end-3000:end]+0.000000000000001))], label='mean')
axs[1, 0].legend(loc='upper right')
axs[1, 0].set_title('M=' + str(M) + ', N=' + str(N) + ', lambda=' + str(lm) + ' (TRAINING)')

axs[0, 1].plot(t05min, xTMissing, label='EVR_II_missing')
axs[0, 1].plot(t05min, xHatTestVec, label='RLS estimate')
axs[0, 1].legend(loc='upper right')
axs[1, 1].plot(t05min, 10*np.log10((eTestVec)**2/np.average(xTMissing)**2), label='10log10(nse^2)')
axs[1, 1].set_title('normalized square error in dB (TEST)')
plt.show()
# hello github a
