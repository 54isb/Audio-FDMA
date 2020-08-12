import numpy as np
import warnings
import matplotlib.pyplot as plt
import pyaudio
import matplotlib as mpl
from time import sleep
from scipy import signal
mpl.rcParams['toolbar'] = 'None'
#Default parameters *****************************************
warnings.filterwarnings('ignore')
SOUND_OUT = 1 #AUDIO OUT: 0: OFF / 1: ON
PULSE = 0 #PULSE SHAPE: 0: RRC, 1: NRZ
Fs = 44100 #SAMPING FREQ. [Hz] fs = 44.1KHz
fc1 = 440.0 #Center frequency [Hz] for User1: Default: 440.0Hz (=A)
fc2 = 1500.0 #Center frequency [Hz] for User2: Default: 840.0Hz
Ts = 0.01 #Symbol rate [sec]
DTs = (int)(Fs*Ts) #Number of samples per symbol
K = 8 #Number of sidelobes in shaped pulse [pulses]
alpha = 0.4 #Rolloff factor : Default 0.4
NUM_SYM = 400 #Number of symbols
SNR = 10 #Signal-to-Noise power Ratio (SNR)[dB] : Default 10dB
Gain = 10.0 #Gain for Audio output : Default 10.0
#************************************************************

#Shaping function based on [1] S. Chennakeshu, et. al., "Differential detection of pi/4-shifted-DQPSK for digital cellular radio," IEEE Trans. Vehicle Tech., vol.42, no.1, 1995
def rrc_shaping(t):
    condlist = [t == 0.0, t == Ts/(4.0*alpha), t == - Ts/(4.0*alpha), (t!=0.0)&(t!=Ts/(4.0*alpha))&(t!=- Ts/(4.0*alpha))]
    funclist = [lambda t: 1.0 - alpha + 4.0 * alpha/np.pi, lambda t: alpha/np.sqrt(2) * ((1+2.0/np.pi) * np.sin(np.pi/(4.0*alpha)) +  (1.0 - 2.0/np.pi) * np.cos(np.pi/(4.0*alpha))), lambda t: alpha/np.sqrt(2) * ((1+2.0/np.pi) * np.sin(np.pi/(4.0*alpha)) +  (1.0 - 2.0/np.pi) * np.cos(np.pi/(4.0*alpha))), lambda t:1.0/(np.pi * t/Ts * (1.0 - np.power(4.0*alpha*t/Ts,2))) * (np.sin(np.pi * (1.0-alpha) * t/Ts) + 4.0*alpha*t/Ts*np.cos(np.pi*(1.0+alpha)*t/Ts))]
    return np.piecewise(t.astype(dtype=np.float64), condlist, funclist)

#Non-Return-to-Zero (NRZ) Pulse
def nrz_shaping(t):
    condlist = [t < -0.5*Ts, t >= 0.5*Ts, (t>=-0.5*Ts)&(t<0.5*Ts)]
    funclist = [lambda t: 0.0, lambda t: 0.0, lambda t:np.sqrt(1.0/Ts)]
    return np.piecewise(t.astype(dtype=np.float64), condlist, funclist)

def mapping(x): #NOTE Gray-labeled QPSK
    condlist = [x ==0, x==1, x==2, x==3]
    funclist = [lambda x: 1.0/np.sqrt(2.0) + 1j * 1.0/np.sqrt(2.0),lambda x: -1.0/np.sqrt(2.0) + 1j*1.0/np.sqrt(2.0),lambda x:1.0/np.sqrt(2.0) - 1j * 1.0/np.sqrt(2.0),lambda x:-1.0/np.sqrt(2.0) - 1j * 1.0/np.sqrt(2.0)]
    return np.piecewise(x.astype(dtype=np.complex), condlist, funclist)

def demapping(x): #NOTE Demapper for Gray-labeled QPSK
    condlist = [(np.real(x)>0.0)&(np.imag(x)>0.0),(np.real(x)<=0.0)&(np.imag(x)>0.0),(np.real(x)>0.0)&(np.imag(x)<=0.0),(np.real(x)<=0.0)&(np.imag(x)<=0.0)]
    funclist = [lambda x: 0,lambda x: 1,lambda x: 2,lambda x: 3]
    return np.piecewise(x.astype(dtype=np.complex), condlist, funclist)

def comparebit(a,b): #Bit comparison for BER calculation
    return np.bitwise_and(1, np.bitwise_xor(a,b)) + np.bitwise_and(1, np.right_shift(np.bitwise_xor(a,b),1))

#Open Audio output
if SOUND_OUT == 1:
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=Fs,
                    frames_per_buffer=1024,
                    output=True)

#Transmission Main*****************************************

#generating filter response
if PULSE==0:
    #NOTE: 1/np.sqrt(Ts) is a normalization factor
    pulse = rrc_shaping(np.arange(-DTs*K,DTs*K+1) * 1.0/Fs) * 1.0/np.sqrt(Ts)
else:
    pulse = nrz_shaping(np.arange(-DTs*K,DTs*K+1) * 1.0/Fs)

#Generate Discrete Signals
data1 = np.random.randint(0,4,(NUM_SYM,1)) #for User1
data2 = np.random.randint(0,4,(NUM_SYM,1)) #for User2
#Mapping Data onto QPSK symbols
qpsk_sym1 = mapping(data1) #for User1
qpsk_sym2 = mapping(data2) #for User2

#generating observation samples in time domain
time_seq1 = np.reshape(np.zeros(NUM_SYM*DTs),[-1,1]) + 1j * np.reshape(np.zeros(NUM_SYM*DTs),[-1,1])
time_seq1[::DTs] = qpsk_sym1
time_seq2 = np.reshape(np.zeros(NUM_SYM*DTs),[-1,1]) + 1j * np.reshape(np.zeros(NUM_SYM*DTs),[-1,1])
time_seq2[::DTs] = qpsk_sym2
#generating pulse-shaped baseband signal
trans_sig1 = np.convolve(np.real(time_seq1).flatten(),pulse) + 1j * np.convolve(np.imag(time_seq1).flatten(),pulse)
trans_sig2 = np.convolve(np.real(time_seq2).flatten(),pulse) + 1j * np.convolve(np.imag(time_seq2).flatten(),pulse)

#Up conversion Baseband -> RF
time_axis = np.arange(np.size(trans_sig1)) * 1/Fs
tr_waveform = np.real(trans_sig1)*np.cos(2.0*np.pi*fc1*time_axis) - np.imag(trans_sig1)*np.sin(2.0*np.pi*fc1*time_axis) + np.real(trans_sig2)*np.cos(2.0*np.pi*fc2*time_axis) - np.imag(trans_sig2)*np.sin(2.0*np.pi*fc2*time_axis)

if SOUND_OUT == 1:
    #Playing sound of carrier wave at fc
    print("Play - CW for User1")
    out_wave = Gain * np.cos(2.0*np.pi*fc1*time_axis)
    stream.write(out_wave.astype(np.float32).tostring())
    sleep(1)
    print("Play - CW for User2")
    out_wave = Gain * np.cos(2.0*np.pi*fc2*time_axis)
    stream.write(out_wave.astype(np.float32).tostring())
    sleep(1)

if SOUND_OUT == 1:
    #Playing sound of transmitted signals
    print("Play - Trans (User1 + User2)")
    out_wave = Gain * tr_waveform
    stream.write(out_wave.astype(np.float32).tostring())
    sleep(1)

#Noise variance : N0/2 * Fs * 1/2
#The constant 1/2 comes from carrier wave
sigma = np.sqrt(1.0/pow(10.0,SNR/10.0) * Fs * 0.5 * 0.5);
waveform = tr_waveform + np.random.normal(0.0,sigma,np.size(tr_waveform))

if SOUND_OUT == 1:
    #Playing Sound of received signals over AWGN channels
    print("Play - Receive")
    out_wave = Gain * waveform
    stream.write(out_wave.astype(np.float32).tostring())
    stream.close()

#Down conversion: RF -> Baseband (Only for User1)
waveform_re = waveform*np.cos(2.0*np.pi*fc1*time_axis)
rec_signal_re = np.convolve(waveform_re,pulse)
waveform_im = - waveform*np.sin(2.0*np.pi*fc1*time_axis)
rec_signal_im = np.convolve(waveform_im,pulse)

#Demodulation***************************
baseband_rec = rec_signal_re[2*DTs*K:NUM_SYM*DTs+2*DTs*K:DTs] + 1j * rec_signal_im[2*DTs*K:NUM_SYM*DTs+2*DTs*K:DTs]
ans = demapping(baseband_rec).astype(dtype=np.int)
error = comparebit(ans.reshape(NUM_SYM,1),data1).sum()

#Showing results************************
print('Center Frequency for User1: ', fc1, '[Hz]')
print('Center Frequency for User2: ', fc2,'[Hz]')
print('Bandwidth: ', 1.0/Ts,'[Hz]')
print('Number of Transmitted Bits:',NUM_SYM*2)
print('Number of Errors of User1:',error)
print('SNR[dB], BER: ',SNR,',',error/(NUM_SYM*2))
