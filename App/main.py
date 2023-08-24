import librosa
import numpy as np
import matplotlib as plt
import seaborn as sns
import sklearn
import pandas as pd
import streamlit as st
@st.cache
def addAWGN(signal, num_bits=16, augmented_num=1, snr_low=15, snr_high=30): 
    signal_len = len(signal)
    noise = np.random.normal(size=(augmented_num, signal_len))
    norm_constant = 2.0**(num_bits-1)
    signal_norm = signal / norm_constant
    noise_norm = noise / norm_constant
    s_power = np.sum(signal_norm ** 2) / signal_len
    n_power = np.sum(noise_norm ** 2, axis=1) / signal_len
    target_snr = np.random.randint(snr_low, snr_high)
    K = np.sqrt((s_power / n_power) * 10 ** (- target_snr / 10))
    K = np.ones((signal_len, augmented_num)) * K  
    return signal + K.T * noise
@st.cache
def zcr(time_series):
  zcrs = librosa.feature.zero_crossing_rate(time_series)
  zcrs_mean = np.mean(zcrs)
  zcrs_std = np.std(zcrs)
  zcrs_max = np.max(zcrs)
  return zcrs,zcrs_mean,zcrs_std,zcrs_max
@st.cache
def plot_zcr(zrcs):
  plt.figure(figsize=(15,5))
  plt.semilogy(zrcs.T, label='Fraction') # apply log transform on y
  plt.ylabel('Fraction per Frame')
  plt.xticks([])
  plt.legend() 
@st.cache
def rmse(time_series):
  return np.sqrt(np.mean(time_series**2))
@st.cache
def cens(time_series,sampling_rate):
  chroma = librosa.feature.chroma_cens(time_series, sampling_rate)
  chroma_mean = np.mean(chroma,axis = 1)
  chroma_std = np.std(chroma,axis = 1)
  return chroma,chroma_mean,chroma_std
@st.cache
def plot_cens_mean(chroma_mean):
  octave = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
  plt.figure(figsize = (15,7))
  plt.title('Mean CENS')
  sns.barplot(x=octave,y=chroma_mean)
@st.cache
def plot_cens_std(chroma_std):
  octave = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
  plt.figure(figsize = (15,7))
  plt.title('Mean CENS')
  sns.barplot(x=octave,y=chroma_std)
@st.cache
def mels(time_series,sampling_rate):
  mel = librosa.feature.melspectrogram(time_series, sampling_rate)
  mel_mean=np.mean(mel.T,axis=0)
  return mel,mel_mean
@st.cache
def plot_mels(mel,sr):
  plt.figure(figsize = (14,14))
  fig, ax = plt.subplots()
  S_dB = librosa.power_to_db(mel, ref=np.max)
  img = librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr,fmax=8000, ax=ax)
  fig.colorbar(img, ax=ax, format='%+2.0f dB')
  ax.set(title='Mel-frequency spectrogram')
@st.cache
def spec_centroid(time_series,sampling_rate):
   spectral_centroids = librosa.feature.spectral_centroid(time_series, sampling_rate)[0]
   spec_centroid_mean = np.mean(spectral_centroids)
   spec_centroid_variance = np.var(spectral_centroids)
   return spectral_centroids,spec_centroid_mean,spec_centroid_variance
@st.cache
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
@st.cache
def plot_spec_centroid(spec_centroids):
  frames = range(len(spec_centroids))
  plt.figure(figsize = (15,7))
  t = librosa.frames_to_time(frames)
  # librosa.display.waveplot(x, sr=sr, alpha=0.4)
  plt.plot(t, normalize(spec_centroids), color='b')
@st.cache
def spec_rolloff(time_series,sampling_rate):
   spectral_rolloff = librosa.feature.spectral_rolloff(time_series+0.01, sampling_rate)[0]
   spec_roll_mean = np.mean(spectral_rolloff)
   spec_roll_variance = np.var(spectral_rolloff)
   return spectral_rolloff,spec_roll_mean,spec_roll_variance
@st.cache
def plot_spec_rolloff(spec_rolloff):
  frames = range(len(spec_rolloff))
  plt.figure(figsize = (15,7))
  t = librosa.frames_to_time(frames)
  # librosa.display.waveplot(x, sr=sr, alpha=0.4)
  plt.plot(t, normalize(spec_rolloff), color='b')
@st.cache
def spec_flux(time_series,sampling_rate):
   spectral_flux = librosa.onset.onset_strength(time_series, sampling_rate)
   spec_flux_mean = np.mean(spectral_flux) 
   spec_flux_variance = np.var(spectral_flux)
   return spectral_flux,spec_flux_mean,spec_flux_variance
@st.cache
def mfcc(time_series,sampling_rate,n_mfccs):
  mfcc = librosa.feature.mfcc(time_series, sampling_rate, n_mfcc=n_mfccs)
  mfcc_mean = np.mean(mfcc.T, axis=0)
  return mfcc,mfcc_mean
@st.cache
def plot_mfcc(mfcc):
  plt.figure(figsize=(15, 7))
  librosa.display.specshow(mfcc, x_axis='time')
  plt.colorbar()
  plt.title('MFCCs') 
@st.cache
def return_features(time_series,sampling_rate,n_mfc):
  zcr_mean = []
  zcr_std = []
  zcr_max = []
  rmses = []
  chroma_mean = []
  chroma_std = []
  mel_mean = []
  spec_centroid_means = []
  spec_centroid_variance = []
  spec_rolloff_means = []
  spec_rolloff_variance = []
  spec_flux_means = []
  spec_flux_variance = []
  mfccs = []

  for i in range(len(time_series)):
    z0,z1,z2,z3 = zcr(time_series[i])
    zcr_mean.append(z1)
    zcr_std.append(z2)
    zcr_max.append(z3)
    rmses.append(rmse(time_series[i]))
    c0,c1,c2 = cens(time_series[i],sampling_rate[i])
    chroma_mean.append(c1)
    chroma_std.append(c2)
    m0,m1 = mels(time_series[i],sampling_rate[i])
    mel_mean.append(m1)
    sc0,sc1,sc2 = spec_centroid(time_series[i],sampling_rate[i])
    spec_centroid_means.append(sc1)
    spec_centroid_variance.append(sc2)
    sr0,sr1,sr2 = spec_rolloff(time_series[i],sampling_rate[i])
    spec_rolloff_means.append(sr1)
    spec_rolloff_variance.append(sr2)
    sf0,sf1,sf2 = spec_flux(time_series[i],sampling_rate[i])
    spec_flux_means.append(sf1)
    spec_flux_variance.append(sf2)
    mc0,mc1 = mfcc(time_series[i],sampling_rate[i],n_mfc)
    mfccs.append(mc1)

  dict1 = {}
  dict1['zcr_mean'] = zcr_mean
  dict1['zcr_std'] = zcr_std
  dict1['zcr_max'] = zcr_max
  dict1['rmse'] = rmses
  dict1['chroma_mean'] = chroma_mean
  dict1['chroma_std'] = chroma_std
  dict1['mel_mean'] = mel_mean
  dict1['spec_centroid_mean'] = spec_centroid_means
  dict1['spec_centroid_variance'] = spec_centroid_variance
  dict1['spec_roll_mean'] = spec_rolloff_means
  dict1['spec_roll_variance'] = spec_rolloff_variance
  dict1['spec_flux_mean'] = spec_flux_means
  dict1['spec_flux_variance'] = spec_flux_variance
  dict1['mfcc'] = mfccs
  return dict1
@st.cache
def make_data(features_dictionary,features_to_choose):
  dataset = pd.DataFrame()
  for i in features_to_choose:
    if(i != 'mfcc' and i != 'chroma_mean' and i != 'chroma_std' and i!='mel_mean'):
      dataset[i] = features_dictionary[i]
    elif(i == 'mfcc'):
      Arr = np.array(features_dictionary[i])
      mfccs_data = pd.DataFrame(Arr,columns = ['m'+str(k) for k in range(len(Arr[0]))])
      dataset = pd.concat([mfccs_data,dataset],axis = 1)
    elif(i == 'chroma_mean'):
      Arr = np.array(features_dictionary[i])
      chroma_mean_data = pd.DataFrame(Arr,columns = ['C_mean','C#_mean','D_mean','D#_mean','E_mean','F_mean','F#_mean','G_mean','G#_mean','A_mean','A#_mean','B_mean'])
      dataset = pd.concat([chroma_mean_data,dataset],axis = 1)
    elif(i == 'chroma_std'):
      Arr = np.array(features_dictionary[i])
      chroma_std_data = pd.DataFrame(Arr,columns = ['C_std','C#_std','D_std','D#_std','E_std','F_std','F#_std','G_std','G#_std','A_std','A#_std','B_std'])
      dataset = pd.concat([chroma_std_data,dataset],axis = 1)
    elif(i == 'mel_mean'):
      Arr = np.array(features_dictionary[i])
      mel_mean_data = pd.DataFrame(Arr,columns = [str(k) for k in range(128)])
      dataset = pd.concat([mel_mean_data,dataset],axis = 1)
  return dataset
