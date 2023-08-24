import streamlit as st

import pickle
import librosa
from main import *
import lightgbm as lgb
import matplotlib.pyplot as plt
import re
import librosa.display
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


st.title("Speech Emotion Recognition")
st.markdown("Here we are using .wav as the input to predict the emotion of speaker")

def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

wav = st.file_uploader("Upload your .wav file",type = ['wav'])
if wav is not None:
        x , sr = librosa.load(wav)
        x = addAWGN(x)[0]
        x_1 = x
        sr_1 = sr
        
        st.audio(wav,format= 'wav')
        features_ = return_features([x],[sr],40)
        time_dataset = make_data(features_,['zcr_mean','zcr_std','zcr_max','rmse'])
        frequency_dataset = make_data(features_,['chroma_mean','chroma_std','mel_mean'])
        spectral_dataset = make_data(features_,['spec_centroid_mean','spec_centroid_variance','spec_roll_mean','spec_roll_variance','spec_flux_mean','spec_flux_variance','mfcc'])
        all_dataset = make_data(features_,['zcr_mean','zcr_std','zcr_max','rmse','chroma_mean','chroma_std','mel_mean','spec_centroid_mean','spec_centroid_variance','spec_roll_mean','spec_roll_variance','spec_flux_mean','spec_flux_variance','mfcc'])
        only_mfcc_dataset = make_data(features_,['mfcc'])



app_mode = st.sidebar.selectbox('Select Page',['Home','Feature Visualisation'])
if(app_mode == 'Home'):
    
    option = st.radio(
    'Select the machine learning model',
    ('Random Forest Classifier', 'LightGBM','QDA'))
    d1_x = pd.read_csv('./datasets/agtd_x_train.csv')
    d1_y = pd.read_csv('./datasets/agtd_y_train.csv')
    d2_x = pd.read_csv('./datasets/agfd_x_train.csv')
    d2_y = pd.read_csv('./datasets/agfd_y_train.csv')
    d3_x = pd.read_csv('./datasets/agsd_x_train.csv')
    d3_y = pd.read_csv('./datasets/agsd_y_train.csv')
    d4_x = pd.read_csv('./datasets/agall_x_train.csv')
    d4_y = pd.read_csv('./datasets/agall_y_train.csv')
    d5_x = pd.read_csv('./datasets/agomfc_x_train.csv')
    d5_y = pd.read_csv('./datasets/agomfc_y_train.csv')



    d1_x = d1_x.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    d1_y = d1_y.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    d2_x = d2_x.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    d2_y = d2_y.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    d3_x = d3_x.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    d3_y = d3_y.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    d4_x = d4_x.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    d4_y = d4_y.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    d5_x = d5_x.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    d5_y = d5_y.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))


    if(option == 'LightGBM'):
            
        model_aug_time = lgb.LGBMClassifier(random_state=39)
        model_aug_time.fit(d1_x.values[:,1:],d1_y.values[:,1:])

        model_aug_frequency = lgb.LGBMClassifier(random_state=39)
        model_aug_frequency.fit(d2_x.values[:,1:],d2_y.values[:,1:])

        model_aug_spectral = lgb.LGBMClassifier(random_state=39)
        model_aug_spectral.fit(d3_x.values[:,1:],d3_y.values[:,1:])

        model_aug_all = lgb.LGBMClassifier(random_state=39)
        model_aug_all.fit(d4_x.values[:,1:],d4_y.values[:,1:])

        model_aug_only_mfcc = lgb.LGBMClassifier(random_state=39)
        model_aug_only_mfcc.fit(d5_x.values[:,1:],d5_y.values[:,1:])
    elif(option == 'Random Forest Classifier'):
        model_aug_time = RandomForestClassifier(random_state=39)
        model_aug_time.fit(d1_x.values[:,1:],d1_y.values[:,1:])

        model_aug_frequency = RandomForestClassifier(random_state=39)
        model_aug_frequency.fit(d2_x.values[:,1:],d2_y.values[:,1:])

        model_aug_spectral = RandomForestClassifier(random_state=39)
        model_aug_spectral.fit(d3_x.values[:,1:],d3_y.values[:,1:])

        model_aug_all = RandomForestClassifier(random_state=39)
        model_aug_all.fit(d4_x.values[:,1:],d4_y.values[:,1:])

        model_aug_only_mfcc = RandomForestClassifier(random_state=39)
        model_aug_only_mfcc.fit(d5_x.values[:,1:],d5_y.values[:,1:])
    elif(option == 'QDA'):
        model_aug_time = QuadraticDiscriminantAnalysis()
        model_aug_time.fit(d1_x.values[:,1:],d1_y.values[:,1:])

        model_aug_frequency = QuadraticDiscriminantAnalysis()
        model_aug_frequency.fit(d2_x.values[:,1:],d2_y.values[:,1:])

        model_aug_spectral = QuadraticDiscriminantAnalysis()
        model_aug_spectral.fit(d3_x.values[:,1:],d3_y.values[:,1:])

        model_aug_all = QuadraticDiscriminantAnalysis()
        model_aug_all.fit(d4_x.values[:,1:],d4_y.values[:,1:])

        model_aug_only_mfcc = QuadraticDiscriminantAnalysis()
        model_aug_only_mfcc.fit(d5_x.values[:,1:],d5_y.values[:,1:])



    
      
    emotion_list = ['neutral','calm','happy','sad','angry','fear','disgust','surprise']
    el = np.array(emotion_list)
    option = st.selectbox(
     'Features to train your model',
     ('Time Domain', 'Frequency Domain', 'Spectral Shape Based','All Features','Only MFCC'))

    st.write('You selected:', option)
    if(wav is not None):
        if(option == 'Time Domain'):
            result = model_aug_time.predict(time_dataset.values)
        elif(option == 'Frequency Domain'):
            result = model_aug_frequency.predict(frequency_dataset.values)
        elif(option == 'Spectral Shape Based'):
            result = model_aug_spectral.predict(spectral_dataset.values)
        elif(option == 'All Features'):
            result = model_aug_all.predict(all_dataset.values)
        elif(option == 'Only MFCC'):
            result = model_aug_only_mfcc.predict(only_mfcc_dataset.values)
        if(result!=None):
                if(result[0] == 1):
                    st.markdown(""" <style> .font {
                    font-size:50px ;} 
                    </style> """, unsafe_allow_html=True)
                    st.markdown(":neutral_face:")
                elif(result[0] == 2):
                    st.markdown(""" <style> .font {
                    font-size:50px ;} 
                    </style> """, unsafe_allow_html=True)
                    st.markdown(":relieved:")
                elif(result[0] == 3):
                    st.markdown(""" <style> .font {
                    font-size:50px ;} 
                    </style> """, unsafe_allow_html=True)
                    st.markdown(":laughing:")
                elif(result[0] == 4):
                    st.markdown(""" <style> .font {
                    font-size:50px ;} 
                    </style> """, unsafe_allow_html=True)
                    st.markdown(":disappointed:")
                elif(result[0] == 5):
                    st.markdown(""" <style> .font {
                    font-size:50px ;} 
                    </style> """, unsafe_allow_html=True)
                    st.markdown(":rage:")
                elif(result[0] == 6):
                    st.markdown(""" <style> .font {
                    font-size:50px ;} 
                    </style> """, unsafe_allow_html=True)
                    st.markdown(":fearful:")
                elif(result[0] == 7):
                    st.markdown(""" <style> .font {
                    font-size:50px ;} 
                    </style> """, unsafe_allow_html=True)
                    st.markdown(":face_vomiting:")
                elif(result[0] == 8):
                    st.markdown(""" <style> .font {
                    font-size:50px ;} 
                    </style> """, unsafe_allow_html=True)
                    st.markdown(":open_mouth:")
                st.subheader("Predicted Emotion : " + str(emotion_list[result[0]-1]))
                
                st.download_button(
                            label="Download Metadata",
                            data=convert_df(all_dataset),
                            file_name='metadata_df.csv',
                            mime='text/csv',
                        )
                
                
               






        

elif(app_mode == 'Feature Visualisation'):
    st.title("Visualisation")
    st.header('Time Domain features')
    st.subheader("Zero Crossing Rate")
    z1,z2,z3,z4 = zcr(x_1)
    fig, ax = plt.subplots(figsize = (9,2))
    ax.semilogy(z1.T, label='Fraction') # apply log transform on y
    ax.set_ylabel('Fraction per Frame')
    ax.set_xticks([])
    ax.legend() 
    st.pyplot(fig)
    st.header("Frequency Domain features")
    st.subheader("Mean CENS")
    c0,c1,c2 = cens(x_1,sr_1)
    fig1,ax1 = plt.subplots(figsize = (9,3.5))
    octave = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    sns.barplot(x=octave,y=c1)
    st.pyplot(fig1)
    st.subheader("Standard Deviation CENS")
    fig2,ax2 = plt.subplots(figsize = (9,3.5))
    sns.barplot(x = octave,y = c2)
    st.pyplot(fig2)
    m1,m2 = mels(x,sr)
    st.subheader("Mel Frequency Spectogram")
    fig3,ax3 = plt.subplots(figsize = (8,4))
    S_dB = librosa.power_to_db(m1, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr,fmax=8000, ax=ax3)
    fig3.colorbar(img, ax=ax3, format='%+2.0f dB')
    ax3.set(title='Mel-frequency spectrogram')
    st.pyplot(fig3)
    st.header("Spectral Shape Based features")
    st.subheader("Spectral Centroids")
    fig4,ax4 = plt.subplots(figsize = (9,2))
    s0,s1,s2 = spec_centroid(x,sr)
    frames = range(len(s0))
    t = librosa.frames_to_time(frames)
    # librosa.display.waveplot(x, sr=sr, alpha=0.4)
    plt.plot(t, normalize(s0), color='b')
    st.pyplot(fig4)
    st.subheader("Spectral Rolloff")
    fig5,ax5 = plt.subplots(figsize = (9,2))
    sr0,sr1,sr2 = spec_rolloff(x,sr)
    frames_r = range(len(sr0))
    t = librosa.frames_to_time(frames_r)
    # librosa.display.waveplot(x, sr=sr, alpha=0.4)
    plt.plot(t, normalize(sr0), color='g')
    st.pyplot(fig5)
    st.subheader("Spectral Flux")
    fig6,ax6 = plt.subplots(figsize = (9,2))
    sf0,sf1,sf2 = spec_flux(x,sr)
    frames_f = range(len(sf0))
    t = librosa.frames_to_time(frames_f)
    # librosa.display.waveplot(x, sr=sr, alpha=0.4)
    plt.plot(t, normalize(sf0), color='r')
    st.pyplot(fig6)
    st.subheader("Mel Frequency cepstral coefficients")
    fig7,ax7 = plt.subplots()
    nmfcc = st.slider('Select n_mfcc', 10,100,40)
    m0,m1 = mfcc(x,sr,nmfcc)
    librosa.display.specshow(m0, x_axis='time')
    plt.colorbar()
    plt.title('MFCCs') 
    st.pyplot(fig7)
    


















    




            
            

             





