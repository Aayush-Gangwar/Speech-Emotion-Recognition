# Speech-Emotion-Recognition

* The idea behind creating this project was to build a machine learning model that could detect emotions from the speech.

## Analyzing audio signals
![ser](https://github.com/Aayush-Gangwar/Speech-Emotion-Recognition/assets/101112022/0665cb27-72bb-4251-9a3f-d59f5dab1525)


### Datasets:
* RAVDESS: 
This dataset includes around 1400 audio file input from 24 different actors. 12 male and 12 female where these actors record short audios in 8 different emotions i.e 1 = neutral, 2 = calm, 3 = happy, 4 = sad, 5 = angry, 6 = fearful, 7 = disgust, 8 = surprised.<br>
Each audio file is named in such a way that the 7th character is consistent with the different emotions that they represent.


## Feature Extraction
The next step involves extracting the features from the audio files which will help our model learn between these audio files.
For feature extraction we make use of the [**LibROSA**](https://librosa.github.io/librosa/) library in python which is one of the libraries used for audio analysis. 
- ### TIME DOMAIN FEATURES:
  - Zero Crossing Rate
  - Root Mean Square Energy

- ### FREQUENCY DOMAIN FEATURES:
    - Mel spectogram features  
    - MFCC (Mel frequency cepstral coefficients)
- Mel frequency cepstral coefficients (MFCC)

## Building Models
### Algorithms Used
- Decision Tree
- RandomForestClassifier
- GradientBoostingClassifier
- KNeighborsClassifier
- MLPClassifier
- SVC
- LightGBM
- QDA
<br>

## Predictions

After tuning the model, tested it out by predicting the emotions for the test data. For a model with the given accuracy these are a sample of the actual vs predicted values.

## Testing out with live voices.
You can test your own voice by deployed website : 
[Speech Emotion Recognition Web]([images/livevoice.PNG?raw=true](https://share.streamlit.io/11happy/prml_course_project/main/app.py))

## Conclusion
Building the models was a challenging task as it involved lot of trail and error methods, tuning etc. The model was tuned to detect emotions with more than 70% accuracy. Accuracy can be increased by including more audio files for training.
