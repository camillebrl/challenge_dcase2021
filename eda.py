import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import soundfile as sound
import librosa
import librosa.display
import IPython.display as ipd
import base64
import io
import urllib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def deltas(X_in):
  """Calcule le delta du log-mel spectrogramme

  Args:
      X_in (list): log-mel spectrogramme

  Returns:
      list: delta log-mel spectrogramme
  """
  X_out = (X_in[2:]-X_in[:-2])/10.0
  X_out = X_out[1:-1]+(X_in[4:]-X_in[:-4])/5.0
  return X_out

def convert(fig):
  """Convertit un objet matplotlib en html (pour visualiser les images du dataframe)

  Args:
      fig (matplotlib.pyplot): mettre l'image à convertir

  Returns:
      html file: retourne un fichier html
  """
  buf = io.BytesIO()
  fig.savefig(buf, format='png')
  buf.seek(0)
  string = base64.b64encode(buf.read())
  uri = 'data:image/png;base64,' + urllib.parse.quote(string)
  html = '<img src = "%s"/>' % uri
  return html

def findnth_left(mot, needle, n):
  """récupère le n-ième emplacement de needle dans un mot (en prenant en compte le caractère)

  Args:
      mot (string): la phrase / mot
      needle (char): le caractère dont on cherche l'emplacement dans le mot
      n (int): on veut le n-ième caractère needle du mot / phrase

  Returns:
      int: la position de l'élément souhaité dans le mot
  """
  parts= mot.split(needle, n+1)
  if len(parts)<=n+1:
      return -1
  return len(mot)+1-len(parts[-1])-len(needle)

def findnth_right(mot, needle, n):
  """récupère le n-ième emplacement de needle dans un mot (sans prendre en compte le caractère)

  Args:
      mot (string): la phrase / mot
      needle (char): le caractère dont on cherche l'emplacement dans le mot
      n (int): on veut le n-ième caractère needle du mot / phrase

  Returns:
      int: la position de l'élément souhaité dans le mot
  """
  parts= mot.split(needle, n+1)
  if len(parts)<=n+1:
      return -1
  return len(mot)-len(parts[-1])-len(needle)

########################################################
########### Familiarisation avec les données ###########
########################################################

# Chargement des données
ThisPath = "./data/data_challenge/"
file_to_use = "./data/data_challenge/fold1_evaluate.csv"
sr = 44100 # nombre total de points de l'audio
SampleDuration = 10 # taille totale des audios (en secondes)
NumFreqBins = 128 # nb de fréquences différentes dans les audios
NumFFTPoints = 2048 # nombre de points qu'il y a dans chaque fenêtre du spectrogramme. On fait le spectrogramme sur 2048 points autour de t (de t-1024,t+1024)
HopLength = int(NumFFTPoints/2) # tous les 1024 points, on fait un nouveau spectrogramme
NumTimeBins = int(np.ceil(SampleDuration*sr/HopLength))

df = pd.read_csv(file_to_use,sep='\t', encoding='ASCII')
wavpaths = df['filename'].tolist()
ClassNames = np.unique(df['scene_label'])
y_val_labels =  df['scene_label'].astype('category').cat.codes.values
df["category"]=y_val_labels

# Représentation des données
dict_son={"name":[],"ville":[],"device":[],"category":[],"image_brute":[],"spectre_lm":[],"stft":[],"spectre_lm_and_delta":[]}
# for i in range(len(wavpaths)-1):
for i in range(300):
    print(i)
    name=df['filename'][i]
    dict_son["name"].append(name)
    ville=name[findnth_left(name,"-",0):findnth_right(name,"-",1)]
    dict_son["ville"].append(ville)
    print(ville)
    # category=df["category"][i]
    # dict_son["category"].append(category)
    # print(category)
    category=df['scene_label'][i]
    dict_son["category"].append(category)
    device=name[findnth_left(name,"-",3):findnth_right(name,".",0)]
    dict_son["device"].append(device)
    son_0,fs=sound.read(ThisPath + name,stop=SampleDuration*sr)
    fig_brute=plt.figure(figsize=(14, 5))
    librosa.display.waveplot(son_0, sr=sr)
    html_image_brute=convert(fig_brute)
    dict_son["image_brute"].append(html_image_brute)
    plt.clf()
    plt.close(fig_brute)
    fig_brute.clear()
    del fig_brute
    X = librosa.stft(son_0)
    Xdb = librosa.amplitude_to_db(abs(X))
    fig_stft=plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    html_ftft=convert(fig_stft)
    dict_son["stft"].append(html_ftft)
    plt.clf()
    plt.close(fig_stft)
    fig_stft.clear()
    del fig_stft
    results_log_mel=np.log(librosa.feature.melspectrogram(son_0,
                                    sr=sr,
                                    n_fft=NumFFTPoints,
                                    hop_length=HopLength,
                                    n_mels=NumFreqBins,
                                    fmin=0.0,
                                    fmax=sr/2,
                                    htk=True,
                                    norm=None))
    Xdb = librosa.amplitude_to_db(abs(results_log_mel))
    fig=plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    html_image=convert(fig)
    dict_son["spectre_lm"].append(html_image)
    plt.clf()
    plt.close(fig)
    fig.clear()
    del fig
    results_delta_log_mel=deltas(results_log_mel)
    results_deltas_deltas_log_mel = deltas(results_delta_log_mel)
    results_concatenated = np.concatenate((results_log_mel[4:-4],results_delta_log_mel[2:-2],results_deltas_deltas_log_mel),axis=-1)
    Xdb = librosa.amplitude_to_db(abs(results_concatenated))
    fig_concatenated=plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    html_image_concatenated=convert(fig_concatenated)
    dict_son["spectre_lm_and_delta"].append(html_image_concatenated)
    plt.clf()
    plt.close(fig_concatenated)
    fig_concatenated.clear()
    del fig_concatenated
    # length=librosa.get_duration(son_0,sr=sr,n_fft=NumFFTPoints,hop_length=HopLength)
    # dict_son["length"].append(length)

html=pd.DataFrame.from_dict(dict_son).set_index("name").to_html("./visualisation/validation.html",escape=False)