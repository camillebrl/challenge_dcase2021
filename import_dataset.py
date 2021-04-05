import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

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


def preprocessing_train(task):
    """importe les données et fait le préprocessing pour le training et la validation

    Args:
        task (string): "train" pour les données de training et "evaluate" pour les données d'évaluation
    """
    ThisPath = "./data/data_challenge/"
    file_to_use = f"./data/data_challenge/fold1_{task}.csv"
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

    # Construction du dataset
    y=[]
    dict_features={"ville":[],"device":[],"spectre_lm_and_delta":[]}
    for i in range(len(wavpaths)):
        category=df["category"][i]
        y.append(category)
        name=df['filename'][i]
        ville=name[findnth_left(name,"-",0):findnth_right(name,"-",1)]
        dict_features["ville"].append(ville) # il faudra mettre le nom de la ville en int (variable catégorielle)
        device=name[findnth_left(name,"-",3):findnth_right(name,".",0)]
        dict_features["device"].append(device) # il faudra mettre le nom du device en int (variable catégorielle)
        son_0,fs=sound.read(ThisPath + name,stop=SampleDuration*sr)
        log_mel=np.log(librosa.feature.melspectrogram(son_0,
                                        sr=sr,
                                        n_fft=NumFFTPoints,
                                        hop_length=HopLength,
                                        n_mels=NumFreqBins,
                                        fmin=0.0,
                                        fmax=sr/2,
                                        htk=True,
                                        norm=None))
        delta_log_mel=deltas(log_mel)
        delta_delta_log_mel = deltas(delta_log_mel)
        log_mels_concatenated = np.concatenate((log_mel[4:-4],delta_log_mel[2:-2],delta_delta_log_mel),axis=-1)
        dict_features["spectre_lm_and_delta"].append(log_mels_concatenated)
    x=pd.DataFrame.from_dict(dict_features)
    return x,y

def preprocessing_test():
    ThisPath = "./data/data_challenge/"
    file_to_use = "./data/data_challenge/fold1_test.csv"
    sr = 44100 # nombre total de points de l'audio
    SampleDuration = 10 # taille totale des audios (en secondes)
    NumFreqBins = 128 # nb de fréquences différentes dans les audios
    NumFFTPoints = 2048 # nombre de points qu'il y a dans chaque fenêtre du spectrogramme. On fait le spectrogramme sur 2048 points autour de t (de t-1024,t+1024)
    HopLength = int(NumFFTPoints/2) # tous les 1024 points, on fait un nouveau spectrogramme
    NumTimeBins = int(np.ceil(SampleDuration*sr/HopLength))

    df = pd.read_csv(file_to_use,sep='\t', encoding='ASCII')
    wavpaths = df['filename'].tolist()

    # Construction du dataset
    dict_features={"ville":[],"device":[],"spectre_lm_and_delta":[]}
    for i in range(len(wavpaths)):
        name=df['filename'][i]
        ville=name[findnth_left(name,"-",0):findnth_right(name,"-",1)]
        dict_features["ville"].append(ville) # il faudra mettre le nom de la ville en int (variable catégorielle)
        device=name[findnth_left(name,"-",3):findnth_right(name,".",0)]
        dict_features["device"].append(device) # il faudra mettre le nom du device en int (variable catégorielle)
        son_0,fs=sound.read(ThisPath + name,stop=SampleDuration*sr)
        log_mel=np.log(librosa.feature.melspectrogram(son_0,
                                        sr=sr,
                                        n_fft=NumFFTPoints,
                                        hop_length=HopLength,
                                        n_mels=NumFreqBins,
                                        fmin=0.0,
                                        fmax=sr/2,
                                        htk=True,
                                        norm=None))
        delta_log_mel=deltas(log_mel)
        delta_delta_log_mel = deltas(delta_log_mel)
        log_mels_concatenated = np.concatenate((log_mel[4:-4],delta_log_mel[2:-2],delta_delta_log_mel),axis=-1)
        dict_features["spectre_lm_and_delta"].append(log_mels_concatenated)
    x=pd.DataFrame.from_dict(dict_features)
    return x