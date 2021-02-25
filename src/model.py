import pickle
import os
import cv2
import numpy as np
from keras.models import load_model

basedir = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(basedir,'pickle','cnn_model.pkl')

model = pickle.load(open(path, 'rb'))
model_f = load_model(os.path.join(basedir,'pickle','cnn_modell.h5'))

def cnn_model_prediction(path):
    pic = cv2.imread(path)
    pic = cv2.resize(pic,(256,256))
    pic = np.reshape(pic,[1,256,256,3])
    y_prob = model_f.predict(pic) 
    y_classes = y_prob.argmax(axis=-1)
    message = ''
    if(y_classes==4):
        message = "Normal Leaf- No pesticide needed."
    elif(y_classes==3):
        message = "Infected with Nematodes- Use Non fumigant systemic nematicides like Carbofuran 3G @ 3g/m, aldicarb snd phenamiphos"
    elif(y_classes==5):
        message = "Infected with Virus(Blight disese- Spray Imidachloprid 0.33 ml or Dimethiate 1 ml and neem oil 2 ml/ litre water mixed with surfactant at 20 days after sowing."
    elif(y_classes==1):
        message = "Infected with Bacteria(Leaf spot disease)- Use Carbendazim 1gm or Mancozeb 2gm or Chlorothalanil 2g, hexaconazole 2 ml/litre of water at 15 days interval starting from 4-5 weeks after planting."
    elif(y_classes==2):
        message = "Infected with Fungus(Tikka disease)- spray application of Carbendazim 1gm or Mancozeb 2gm or Chlorothalanil 2g, hexaconazole 2 ml/litre of water at 15 days interval starting from 4-5 weeks after planting."
    elif(y_classes==0:
        message = "Something wrong"
    else:
        message = "Something went wrong"
    return message   