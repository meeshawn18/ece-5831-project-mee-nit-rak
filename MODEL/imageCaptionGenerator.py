# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:22:24 2022

@author: Meeshawn Nithesh Raksha
"""

#%%  Import necessary packages
import numpy as np
import string
import matplotlib.pyplot as plt
import os
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
# from tensorflow.keras.layers import TextVectorization

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Dense, LSTM, Embedding, Add
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from tqdm import tqdm

seed = 111
np.random.seed(seed)
tf.random.set_seed(seed)

#%% Defining Model Constants 

# Path to the images
IMAGES_PATH = '\dataset\Flickr8k_Dataset\Flicker8k_Dataset'
IMAGES_FOLDER = os.getcwd() + IMAGES_PATH

# Path to image captions
CAPTIONS_PATH = '\dataset\Flickr8k_text'
CAPTIONS_FOLDER = os.getcwd() + CAPTIONS_PATH

# Desired image dimensions
IMAGE_SIZE = (299, 299)

# Vocabulary size
VOCAB_SIZE = 10000

# Fixed length allowed for any sequence
SEQ_LENGTH = 25

# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512

# Per-layer units in the feed-forward network
FF_DIM = 512

# Other training parameters
BATCH_SIZE = 64
EPOCHS = 30
AUTOTUNE = tf.data.experimental.AUTOTUNE

#%% Reading and storing the image filenames

def extractName(filename):
    file = open(filename, 'r')
    text = file.read()
    text = text.split('\n')
    file.close()
    return text    
    
#%%
train_imgs = extractName('./dataset/Flickr8k_text/Flickr_8k.trainImages.txt')
train_imgs = [x for x in train_imgs if x != '']

test_imgs = extractName('./dataset/Flickr8k_text/Flickr_8k.testImages.txt')
test_imgs = [x for x in test_imgs if x != '']

dev_imgs = extractName('./dataset/Flickr8k_text/Flickr_8k.devImages.txt')
dev_imgs = [x for x in dev_imgs if x != '']

#%% Loading images and extracting final layer features/weights

def extractFinalLayer(IMAGES_FOLDER, img_name, model):
    # Convert all the images to size 299x299 as expected by the
    # inception v3 model
    img = load_img(os.path.join(IMAGES_FOLDER, img_name), target_size=IMAGE_SIZE)
    # Convert PIL image to numpy array of 3-dimensions
    x = img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess images using preprocess_input() from inception module
    x = preprocess_input(x)
    x = model.predict(x)
    # reshape from (1, 2048) to (2048, )
    x = np.reshape(x, x.shape[1])
    return x


#%%
# Create an instance of the Inception V3 network
model_inceptionv3 = InceptionV3(weights='imagenet')
model_inceptionv3 = Model(model_inceptionv3.input, model_inceptionv3.layers[-2].output) 
finalLayer = extractFinalLayer(IMAGES_FOLDER,  train_imgs[0], model_inceptionv3)
# print(finalLayer.shape)

#%%

dict_image_eigen_vector = {}
def featureExtractions(images):
    for image in tqdm(images):
        image_eigen_vectors = extractFinalLayer(IMAGES_FOLDER, image, model_inceptionv3)
        dict_image_eigen_vector[image] = image_eigen_vectors


#%% DO NOT RUN THIS!!!!!!!!

featureExtractions(train_imgs)

#%% Saving the 2048 length image feature vector to a pickle file

with open('gil_strang.pkl', 'wb') as f:
    pickle.dump(dict_image_eigen_vector, f)
f.close()

#%%
with open('gil_strang.pkl', 'rb') as f:
    gil_strang = pickle.load(f)
f.close()



#%% Read and store the image captions into a dictionary

file = open('./dataset/Flickr8k_text/Flickr8k.token.txt', 'r')
print('Reading and storing the image filenames and the corresponding captions\n' )

dict_descriptions = {}
for line in file:
    sentence = line.strip()
    sentence = sentence.split ('\t')   
    
    img_file_name = sentence[0].split('.')[0]
    caption = sentence[1]
    
    if dict_descriptions.get(img_file_name) == None:
        dict_descriptions[img_file_name] = []
     
    caption = 'startseq' + ' ' + caption + ' ' + 'endseq'
    dict_descriptions[img_file_name].append(caption)
    
file.close()
#%% Pre-processing/cleaning the captions:
maxLength = 0
print('Pre-processing and cleaning the captions')
for file, captions in dict_descriptions.items():
    for idx in range(len(captions)):       
        captions[idx] = captions[idx].lower()
        captions[idx] = captions[idx].translate(str.maketrans('', '', string.punctuation))
        captions[idx] = [word for word in captions[idx].split(' ') if len(word)>1]
        captions[idx] = [word for word in captions[idx] if word.isalpha()]
        captions[idx] = ' '.join(captions[idx])
        currLen = len(captions[idx].split(' '))                
        if currLen > maxLength:
                maxLength = currLen

#%% Create a dictionary of unique words:
vocabulary = {}
for key, captions in dict_descriptions.items():
    for caption in captions:
        for word in caption.split(' '):
            vocabulary[word] = vocabulary.get(word, 0) + 1

#%%
# word_count_thresh = 10
# reduced_vocabulary = []

# for word, count in vocabulary.items():
#     if count >= word_count_thresh:
#         print(word)
#         reduced_vocabulary.append(word)
        
# #%% Writing the vocab list
# with open('VocabList.txt', 'w') as f:
#     for word in vocabulary:
#         f.write(word)
#         f.write('\n')
# f.close()

#%%
wrd2idx = {}
idx2wrd = {}
idx = 1
for word in vocabulary:
    wrd2idx[word] = idx
    idx2wrd[idx] = word
    idx += 1
        
#%% Creating Word embeddings for all the words in the vocabulary:
    
glove_embeddings = {}
with open('./dataset/glove.6B/glove.6B.200d.txt', 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        sentence = line.strip()
        sentence = sentence.split()
        word = sentence[0]
        feature_vector = sentence[1:]
        # print(word)
        glove_embeddings[word] = np.asarray(feature_vector, dtype='float32')
f.close()
#%% Word embeddings for all the unique words in the captions/vocabulary
dim_glove_vector = 200
count = 0
vocab_embeddings = np.zeros((len(vocabulary)+1, dim_glove_vector))
for word in tqdm(vocabulary):
    if word in glove_embeddings:
        vocab_embeddings[wrd2idx[word]] = glove_embeddings[word]
    else:
        count = count+1
#%%      
with open('LSTM_vocab_embeddings.pkl', 'wb') as f:
    pickle.dump(vocab_embeddings, f)
f.close()

#%%
with open('LSTM_vocab_embeddings.pkl', 'rb') as f:
    vocab_embeddings = pickle.load(f)
f.close()

#%%
def data_prep(dict_descriptions, dict_image_eigen_vector, maxLength, num_batch_size):
    X1 = [] # Image input feature/eigen vector
    X2 = [] # Input sequence
    Y = [] # Target word/output seq
    # Looping through every image
    n = 0
    while True:
        for file_name, feature_vector in dict_image_eigen_vector.items():
            n += 1
            print(file_name)
            captions = dict_descriptions[file_name.split('.')[0]]
            for caption in captions:
                seq = [wrd2idx[word] for word in caption.split()]
                # Creating input-output datapoints
                for idx in range(1, len(seq)):
                    partial_caption = seq[:idx]    
                    target_word = seq[idx]
                    partial_caption = pad_sequences([partial_caption], maxlen=maxLength)[0]
                    target_word = to_categorical([target_word], num_classes=len(vocabulary))[0]
                    X1.append(feature_vector) 
                    X2.append(partial_caption)
                    Y.append(target_word)
                    
            if n==num_batch_size:
                n=0
                yield [[np.array(X1),np.array(X2)],np.array(Y)]
                X1, X2, Y  = list(), list(), list()
                
#%%                
final_generator = data_prep(dict_descriptions,dict_image_eigen_vector,maxLength,num_batch_size=5)

inputs, outputs = next(final_generator)

print(inputs[0].shape)
print(inputs[1].shape)
print(outputs.shape)

#%%
# image feature extractor model
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# partial caption sequence model
inputs2 = Input(shape=(maxLength,))
se1 = Embedding(len(vocabulary), dim_glove_vector, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
# decoder (feed forward) model
decoder1 = Add()([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(len(vocabulary), activation='softmax')(decoder2)
# merge the two input models
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
# use pre-fixed weights for embeddding layer and not trainable.
model.layers[2].set_weights([vocab_embeddings])
model.layers[2].trainable = False
# model compile
model.compile(loss='categorical_crossentropy', optimizer='adam')



                            
                

                
        
        
    



    
    
    
    