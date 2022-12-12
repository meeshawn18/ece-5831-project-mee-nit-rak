# IMAGE CAPTION GENERATOR
## ECE-5831 Pattern Recognition and Neural Network (FALL 2022)

### Authors: Meeshawn Marathe, Nithesh Veerappa, Raksha Varahamurthy

In this work, we perform the supervised learning
of natural language caption generation from a set of images
annotated with their corresponding captions. This task of gener-
ating captions is realised by constructing a multimodel neural
network architecture that cascades CNN and LSTM models
in order to perform feature extraction from images and text
sequence learning from their corresponding captions respectively.
Transfer learning is utilized to extract salient features from the
images (Flicker8K dataset) with the help of a pre-trained CNN
model (InceptionV3 pre-trained on imagenet). Text features from
the captions are obtained with the help of GloVe embeddings
and text sequence learning is then achieved by providing the
glove embeddings for the vocabulary to the LSTM for language
learning. Finally, text and image features are concatenated to
input a neural network which is trained to achieve caption
generation for unseen images.
