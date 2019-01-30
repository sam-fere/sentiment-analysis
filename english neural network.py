
# coding: utf-8

# In[23]:


import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pickle



# Read the dataset:



tweets = pd.read_csv('Tweets.csv', sep=',')
# Select only interestig fields:



data = tweets[['text','airline_sentiment']]


# Clean up the dataset, conidering only positive and negative tweets:


tweet_text=data['text'][data.airline_sentiment != "neutral"]
tweet_label=data['airline_sentiment']
data = data[data.airline_sentiment != "neutral"]
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

print(data[ data['airline_sentiment'] == 'positive'].size)
print(data[ data['airline_sentiment'] == 'negative'].size)


# Tokenization:

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)



# ## Neural network:

embed_dim = 128
lstm_out = 196
batch_size = 512
model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(Dropout(0.5))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation=tf.nn.softmax))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# declaring dataset



Y = pd.get_dummies(data['airline_sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# Selecting some data for training and some for validation

X_val = X_train[:500]
Y_val = Y_train[:500]

partial_X_train = X_train[500:]
partial_Y_train = Y_train[500:]


# ## Train the network:

history = model.fit(partial_X_train,
                    partial_Y_train,
                    epochs = 30,
                    batch_size=batch_size,
                    validation_data=(X_val, Y_val))

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# ## Validation


pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_val)):

    result = model.predict(X_val[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]

    if np.argmax(result) == np.argmax(Y_val[x]):
        if np.argmax(Y_val[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1

    if np.argmax(Y_val[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1



print ("pos_acc", 100* pos_correct/pos_cnt, "%")
print ("neg_acc", 100 * neg_correct/neg_cnt, "%")
score, acc = model.evaluate (X_test,Y_test,verbose=2, batch_size=batch_size)
print ("Score: %.2f"%(score))
print ("Validation Accuracy %.2f" % (acc))
with open('english_model.pickle', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('english_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

