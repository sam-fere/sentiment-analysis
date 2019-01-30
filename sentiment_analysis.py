# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 17:16:15 2019

@author: sam
"""

import numpy as np # linear algebra

import pickle

from keras.preprocessing.sequence import pad_sequences

import sys

from textblob import TextBlob

from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic
" import the gui design file"

Ui_MainWindow, QtBaseClass = uic.loadUiType('gui/gui.ui')



"create main application class"
class MyApp(QMainWindow):
	def __init__(self):
		super(MyApp, self).__init__()
		#read main window from gui file"
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)
		#analyze button action"
		self.ui.analyzeButton.clicked.connect(self.display)
		#reset button action"
		self.ui.resetButton.clicked.connect(self.reset)

	#display function that contains all the codes"
	def display(self):
		# read text from the text box"
		twt = [self.ui.textEdit.toPlainText()]
		
		langs=TextBlob(twt[0])
		#detect the language type"
		lang=langs.detect_language()
		#the function if the detected language is arabic"
		if lang=='ar':
			#load arabic model
		    with open('models/arabic_model.pickle', 'rb') as handle:
		        model = pickle.load(handle)
			# load arabic texet tokenizer 
		    with open('models/arabic_tokenizer.pickle', 'rb') as handle:
		        tokenizer = pickle.load(handle)
		    twt = tokenizer.texts_to_sequences(twt)
		    #padding the tweet to have exactly the same shape as `embedding_2` input
		    twt = pad_sequences(twt, maxlen=154, dtype='int32', value=0)
		    
		    sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
		    
			# 
		    if(np.argmax(sentiment) == 0):
			    
			    sent="negative"
		    elif (np.argmax(sentiment) == 1):
			    
			    sent="positive"
		elif lang =='en':
		    with open('models/english_model.pickle', 'rb') as handle:
		        model = pickle.load(handle)
		    with open('models/english_tokenizer.pickle', 'rb') as handle:
		        tokenizer = pickle.load(handle)
		    twt = tokenizer.texts_to_sequences(twt)
		    #padding the tweet to have exactly the same shape as `embedding_2` input
		    twt = pad_sequences(twt, maxlen=32, dtype='int32', value=0)
		    print(lang)
		    sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
		    
			# assign the sentiment result to sent
		    if(np.argmax(sentiment) == 0):
			    
			    sent="negative"
		    elif (np.argmax(sentiment) == 1):
			    
			    sent="positive"

		# display language value in language text box
		self.ui.langEdit.setText(lang)
		# display sentiment result value in result text box
		self.ui.resultEdit.setText(sent)



	def reset(self):
		# reset values of text box, language box and result box
	    self.ui.textEdit.setText('')
	    self.ui.langEdit.setText('')
	    self.ui.resultEdit.setText('')

# run the application
if __name__ == '__main__':
	app = QApplication(sys.argv)
	window = MyApp()
	window.show()
	sys.exit(app.exec_())





