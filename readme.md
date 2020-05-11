<h1><strong>Simple examples of RNNs</strong></h1>
<h2><strong>I. IMDB sentiment classification:</strong></h2>
<h3>1. Dataset:</h3>
<p>In this section, we apply RNNs (particularly LSTM) to IMDB dataset. The original dataset can be found <a href="https://www.imdb.com/interfaces/">here</a> on the home page.</p>
<p>Luckily, Keras has a built-in IMDB dataset, by inserting this line of code to your source:</p>
<p>from keras.datasets import imdb</p>
<p>&nbsp;</p>
<h3>2. Load the dataset</h3>
<p>vocabulary_size = 5000</p>
<p>(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocabulary_size)</p>
<p>The dataset contains 2 columns: review column is the comments from reviewers and the sentiment column (0 for negative and 1 for positive)</p>
<p>Note that the review is stored as a sequence of integers. These are word IDs that have been pre-assigned to individual words, and the label is an integer.</p>
<p><img src="https://github.com/nvty13/rnn-examples/blob/master/images/IMDB.png?raw=true" alt="" width="100%" /></p>
<p>&nbsp;3. The model structure:</p>
<pre>Model: "sequential_8"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_7 (Embedding)      (None, 500, 32)           160000    
_________________________________________________________________
lstm_7 (LSTM)                (None, 100)               53200     
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 101       
=================================================================
Total params: 213,301
Trainable params: 213,301
Non-trainable params: 0</pre>
<p><img src="https://github.com/nvty13/rnn-examples/blob/master/images/JPEG%20image-DE00FD10A46B-1.jpeg?raw=true" alt="" width="100%" /></p>
<p>5. Training and testing:</p>
<p>num_epochs = 3</p>
<p># If exist a saved model:<br />if path.exists("rnn.h5"):<br /> model.load_weights("rnn.h5")<br />else:<br /> # Otherwise, train from scratch<br /> model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=num_epochs)<br /> model.save("rnn.h5")</p>
<p>&nbsp;</p>
<p>6. Accuracy:</p>
<p>&nbsp;</p>
<p><strong>2. VCB stock prediction</strong></p>
<p>&nbsp;</p>
<div id="gtx-anchor" style="position: absolute; visibility: hidden; left: 8px; top: 1597.36px; width: 602.578px; height: 143px;">&nbsp;</div>
<div class="jfk-bubble gtx-bubble" style="visibility: visible; left: -14px; top: 1252px; opacity: 1;" role="alertdialog" aria-describedby="bubble-11">&nbsp;</div>
