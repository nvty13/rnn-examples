<h1><strong>Simple examples of RNNs</strong></h1>
<h2><strong>I. IMDB sentiment classification:</strong></h2>
<h3>1. Dataset:</h3>
<p>In this section, we apply RNNs (particularly LSTM) to IMDB dataset. The original dataset can be found <a href="https://www.imdb.com/interfaces/">here</a> on the home page.</p>
<p>Luckily, Keras has a built-in IMDB dataset, by inserting this line of code to your source:</p>
<p>from keras.datasets import imdb</p>
<p>&nbsp;</p>
<h3>2. Load the dataset</h3>
<pre>vocabulary_size = 5000<br /><br />(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocabulary_size)</pre>
<p>The dataset contains 2 columns: review column is the comments from reviewers and the sentiment column (0 for negative and 1 for positive)</p>
<p>Note that the review is stored as a sequence of integers. These are word IDs that have been pre-assigned to individual words, and the label is an integer.</p>
<p><img src="https://github.com/nvty13/rnn-examples/blob/master/images/IMDB.png?raw=true" alt="" width="100%" /></p>
<h3>&nbsp;3. The model structure:</h3>
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
<h3>5. Training and testing:</h3>
<pre>num_epochs = 3<br /><br /># If exist a saved model:<br />if path.exists("rnn.h5"):<br /> model.load_weights("rnn.h5")<br />else:<br /> # Otherwise, train from scratch<br /> model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=num_epochs)<br /> model.save("rnn.h5")</pre>
<h3>6. Accuracy:</h3>
<div class="input">
<div class="inner_cell">
<div class="input_area" aria-label="Edit code here">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" draggable="false" tabindex="-1">
<div class="CodeMirror-sizer">
<div>
<div class="CodeMirror-lines" role="presentation">
<div role="presentation">
<div class="CodeMirror-cursors"></div>
<div class="CodeMirror-code" role="presentation">
<pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-builtin">print</span>(<span class="cm-variable">X_test</span>)</span></pre>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_text output_stream output_stdout">
<pre>[[   0    0    0 ...   14    6  717]
 [   0    0    0 ...  125    4 3077]
 [  33    6   58 ...    9   57  975]
 ...
 [   0    0    0 ...   21  846    2]
 [   0    0    0 ... 2302    7  470]
 [   0    0    0 ...   34 2005 2643]]</pre>
</div>
</div>
</div>
</div>
<div class="input">
<div class="inner_cell">
<div class="input_area" aria-label="Edit code here">
<div class="CodeMirror cm-s-ipython">
<div class="cell code_cell rendered selected" tabindex="2">
<div class="input">
<div class="inner_cell">
<div class="input_area" aria-label="Edit code here">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" draggable="false" tabindex="-1">
<div class="CodeMirror-sizer">
<div>
<div class="CodeMirror-lines" role="presentation">
<div role="presentation">
<div class="CodeMirror-code" role="presentation">
<pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-builtin">print</span>(<span class="cm-variable">model</span>.<span class="cm-property">predict</span>(<span class="cm-variable">X_test</span>))</span></pre>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_text output_stream output_stdout">
<pre>[[0.02767599]
 [0.99822277]
 [0.49427885]
 ...
 [0.02382611]
 [0.14700828]
 [0.75354964]]</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell code_cell rendered unselected" tabindex="2">
<div class="input">
<div class="inner_cell">
<div class="input_area" aria-label="Edit code here">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" tabindex="-1">
<div class="CodeMirror-sizer">
<div class="CodeMirror-lines" role="presentation">
<div role="presentation">
<div class="CodeMirror-cursors">
<div class="CodeMirror-cursor"></div>
</div>
<div class="CodeMirror-code" role="presentation">
<pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">scores</span> <span class="cm-operator">=</span> <span class="cm-variable">model</span>.<span class="cm-property">evaluate</span>(<span class="cm-variable">X_test</span>, <span class="cm-variable">y_test</span>, <span class="cm-variable">verbose</span><span class="cm-operator">=</span><span class="cm-number">0</span>)<br /><span class="cm-builtin">print</span>(<span class="cm-string">'Test accuracy:'</span>, <span class="cm-variable">scores</span><span class=" CodeMirror-matchingbracket">[</span><span class="cm-number">1</span><span class=" CodeMirror-matchingbracket">]</span>)</span></pre>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_text output_stream output_stdout">
<pre>Test accuracy: 0.8713200092315674</pre>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<p><strong></strong></p>
<p><strong>2. VCB stock prediction</strong></p>
<p>&nbsp;</p>
<div id="gtx-anchor" style="position: absolute; visibility: hidden; left: 8px; top: 1597.36px; width: 602.578px; height: 143px;">&nbsp;</div>
<div class="jfk-bubble gtx-bubble" style="visibility: visible; left: -14px; top: 1252px; opacity: 1;" role="alertdialog" aria-describedby="bubble-11">&nbsp;</div>
