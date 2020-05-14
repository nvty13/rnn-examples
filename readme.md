<h1><strong>Simple examples of RNNs</strong></h1>
<h2><strong>I. IMDB sentiment classification:</strong></h2>
<h3>1. Dataset:</h3>
<p>In this section, we apply RNNs (particularly LSTM) to IMDB dataset. The original dataset can be found <a href="https://www.imdb.com/interfaces/">here</a> on the home page.</p>
<p>Luckily, Keras has a built-in IMDB dataset, by inserting this line of code to your source:</p>
<pre><p>from keras.datasets import imdb</p></pre>
<p>&nbsp;</p>
<h3>2. Load the dataset</h3>
<pre>vocabulary_size = 5000<br /><br />(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocabulary_size)</pre>
<p>The dataset contains 2 columns: review column is the comments from reviewers and the sentiment column (0 for negative and 1 for positive)</p>
<p>Note that the review is stored as a sequence of integers. These are word IDs that have been pre-assigned to individual words, and the label is an integer.</p>
<p><img src="https://github.com/nvty13/rnn-examples/blob/master/images/IMDB.png?raw=true" alt="" width="500" style="display: block; margin-left: auto; margin-right: auto;" /></p>
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
<p><img src="https://github.com/nvty13/rnn-examples/blob/master/images/JPEG%20image-DE00FD10A46B-1.jpeg?raw=true" alt="" width="500" style="display: block; margin-left: auto; margin-right: auto;" /></p>
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
 [   0    0    0 ... 2302    7  470]:
 [   0    0    0 ...   34 2005 2643]]</pre>
</div>
</div>
</div>
</div>
<p>The outputs below are the corresponding predicted values of each of the above inputs:</p>
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
<pre>[[0.02767599]  ‣ "Negative"
 [0.99822277]  ‣ "Positive"     
 [0.49427885]  ‣ "Negative"
 ...
 [0.02382611]  ‣ "Negative"
 [0.14700828]  ‣ "Negative"
 [0.75354964]] ‣ "Positive"</pre>
</div>
</div>
</div>
</div>
</div>
 <pre>
 Input                               |  Predicted value |  Predicted Sentiment
 [   0    0    0 ...   14    6  717] |  [0.02767599]    |  0 (Neg)
 [   0    0    0 ...  125    4 3077] |  [0.99822277]    |  1 (Pos)
 [  33    6   58 ...    9   57  975] |  [0.49427885]    |  0 (Neg)
 ...
 [   0    0    0 ...   21  846    2] |  [0.02382611]    |  0 (Neg)
 [   0    0    0 ... 2302    7  470] |  [0.14700828]    |  0 (Neg)
 [   0    0    0 ...   34 2005 2643] |  [0.75354964]    |  1 (Pos)
 </pre>
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
<div class="input">
<div class="inner_cell">
<div class="input_area" aria-label="Edit code here">
<div class="CodeMirror cm-s-ipython">
<div class="CodeMirror-scroll" draggable="false" tabindex="-1">
<div class="CodeMirror-sizer">
<div>
<div class="CodeMirror-lines" role="presentation">
<div role="presentation"></div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
<h2><strong>II. VCB stock prediction</strong></h2>
Note: This tutorial is for studying purpose only. In fact, the stock price can be effected by many factors.
<h3>1. Dataset:</h3>
<p>Training data: <a href="https://github.com/nvty13/rnn-examples/raw/master/vcb_2009_2018.csv">vcb_2009_2018.csv</a> is the stock values from 2009 - 2018&nbsp;</p>
<p>Testing data: <a href="https://github.com/nvty13/rnn-examples/raw/master/vcb_2019.csv">vcb_2019.csv</a> is the stock values of the year 2019</p>
<div id="gtx-anchor" style="position: absolute; visibility: hidden; left: 8px; top: 1597.36px; width: 602.578px; height: 143px;">&nbsp;</div>
<h3 class="jfk-bubble gtx-bubble" style="visibility: visible; left: -14px; top: 1252px; opacity: 1;" role="alertdialog" aria-describedby="bubble-11">2. Loading the dataset</h3>
<pre>
dataset_train = pd.read_csv('vcb_2009_2018.csv')
</pre>
<div class="jfk-bubble gtx-bubble" style="visibility: visible; left: -14px; top: 1252px; opacity: 1;" role="alertdialog" aria-describedby="bubble-11">The dataset contains a lot of columns but we only focus on the "CLOSE" price value</div>
<img src="https://raw.githubusercontent.com/nvty13/rnn-examples/master/images/VCB%20dataset.png" width="100%">
<h3 class="jfk-bubble gtx-bubble" style="visibility: visible; left: -14px; top: 1252px; opacity: 1;" role="alertdialog" aria-describedby="bubble-11">3. Model structure</h3>
<pre>Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm_5 (LSTM)                (None, 60, 50)            10400     
_________________________________________________________________
dropout_5 (Dropout)          (None, 60, 50)            0         
_________________________________________________________________
lstm_6 (LSTM)                (None, 60, 50)            20200     
_________________________________________________________________
dropout_6 (Dropout)          (None, 60, 50)            0         
_________________________________________________________________
lstm_7 (LSTM)                (None, 60, 50)            20200     
_________________________________________________________________
dropout_7 (Dropout)          (None, 60, 50)            0         
_________________________________________________________________
lstm_8 (LSTM)                (None, 50)                20200     
_________________________________________________________________
dropout_8 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 51        
=================================================================
Total params: 71,051
Trainable params: 71,051
Non-trainable params: 0</pre>
<h3 class="jfk-bubble gtx-bubble" style="visibility: visible; left: -14px; top: 1252px; opacity: 1;" role="alertdialog" aria-describedby="bubble-11">&nbsp;4. Training and testing</h3>
<pre class="jfk-bubble gtx-bubble" style="visibility: visible; left: -14px; top: 1252px; opacity: 1;" role="alertdialog" aria-describedby="bubble-11"># If exist a saved model, then load pretrained weight<br />if path.exists("mymodel.h5"):<br /> regressor.load_weights("mymodel.h5")<br />else:<br /> # otherwise, train from beginning<br /> regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)<br /> regressor.save("mymodel.h5")</pre>
<p>6. Predicting:</p>
<pre>dataset_total = pd.concat((dataset_train['CLOSE'], dataset_test['CLOSE']), axis = 0)<br />inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values<br />inputs = inputs.reshape(-1,1)<br />inputs = sc.transform(inputs)<br /><br />X_test = []<br />no_of_sample = len(inputs)<br /><br />for i in range(60, no_of_sample):<br /> X_test.append(inputs[i-60:i, 0])<br /><br />X_test = np.array(X_test)<br />X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))<br />predicted_stock_price = regressor.predict(X_test)<br />predicted_stock_price = sc.inverse_transform(predicted_stock_price)</pre>
<p>7. Plotting the predicted value and the ground-truth values</p>
<pre># Plotting:<br />plt.plot(real_stock_price, color = 'red', label = 'Real VCB Stock Price')<br />plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted VCB Stock Price')<br />plt.title('VCB Stock Price Prediction')<br />plt.xlabel('Time')<br />plt.ylabel('VCB Stock Price')<br />plt.legend()<br />plt.show()</pre>
<p><img src="https://github.com/nvty13/rnn-examples/blob/master/VCB%20Stock%20prediction.png?raw=true" alt="" width="500" style="display: block; margin-left: auto; margin-right: auto;" /></p>
<p></p>
<h2>III. References:</h2>
<p>These examples were referenced from:</p>
<p class="fp b fq gb fs gc fu gd fw ge fy gf ct"><a href="https://towardsdatascience.com/a-beginners-guide-on-sentiment-analysis-with-rnn-9e100627c02e">A Beginner&rsquo;s Guide on Sentiment Analysis with RNN</a></p>
<p><a href="https://github.com/thangnch/MiAI_Stock_Predict"><span>Use LSTM to predict Vietcombank stock prices</span></a></p>
<p>Please email me if you have any problems: nvty@ctu.edu.vn</p>
