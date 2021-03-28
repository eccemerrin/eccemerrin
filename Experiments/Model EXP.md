# 1-Building Blocks


### 1.1 Encoder-Decoder Architecture


Encoder-decoder architecture is a well-know and frequently used neural network structure, which is proven to work highly efficient with sequential data. Here, an encoder network takes the input sequence and maps it into a vector called context vector. This context vector contains all the relevant information about the input sequence. A decoder network then uses this context vector to output the final product. In theory, encoder and decoder networks can contain any sort of neural networks but they traditionally contain RNN’s [[1](https://arxiv.org/abs/1406.1078)].


#### 1.1.1 Positional Encoding

Using a nonrecurrent neural network as the encoder network might be beneficial because of several reasons. Firstly, being nonrecurrent allows parallelization, which in turn reduces training time tremendously. Secondly, using a nonrecurrent neural network gets rid of the so-called “vanishing gradient problem”. That is, in some cases where one is training a model with gradient descent method and backpropagation, the gradient will be vanishingly small so that it prevents the weight from changing its value. However, using a nonrecurrent neural network as the encoder network comes with its own challenges. One of them is incorporating positional information since they have no inherited way of modelling position in a sequence.

One way of incorporating positional information is using an additive trigonometric positional encoding as proposed by Vaswani et al. [[2](https://arxiv.org/abs/1706.03762)]. This positional encoding method exploits the basic properties of trigonometric functions to determine the relative positions of two vectors.

Another way of incorporating positional information is to use LSTM layers prior to the layer which is nonrecurrent as proposed by Sperber et al. [[3](https://arxiv.org/abs/1803.09519)]. This method creates an alternative to additive trigonometric positional encoding and is used where additive trigonometric positional encoding can be problematic. Here, Long Short Term Memory (LSTM) [[5](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735)] is a specific type of RNN, which performs better with long-sequences and long distance dependencies. 

### 1.2 Attention Mechanism


Most abstractly, attention mechanism is just a function which maps a query vector and a set of key-value vector pairs to a single output vector. Here, a score for each key is calculated using the query by a compatibility function. The compatibility function is what makes the difference between different attention functions. Then, the softmax function maps the set of scores to the set of alignment scores. Lastly, each value vector is scaled by their corresponding alignment score and the summation of all those resulting vectors gives the output of the attention function for that query. 


![alignment scores](https://www.dropbox.com/s/wa0nne2g44ykzrl/attention.PNG?dl=0&raw=1)


Where q is the query vector, k<sub>i</sub> is the ith key vector, v<sub>i</sub> is the ith value vector and $\alpha$<sub>i</sub> is the alignment score for ith key vector.

Although there are many different attention functions, two most commonly used are additive and dot-product attention. 

#### 1.2.1 Additive Attention

The compatibility function of additive attention is a feed-forward neural network with one hidden layer [[4]( https://arxiv.org/abs/1409.0473)]. It learns how to assign scores to a given query and a key vector throughout the training process, which sometimes makes the model learn slower.

![additive attention](https://www.dropbox.com/s/0vvyhmwal370kz1/addditive.PNG?dl=0&raw=1)

#### 1.2.2 Dot-Product Attention

Dot-product attention calculates scores by simply taking the dot product of the query and the key vectors. One slightly modified version of dot-product attention is scaled dot-product attention [[2](https://arxiv.org/abs/1706.03762)], which basically scales the result of dot-product before softmax function acts on it. The motivation behind this is when the input is large, the softmax function may have a vanishing gradient causing some sort of vanishing gradient problem.

![dot-product and scaled dot product attention](https://www.dropbox.com/s/idwuksd4652lorw/dot.PNG?dl=0&raw=1)

#### 1.2.3 Multi-Head Attention


One way to increase the efficiency of attention layers is to use multi-head attention [[2](https://arxiv.org/abs/1706.03762)]. Here, instead of applying a single attention function to the given query, key and value vectors, the same attention function is applied to different linear projections of the given query, key and value vectors. Those linear projections are performed by learned linear layers. Then, the resulting sets of output vectors are concatenated and another linear layer is applied to concatenated output vectors. The motivation for this is quite similar to motivation for using convolutional layers with more than one convolution. Using multi-head attention, each head can be specialized in a certain task. For instance, in a translation task, one head can be specialized in detecting noun-verb agreement while another has no idea about noun and verbs.

![multi-head attention](https://www.dropbox.com/s/oxlxpl1g7w84yev/multi.PNG?dl=0&raw=1)


#### 1.2.4 Masking

Masking is a useful tool when dealing with a model that should not use some part of the data when outputting a certain output. The most common example of this is a prediction model working with a time series. When training, the model should only use the past part of the data in order to predict the next output. Therefore, one has to mask the future part of the data. One way to accomplish that is setting the scores of future keys to negative infinity considering how the softmax function works.

### 1.3 Residual Connections 

In some deep neural networks, deepening the model might cause model's accuracy to drop on training data. This is called the degradation problem. Residual connections are proven to prevent this phenomenon [[6](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)]. Residual connections are indeed a simple mechanism. The output of a layer is _x + LayerFunction(x)_, where _x_ is the input to the layer and _LayerFunction(x)_ is the function implemented by the layer itself. 

### 1.4 Transformer Architecture

The transformer model is a type of neural network for sequential data without any recurrent relationship. The first transformer model [[2](https://arxiv.org/abs/1706.03762)] emerged for translation task. Here, an encoder network used attention mechanism for comparing each word with each other and encode the meaning of the sentence to a context matrix. A decoder network then used this context matrix and attention mechanism to produce the output sentence. 

![Transforemer Architecture](https://www.dropbox.com/s/246uzy3jmv9q70v/transformer.PNG?dl=0&raw=1)

### 1.5 Network in Network

Network in network (NIN) [[8](https://arxiv.org/abs/1312.4400)] is a deep network structure that enhances model discriminability. Network in network structure is quite similar to that of convolutional layers. Here, instead of using linear filters followed by a nonlinear activation function like convolutional layers, we instead build a micro neural network with more complex structures. NIN is used to deepen the model without increasing the number of learnable parameters. 

![NIN Structure](https://www.dropbox.com/s/bh3j45btx26s5uh/NIN.PNG?dl=0&raw=1)

### 1.6 Earthquake Transformer Architecture 

Earthquake Transformer (EQT) [[7](https://www.nature.com/articles/s41467-020-17591-w.epdf?sharing_token=IiqAaF4NxwhUWGQLLLyTw9RgN0jAjWel9jnR3ZoTv0Nn-FaUKb3nu4lFkVXeZX_BCz5eMr5DkfCxQ3XASbeWwldzdU9oZF3d2MMG4cz6GWhVklzzzlL0QeMcf9kJJxA8wJAFfFCmtdlpQklDmGG7qRVjJxlCK-nusJjMFWE2oEk%3D)] is the state of art model for earthquake detecting. Here, a very deep encoder is used to encode the waveform. Then, 3 different decoders are used for detecting the earthquake, picking the P-phase and picking the S-phase. 

In order to manage computational complexity, convolutional layers with _MaxPooling/2_ are used. Following that, there are CNN layers with residual connections since the model is indeed very deep and, therefore, residual connections [[6](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)] are necessary. Bi-directional LSTM layers with NIN and a single LSTM layer are used to increase the depth of the network and incorporate positional information. 

At the end of the encoder 2 transformers with global attention is applied. It is claimed that with these two transformer blocks model learns to identify the earthquake signal from time series. Then, local attention blocks at the beginning of the phase-picker decoders sharpen the focus to individual seismic phases.
 
![eqt_model.png](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41467-020-17591-w/MediaObjects/41467_2020_17591_Fig1_HTML.png?as=webp)


# 2-Modeling Experiments #

Modeling experiements were done so as to build a better EQTransformer. 

In order to do that we tried 14 different models in total. 

In each experiments we remove existing parts or add parts to existing code.

* Models trained on _STEADmini_ :
  * Salvation
  * Genesis
  * Vanilla
  * GRU
  * 4x4
  * 2LSTM
  * EQT (for comparison)

* Models trained on _merged_ :
  * BCLOS
  * BC
  * BO
  * Partial Vanilla
  * EQT (for comparison)
  
* Models trained on _STEAD-micro_ :
  * nb_filters_changed_EQTUtils
  * kernel_size_changed
  * nb_filters_changed_trainer
  * EQT (for comparison)
  

### 2.1 Models Trained On _Merged_ : ###

#### 2.1.1 BCLOS ####
BCLOS is an acronym for *BiLSTM Closed LSTM Open*.

BiLSTM's have a structure which allow them to reach information from backward and forward. However, LSTM's have a structure which allow them to reach information just from backward. 

So, I wonder what happen if I remove a bidirectional LSTM block from model and add an unidirectional LSTM block in the place of it. So, I removed BiLSTM part from existing code and add a LSTM block in the place of BiLSTM. 


Before seeing results, I expected that results of Unidirectional layers would be much worse then Bidirectional layers. Suprisingly, *BCLOS* was better in 19 parameters out of 25. Those are;
 * **det_recall**, **det_precision**, **d_fp**, **d_tn**, **d,_fn**
 * **s_recall**, **s_precison**, **s_mae**, **s_rmse**, **s_fp**, **s_fn**
 * **p_recall**, **p_precision**,**p_mae**, **p_rmse**, **p_tp**, **p_fp**, **p_tn**, **p_fn**


#### 2.1.2 BC ####
BC is an acronym for *Both Closed*. 

In this architecture, I removed both BiLSTM layers and added LSTM layers  from the common layers mentioned in [Google doc](https://docs.google.com/document/d/1JxLc_Bp0wNTSlZUpWlf_87riHRQKDenNGf665mFMVpQ/edit#heading=h.moi7c1x12w31).

As expected, result of this architecture was worse than **BCLOS**. *BC* was better than EQT in 9 parameters out of 25. Those are;
 * **det_recall**,  **d_fn**
 * **s_rmse**, **s_tn**
 * **p_precision**, **p_mae**, **p_rmse**, **p_fp**, **p_tn**

#### 2.1.3 Partial Vanilla ####

In *Partial Vanilla* architecture, I removed BiLSTM layers and CNN Layers from the common layers mentioned in [Google doc](https://docs.google.com/document/d/1JxLc_Bp0wNTSlZUpWlf_87riHRQKDenNGf665mFMVpQ/edit#heading=h.moi7c1x12w31).

*Partial Vanilla* was better than EQT in 9 parametersout of 25. Those are;
 * **det_recall**,  **d_fn**
 * **s_mae**, **s_rmse**, **s_tn**
 * **p_recall**, **p_mae**, **p_rmse**, **p_fn**

#### 2.1.4 BO ####
BO is an acronym for *Both Open*.

In this architecture; I add LSTM layers, as successive layers of CNN layers, to the common layers mentioned in [Google doc](https://docs.google.com/document/d/1JxLc_Bp0wNTSlZUpWlf_87riHRQKDenNGf665mFMVpQ/edit#heading=h.moi7c1x12w31). 

*BO* was better than EQT in 6 parameters out of 25. Those are;
 * **det_recall**
 * **s_mae**, **s_rmse**, **s_tn**
 * **p_mae**, **p_rmse**

### 2.2 Models Trained On _STEAD-micro_ ###

#### 2.2.1 nb_filters_changed_EQTUtils ####

In *nb_filters_changed model* we changed the filter size in EqT_utils.py (line=2734). The filter sizes we changed in Eqt_utils.py are related to last CNN block's filter sizes in the architecture mentioned in [Google doc](https://docs.google.com/document/d/1JxLc_Bp0wNTSlZUpWlf_87riHRQKDenNGf665mFMVpQ/edit#heading=h.moi7c1x12w31). The orginal fiter sizes were nb_filters=[8, 16, 16, 32, 32, 96, 96, 128] then, changed to nb_filters=[8, 16, 16, 32, 32, 64, 64, 128].

*nb_filters_changed_EQTUtils* was better than EQT in 13 parameters out of 24. Those are;
* **det_recall**, **d_tp**, **d_fp**
* **p_recall**, **p_mae**, **p_tp**, **p_fp**
* **s_recall**, **s_mae**, **s_rmse**, **s_tp**, **s_fp**
* **#events**

#### 2.2.2 kernel_size_changed ####

In *kernel_size_changed model* we changed the kernel sizes in trainer.py (line=417). The kernel sizes we changed in trainer.py are realted to the first CNN block's kernel sizes in the architecture mentioned in [Google doc](https://docs.google.com/document/d/1JxLc_Bp0wNTSlZUpWlf_87riHRQKDenNGf665mFMVpQ/edit#heading=h.moi7c1x12w31). The original kernel sizes were kernel_size=[11, 9, 7, 7, 5, 5, 3] then, changed to kernel_size=[10, 9, 8, 7, 6, 5, 4].

*kernel_size_chaned* model was better than EQT in 11 parameters out of 24. Those are;
* **det_recall**, **det_precision**, **d_tp**
* **p_recall**, **p_precision**, **p_mae**, **p_tp**
* **s_precision**, **s_tp**, **s_fn**
* **#events**

#### 2.2.3 nb_filters_changed_trainer ####

In *nb_filters_changed_trainer* model we changed the filter sizes in trainer.py (line=416). The filter sizes we changed in trainer.py are related to the first CNN block's filter sizes in the architecture mentioned in [Google doc](https://docs.google.com/document/d/1JxLc_Bp0wNTSlZUpWlf_87riHRQKDenNGf665mFMVpQ/edit#heading=h.moi7c1x12w31). The original filter sizes were nb_filters=[8, 16, 16, 32, 32, 64, 64] then, changed to nb_filters=[8, 8, 32, 32, 128, 128, 128].

*nb_filters_changed_trainer* model was better than EQT in 12 parameters out of 24. Those are;
* **det_recall**, **det_precision**, **d_tp**
* **p_recall**, **p_precision**, **p_tp**
* **s_recall**, **s_precision**, **s_mae**, **s_rmse**, **s_tp**
* **#events**


## References

[1] Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. _arXiv preprint arXiv:1406.1078_.

[2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. _arXiv preprint arXiv:1706.03762_.

[3] Sperber, M., Niehues, J., Neubig, G., Stüker, S., & Waibel, A. (2018). Self-attentional acoustic models. _arXiv preprint arXiv:1803.09519_.

[4] Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. _arXiv preprint arXiv:1409.0473_.

[5] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. _Neural computation_, _9_(8), 1735-1780.

[6] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ (pp. 770-778).

[7] Mousavi, S. M., Ellsworth, W. L., Zhu, W., Chuang, L. Y., & Beroza, G. C. (2020). Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. _Nature communications_, _11_(1), 1-12.

[8] Lin, M., Chen, Q., & Yan, S. (2013). Network in network. _arXiv preprint arXiv:1312.4400_.
