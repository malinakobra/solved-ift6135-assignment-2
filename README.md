Download Link: https://assignmentchef.com/product/solved-ift6135-assignment-2
<br>
<strong>Summary:</strong>

In this assignment, you will perform <strong>sequential language modeling </strong>on the <a href="https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/">Wikitext-2</a> dataset. The dataset contains sequences of <em>tokens </em>(i.e., words or subwords) extracted from contiguous sentences. Each unique token is denoted by an integer in the range [1<em>,N</em>], where <em>N </em>is the number of all unique tokens (<em>N </em>is also often referred to as the <em>vocabulary size</em>). For this assignment, we use a vocabulary of size <em>N </em>= 40<em>,</em>478, as originally defined for OpenAI’s GPT. Sequential language models do <em>next-word prediction</em>: they predict tokens in a sequence one at a time, with each prediction based on all the previous elements of the sequence. A trained sequential language model can then be used to generate new sequences of text, by making each prediction conditioned on the past <em>predictions</em>.

In this assignment, you will implement a sequential language model (an <strong>LSTM</strong>), and a masked language model (<strong>Transformer </strong>inspired by OpenAI GPT). In problem 1, you will use built-in PyTorch modules to implement an LSTM. In problem 2, you will implement various building blocks of a transformer, including <strong>LayerNorm </strong>(layer normalization) and the <strong>Attention </strong>mechanism.

<strong>The Wikitext-2 dataset </strong>comprises 2 million words extracted from the set of verified “Good” and “Featured” articles on Wikipedia. See this <a href="https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/">blog post</a> for details about the Wikitext dataset and sample data. The dataset you get with the assignment has already been preprocessed using OpenAI’s GPT vocabulary, and each file is a compressed numpy array containing two arrays: tokens containing a flattened list of (integer) tokens, and sizes containing the size of each document.

You are provided a PyTorch dataset class (<a href="https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset">torch.utils.data.Dataset</a><a href="https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset">)</a> named Wikitext2 in the utils folder. This class loads the Wikitext-2 dataset and generates fixed-length sequences from it. Throughout this assignment, <strong>all sequences will have length 256</strong>, and we will use zero-padding to pad shorter sequences. Each sample from the dataset is a dictionary containing 3 keys: source (the input sequence), target (the target sequence, which is just the input sequence shifted one position to the left)), and mask (a binary vector of the same shape as the input indicating whether

– Do not distribute –

the token is valid (1), or is just a zero-padding to get the sequence length to 256). For example (the sequences have been shortened for presentation):

<table width="543">

 <tbody>

  <tr>

   <td width="65">source</td>

   <td width="47">304</td>

   <td width="47">4731</td>

   <td width="47">8406</td>

   <td width="39">614</td>

   <td width="24">0</td>

   <td width="24">0</td>

   <td width="24">0</td>

   <td width="86"><em>…</em></td>

   <td width="139">torch.LongTensor</td>

  </tr>

  <tr>

   <td width="65">target</td>

   <td width="47">4731</td>

   <td width="47">8406</td>

   <td width="47">614</td>

   <td width="39">304</td>

   <td width="24">0</td>

   <td width="24">0</td>

   <td width="24">0</td>

   <td width="86"><em>…</em></td>

   <td width="139">torch.LongTensor</td>

  </tr>

  <tr>

   <td width="65">mask</td>

   <td width="47">1</td>

   <td width="47">1</td>

   <td width="47">1</td>

   <td width="39">1</td>

   <td width="24">0</td>

   <td width="24">0</td>

   <td width="24">0</td>

   <td width="86"><em>…</em></td>

   <td width="139">torch.FloatTensor</td>

  </tr>

 </tbody>

</table>

In practice though, you will work with mini-batches of data, each with batchsize B elements. You can wrap this dataset object into a <a href="https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader">torch.utils.data.DataLoader</a><a href="https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader">,</a> which will return a dictionary with keys source, target, and mask, each of shape (B, 256).

<strong>Tests </strong>: For you to quickly verify that all the functions you implemented have valid input-output signatures (e.g. tensor types and shapes), we have provided public tests that will check if your output tensors have the correct shape, given random input data. Check the comments (docstrings) in each function to see what input/output shapes are expected. We recommend you to run these tests locally while completing the functions using the following command (to run inside your assignment folder)

python -m unittest

For students using Google Colab to complete their assignments, a cell with this command is available in the main.ipynb notebook. Note that these tests are not testing if the values returned by the functions are valid (only if the tensor shapes are correct), and will not be graded. You will be graded on a separate set of tests in Gradescope.

If the tests on Gradescope fail, as a rule of thumb <em>x </em>corresponds to the value in your assignment (e.g. the value returned by your function), and <em>y </em>is the expected value.

<strong>Emebeddings &amp; parameter sharing </strong>In both the LSTM and the Transformer, the embedding layer is among the layers that contain the most parameters. Indeed, it consists of a matrix of size (vocabulary size, embedding size) (note that here vocabulary size is equal to <em>N </em>+1, to account for zero-padding), which represents about 31M parameters. Similarly, in both models, we also have a classifier of about the same size (i.e. the output has size vocabulary size as well, to represent the probability of the next word). In order to speed up training, and for simplification in this assignment, we make two assumptions for both models:

<ul>

 <li>Following the architecture from OpenAI’s GPT, the weights of the classifier layer and the embedding layer will be shared.</li>

 <li>You will use pretrained embeddings from OpenAI’s GPT (they are provided in the run exp.py script, described in Problem 3), which will remain frozen over the course of training.</li>

</ul>

<strong>Coding instructions </strong>You will be required to use PyTorch to complete all questions. Moreover, this assignment <strong>requires running the models on GPU </strong>(otherwise it will take an incredibly long time); if you don’t have access to your own resources (e.g. your own machine, a cluster), please use Google Colab (the notebook main.ipynb is here to help you). For some questions, you will be asked to not use certain functions in PyTorch and implement these yourself using primitive functions from torch; in that case, the functions in question are explicitly disabled in the tests on Gradescope.

<h1>Problem 1</h1>

<strong>Implementing an LSTM (9pts) </strong>In this problem, you will be using PyTorch’s built-in modules in order to implement an LSTM. The architecture you will be asked to implement is the following:

to create this model. In particular, self.embedding is a <a href="https://pytorch.org/docs/1.7.1/generated/torch.nn.Embedding.html">nn.Embedding</a> module that converts sequences of token indices into embeddings, self.lstm is a <a href="https://pytorch.org/docs/1.7.1/generated/torch.nn.LSTM.html">nn.LSTM</a> module that runs an LSTM over a sequence of vectors, and self.classifier is a 2-layer MLP responsible for classification.

<ol>

 <li>Using the different modules described above, complete the forward() This function must return the log-probabilities (not the logits) of the next words in the sequence, as well as the final hidden state of the LSTM.</li>

 <li>Complete the loss() function, that returns the mean negative log-likelihood of the entire sequences in the minibatch (and also averaged over the mini-batch dimension). More precisely, for a single sequence in the mini-batch</li>

</ol>

<em>,</em>

where <em>w </em>are the predictions made by the model, and <strong>1</strong>(<em>i </em>= <em>w<sub>t</sub></em><sub>+1</sub>) is the indicator function e

which equals 1 if <em>i </em>= <em>w<sub>t</sub></em><sub>+1</sub>, and 0 otherwise. Note that here <em>T </em>might be smaller than 256 (called sequence length in the code), because the sequence might be zero-padded; you may use mask for this. The loss function directly takes the log-probabilities as input (e.g. returned by the forward function).

<h1>Problem 2</h1>

<strong>Implementing a GPT (Generative-Pretrained Transformer) (27pts) </strong>While typical RNNs “remember” past information by taking their previous hidden state as input at each step, recent years have seen a profusion of methodologies for making use of past information in different ways. The transformer<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> is one such fairly new architecture which uses several self-attention networks (“heads”) in parallel, among other architectural specifics. Implementing a transformer is a fairly involved process – so we provide most of the boilerplate code and your task is only to implement the multi-head scaled dot-product attention mechanism, as well as the layernorm operation.

<strong>Implementing Layer Normalization (5pts)</strong>: You will first implement the layer normalization (LayerNorm) technique that we have seen in class. For this assignment, <strong>you are not allowed </strong>to use the PyTorch <a href="https://pytorch.org/docs/1.7.1/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm">nn.LayerNorm</a> module (nor any function calling torch.layer norm).

As defined in the <a href="https://arxiv.org/abs/1607.06450">layer normalization paper</a><a href="https://arxiv.org/abs/1607.06450">,</a> the layernorm operation over a minibatch of inputs <em>x </em>is defined as

layernorm(weight + bias

where E[<em>x</em>] denotes the expectation over <em>x</em>, Var[<em>x</em>] denotes the variance of <em>x</em>, both of which are only taken over the last dimension of the tensor <em>x </em>here. weight and bias are learnable affine parameters.

<ol>

 <li>In the file gpt1 solution template.py, implement the forward() function of the LayerNorm class. Pay extra attention to the lecture slides on the exact details of how E[<em>x</em>] and Var[<em>x</em>] are computed. In particular, PyTorch’s function <a href="https://pytorch.org/docs/1.7.1/generated/torch.var.html">torch.var</a> uses an unbiased estimate of the variance by default, defined as the formula on the left-hand side</li>

</ol>

Var(<em>X</em>)unbiased    Var(<em>X</em>)biased

whereas LayerNorm uses the biased estimate on the right-hand size (where <em>X </em>here is the mean estimate). Please refer to the docstrings of this function for more information on input/output signatures.

<strong>Implementing the attention mechanism (17pts)</strong>: You will now implement the core module of the transformer architecture – the multi-head attention mechanism. Assuming there are <em>m </em>attention heads, the attention vector for the head at index <em>i </em>is given by:

[<em>q</em><sub>1</sub><em>,…,</em><em>q<sub>m</sub></em>] = <em>QW<sub>Q </sub></em>+ <em>b<sub>Q                                 </sub></em>[<em>k</em><sub>1</sub><em>,…,</em><em>k<sub>m</sub></em>] = <em>KW<sub>K </sub></em>+ <em>b<sub>K         </sub></em>[<em>v</em><sub>1</sub><em>,…,</em><em>v<sub>m</sub></em>] = <em>VW<sub>V </sub></em>+ <em>b<sub>V</sub></em>

<em>A<sub>i </sub></em>= softmax

<em>h</em><em>i </em>= <em>A</em><em>i</em><em>v</em><em>i</em>

<em>A</em>(<em>Q,</em><em>K,</em><em>V </em>) = concat(<em>h</em><sub>1</sub><em>,…,</em><em>h<sub>m</sub></em>)<em>W<sub>O </sub></em>+ <em>b<sub>O</sub></em>

Here <em>Q,</em><em>K,</em><em>V </em>are queries, keys, and values respectively, where all the heads have been concatenated into a single vector (e.g. here <em>K </em>∈ R<em><sup>T</sup></em><sup>×<em>md</em></sup>, where <em>d </em>is the dimension of a single key vector, and <em>T </em>the length of the sequence). <em>W<sub>Q</sub>,</em><em>W<sub>K</sub>,</em><em>W<sub>V </sub></em>are the corresponding projection matrices (with biases <em>b</em>), and <em>W<sub>O </sub></em>is the output projection (with bias <em>b<sub>O</sub></em>). <em>Q,</em><em>K, </em>and <em>V </em>are determined by the output of the previous layer in the main network. <em>A<sub>i </sub></em>are the attention values, which specify which elements of the input sequence each attention head attends to. We recommend you to check <a href="https://jalammar.github.io/illustrated-gpt2/">this tutorial</a> for implementation details of GPT. In this question, <strong>you are not allowed </strong>to use the module <a href="https://pytorch.org/docs/1.7.1/generated/torch.nn.MultiheadAttention.html">nn.MultiheadAttention</a> (or any function calling torch.nn.functional.multi head attention forward). Please refer to the docstrings of each function for a precise description of what each function is expected to do, and the expected input/output tensors and their shapes.

<ol start="2">

 <li>The equations above require many vector manipulations in order to split and combine head vectors together. For example, the concatenated queries <em>Q </em>are split into <em>m </em>vectors [<em>q</em><sub>1</sub><em>,…,</em><em>q<sub>m</sub></em>] (one for each head) after an affine projection by <em>W<sub>Q</sub></em>, and the <em>h<sub>i</sub></em>’s are then concatenated back for the affine projection with <em>W<sub>O</sub></em>. In the class MultiHeadedAttention, implement the utility functions split heads() and merge heads() to do both of these operations, as well as a transposition for convenience later. For example, for the 1st sequence in the mini-batch:</li>

</ol>

y = splitheads(x) → y[0, 1, 2, 3] = x[0, 2, num heads * 1 + 3] x = mergeheads(y) → x[0, 1, num heads * 2 + 3] = y[0, 2, 1, 3]

These two functions are exactly inverse from one another. Note that in the code, the number of heads <em>m </em>is called self.num heads, and the head dimension <em>d </em>is self.head size. Your functions must handle mini-batches of sequences of vectors, see the docstring for details about the input/output signatures.

<ol start="3">

 <li>In the class MultiHeadedAttention, implement the function get attention weights(), which is responsible for returning <em>A<sub>i</sub></em>’s (for all the heads at the same time) from <em>q<sub>i</sub></em>’s and <em>k<sub>i</sub></em>’s. Remember that the language model here is <em>auto-regressive </em>(also sometimes called <em>causal</em>), meaning that the attention must be computed only on past inputs, and not the future.</li>

</ol>

Concretely, this means that instead of taking the softmax over the whole sequence, we need to introduce a binary mask <em>s<sub>t </sub></em>(which is different from the mask key in the dataloader), where <em>s<sub>t</sub></em>(<em>τ</em>) is equal to 1 if the current element can attend position <em>τ </em>in the sequence (i.e. if <em>τ </em>≤ <em>t</em>), and 0 otherwise. The softmax is then modified as

[softmax(<em>.</em>

In practice, in order to avoid potential numerical stability issues, we recommend to use a different implementation:

[softmax(where                                                               <em>x<sub>τ </sub></em>= <em>x<sub>τ</sub>s<sub>t</sub></em>(<em>τ</em>) − 10<sup>4</sup>(1 − <em>s<sub>t</sub></em>(<em>τ</em>)) e

The second version is almost equivalent to the first (up to numerical precision), as long as

, which is the case in practice. You are strongly recommended to use vectorized operations as much as possible in order to speed-up training in Problem 3.

<ol start="4">

 <li>Using the functions you have implemented, complete the function apply attention() in the class MultiHeadedAttention, which computes the vectors <em>h<sub>i</sub></em>’s as a function of <em>q<sub>i</sub></em>’s, <em>k<sub>i</sub></em>’s and <em>v<sub>i</sub></em>’s, and concatenates the head vectors.</li>

</ol>

apply attention( concat(<em>h</em><sub>1</sub><em>,…,</em><em>h<sub>m</sub></em>).

<ol start="5">

 <li>Using the functions you have implemented, complete the function forward() in the class MultiHeadedAttention. You may implement the different affine projections however you want (do not forget the biases), and you can add modules to the init () How many learnable parameters does your module have, as a function of num heads and head size?</li>

</ol>

<strong>The GPT forward pass (5pts)</strong>: You now have all building blocks to implement the forward pass of a miniature GPT model. You are provided a module Block which corresponds to a full block with self-attention and a feed-forward neural network, with skip-connections, using the modules LayerNorm and MultiHeadedAttention you implemented before. The architecture of the GPT model is the following:

log<em>p</em>(<em>w</em><sub>2 </sub>| <em>w</em><sub>1:1</sub>) log<em>p</em>(<em>w</em><sub>3 </sub>| <em>w</em><sub>1:2</sub>) log<em>p</em>(<em>w</em><sub>4 </sub>| <em>w</em><sub>1:3</sub>) log<em>p</em>(<em>w</em><sub>5 </sub>| <em>w</em><sub>1:4</sub>) log<em>p</em>(<em>w</em><sub>6 </sub>| <em>w</em><sub>1:5</sub>)

In this part of the exercise, you will fill in the MiniGPT1 class in gpt1 solution template.py. This module contains all the blocks necessary to create this model. In particular, self.embedding is a module responsible for converting sequences of token indices into embeddings (using input and positional embeddings), self.layers is a <a href="https://pytorch.org/docs/1.7.1/generated/torch.nn.ModuleList.html">nn.ModuleList</a> containing the different Block layers, and self.classifier is a linear layer responsible for classification.

<ol start="6">

 <li>In the class MiniGPT1, implement the function get embeddings() which computes the embedding vectors, based on the input sequences and the positions of the tokens. See the docstrings of GPT1Embedding (in utils/embeddings.py) for details about this module.</li>

 <li>In the class MiniGPT1, complete the function forward() using the different modules described above. This function must return the log-probabilities (not the logits) of the next words in the sequence.</li>

 <li>Complete the loss() function, that returns the mean negative log-likelihood of the entire sequences in the mini-batch (and also averaged over the mini-batch dimension). See the definition of the loss in Problem 1.</li>

</ol>

<h1>Problem 3</h1>

<strong>Training language models and model comparison (25pts) </strong>Unlike in classification problems, where the performance metric is typically accuracy, in language modelling, the performance metric is typically based directly on the cross-entropy loss, i.e. the negative log-likelihood (<em>NLL</em>) the model assigns to the tokens. For word-level language modelling it is standard to report <strong>perplexity (PPL)</strong>, which is the exponentiated average per-token NLL (over all tokens):

<em> ,</em>

where <em>t </em>is the index with the sequence, and <em>j </em>indexes different sequences. The purpose of this question is to perform model exploration, which is done using a validation set. As such, we do not require you to run your models on the test set.

You will train each of the following architectures using an optimization technique and scheduler of your choice. For reference, we have provided a <em>feature-complete </em>training script (run exp.py) that uses the <a href="https://pytorch.org/docs/stable/optim.html#torch.optim.AdamW">AdamW</a> optimizer. You are free to modify this script as you deem fit. You do not need to submit code for this part of the assignment. However, you are required to create a report that presents the perplexities and training curve comparisions as specified in the following questions.

<strong>Note</strong>: For each experiment, closely observe the training curves, and report the lowest validation perplexity score across epochs (not necessarily the validation score for the last epoch)

<strong>Configurations to run</strong>: At the top of the runner file (run exp.py), we have provided 12 experiment configurations for you to run. Together, these configurations span several choices of neural network architecture, optimizer, and weight-decay parameters. Perform the following analysis on the logs.

<ol>

 <li>You are asked to run 12 experiments with different architectures, optimizers, and hyperparameters settings. These parameter settings are given to you at the top of the runner file (run exp.py). For each of these 12 experiments, plot learning curves (train and validation) of perplexity over both <strong>epochs </strong>and <strong>wall-clock-time</strong>. Figures should have labeled axes and a legend and an explanatory caption.</li>

 <li>Make a table of results summarizing the train and validation performance for each experiment, indicating the architecture and optimizer. Sort by architecture, then number of layers, then optimizer, and use the same experiment numbers as in the runner script for easy reference. Bold the best result for each architecture.<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a> The table should have an explanatory caption, and appropriate column and/or row headers. Any shorthand or symbols in the table should be explained in the caption.</li>

 <li>Which hyperparameters + optimizer would you use if you were most concerned with wall-clock time? With generalization performance?</li>

 <li>Between the experiment configurations 1-4 and 5-8 at the top of run exp.py, only the optimizer changed. What difference did you notice about the four optimizers used? What was the impact of weight decay, momentum, and Adam?</li>

 <li>Compare experiments 1 and 5. Which model did you think performed better (LSTM or GPT)? Why?</li>

 <li>In configurations 5-8 and in configurations 11 and 12, you trained a transformer with various hyper-parameter settings. Given the recent high profile transformer based language models, are the results as you expected? Speculate as to why or why not.</li>

 <li>For each of the experiment configurations above, measure the average steady-state GPU memory usage (nvidia-smi is your friend!). Comment about the GPU memory footprints of each model, discussing reasons behind increased or decreased memory consumption where applicable.</li>

 <li>Comment on the overfitting behavior of the various models you trained, under different hyperparameter settings. Did a particular class of models overfit more easily than the others? Can you make an informed guess of the various steps a practitioner can take to prevent overfitting in this case? (You might want to refer to sets of experiments 2, 9, 10 for the LSTM and 6, 11, 12 for GPT – that evaluate models of increasing capacities).</li>

</ol>

<a href="#_ftnref1" name="_ftn1">[1]</a> See <a href="https://arxiv.org/abs/1706.03762">https://arxiv.org/abs/1706.03762</a> for more details.

<a href="#_ftnref2" name="_ftn2">[2]</a> You can also make the table in LaTeX; for convenience you can use tools like <a href="https://www.tablesgenerator.com/">LaTeX table generator</a> to generate tables online and get the corresponding LaTeX code.