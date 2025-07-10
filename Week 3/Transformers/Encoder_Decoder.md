# Encoder-Decoder Seq2Seq Models

Reference: [Encoder-Decoder Seq2Seq Models](https://medium.com/analytics-vidhya/encoder-decoder-seq2seq-models-clearly-explained-c34186fbf49b)

## Introduction

Sequence Modelling problems refer to the problems where either the input and/or the output is a sequence of data (words, letters…etc.)

Consider a very simple problem of predicting whether a movie review is positive or negative. Here our input is a sequence of words and output is a single number between 0 and 1. If we used traditional DNNs, then we would typically have to encode our input text into a vector of fixed length using techniques like BOW, Word2Vec, etc. But note that here the sequence of words is not preserved and hence when we feed our input vector into the model, it has no idea about the order of words and thus it is missing a very important piece of information about the input.

Thus to solve this issue, RNNs came into the picture. In essence, for any input X = (x₀, x₁, x₂, … xₜ) with a variable number of features, at each time-step, an RNN cell takes an item/token xₜ as input and produces an output hₜ while passing some information onto the next time-step. These outputs can be used according to the problem at hand.

The movie review prediction problem is an example of a very basic sequence problem called many to one prediction. There are different types of sequence problems for which modified versions of this RNN architecture are used.

Sequence-to-Sequence (Seq2Seq) problems is a special class of Sequence Modelling Problems in which both, the input and the output is a sequence. Encoder-Decoder models were originally built to solve such Seq2Seq problems.

## Encoder-Decoder Architecture

At a very high level, an encoder-decoder model can be thought of as two blocks, the encoder and the decoder connected by a vector which we will refer to as the ‘context vector’.

1. ***Encoder***: The encoder processes each token in the input-sequence. It tries to cram all the information about the input-sequence into a vector of fixed length i.e. the ‘context vector’. After going through all the tokens, the encoder passes this vector onto the decoder.
2. ***Context vector***: The vector is built in such a way that it's expected to encapsulate the whole meaning of the input-sequence and help the decoder make accurate predictions. We will see later that this is the final internal states of our encoder block.
3. ***Decoder***: The decoder reads the context vector and tries to predict the target-sequence token by token.

The model can be thought of as two LSTM cells with some connection between them. The main thing here is how we deal with the inputs and the outputs.

### Encoder Block
The encoder part is an LSTM cell. It is fed in the input-sequence over time and it tries to encapsulate all its information and store it in its final internal states hₜ (hidden state) and cₜ (cell state). The internal states are then passed onto the decoder part, which it will use to try to produce the target-sequence. This is the ‘context vector’ which we were earlier referring to.

The outputs at each time-step of the encoder part are all discarded

### Decoder Block
So after reading the whole input-sequence, the encoder passes the internal states to the decoder and this is where the prediction of output-sequence begins. The decoder block is also an LSTM cell. The main thing to note here is that the initial states (h₀, c₀) of the decoder are set to the final states (hₜ, cₜ) of the encoder. These act as the ‘context’ vector and help the decoder produce the desired target-sequence.

Now the way decoder works, is, that its output at any time-step t is supposed to be the tᵗʰ word in the target-sequence/Y_true. The output at each time-step is fed as input to the next time-step.