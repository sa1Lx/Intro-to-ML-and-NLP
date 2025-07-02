# Transformers

Reference: [here](https://jalammar.github.io/illustrated-transformer/)

A transformer is a type of neural network architecture that has revolutionized natural language processing (NLP) and other sequence-based tasks. It was introduced in the paper "Attention is All You Need" by Vaswani et al. in 2017.<br>
It contains an encoding component, a decoding component, and connections between them.<br>
The encoding component is a stack of encoders.  The decoding component is a stack of decoders of the same number.<br>

The encoders are all identical in structure (yet they do not share weights). Each one is broken down into two sub-layers:
1. Self Attention: The encoder’s inputs first flow through a self-attention layer – a layer that helps the encoder look at other words in the input sentence as it encodes a specific word.
2. Feed Forward Neural Network: The outputs of the self-attention layer are fed to a feed-forward neural network. The exact same feed-forward network is independently applied to each position.

The decoder has both those layers, but between them is an attention layer that helps the decoder focus on relevant parts of the input sentence (similar what attention does in seq2seq models).

The embedding only happens in the bottom-most encoder. The abstraction that is common to all the encoders is that they receive a list of vectors each of the size 512 – In the bottom encoder that would be the word embeddings, but in other encoders, it would be the output of the encoder that’s directly below. The size of this list is hyperparameter we can set – basically it would be the length of the longest sentence in our training dataset.

After embedding the words in our input sequence, each of them flows through each of the two layers of the encoder.

The word in each position flows through its own path in the encoder. There are dependencies between these paths in the self-attention layer. The feed-forward layer does not have those dependencies, however, and thus the various paths can be executed in parallel while flowing through the feed-forward layer. 

