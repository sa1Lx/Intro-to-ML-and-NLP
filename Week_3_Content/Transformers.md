# Transformers

Transformers are a type of deep learning model introduced in the 2017 paper "Attention is All You Need" by Vaswani et al . They revolutionized natural language processing (NLP) and are now used in many domains like vision, speech, and more.

# Recurrent Neural Networks (RNNs) 

Today, different Machine Learning techniques are used to handle different types of data. One of the most difficult types of data to handle and the forecast is sequential data. Sequential data is different from other types of data in the sense that while all the features of a typical dataset can be assumed to be order-independent, this cannot be assumed for a sequential dataset. To handle such type of data, the concept of Recurrent Neural Networks was conceived. It is different from other Artificial Neural Networks in its structure. While other networks "travel" in a linear direction during the feed-forward process or the back-propagation process, the Recurrent Network follows a recurrence relation instead of a feed-forward pass and uses Back-Propagation through time to learn. 
Reference: [RNN by GFG_1](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/) & [RNN by GFG_2](https://www.geeksforgeeks.org/machine-learning/recurrent-neural-networks-explanation/)

In order to understand the Transformer Architecture, It is Important to understand the RNN Stucture:

## How is RNN different from a regular Neural Network?

Recurrent Neural Networks (RNNs) differ from regular neural networks in how they process information. While standard neural networks pass information in one direction i.e from input to output, RNNs feed information back into the network at each step.

Imagine reading a sentence and you try to predict the next word, you don’t rely only on the current word but also remember the words that came before. RNNs work similarly by “remembering” past information and passing the output from one step as input to the next i.e it considers all the earlier words to choose the most likely next word. This memory of previous steps helps the network understand context and make better predictions.

## Key Components of RNNs

1. ***Recurrent Neurons***: The fundamental processing unit in RNN is a Recurrent Unit. They hold a hidden state that maintains information about previous inputs in a sequence. Recurrent units can "remember" information from prior steps by feeding back their hidden state, allowing them to capture dependencies across time.

2. ***RNN Unfolding***: RNN unfolding or unrolling is the process of expanding the recurrent structure over time steps. During unfolding each step of the sequence is represented as a separate layer in a series illustrating how information flows across each time step.

## Architecture of RNNs

RNNs share similarities in input and output structures with other deep learning architectures but differ significantly in how information flows from input to output. Unlike traditional deep neural networks where each dense layer has distinct weight matrices. RNNs use shared weights across time steps, allowing them to remember information over sequences.

The Recurrent Neural Network consists of multiple fixed activation function units, one for each time step. Each unit has an internal state which is called the hidden state of the unit. This hidden state signifies the past knowledge that the network currently holds at a given time step. This hidden state is updated at every time step to signify the change in the knowledge of the network about the past. The hidden state is updated using the following recurrence relation:- <br>
$h_t = f_W(h_{t-1}, x_t)$

At each time step, the new hidden state is calculated using the recurrence relation as given above. This new generated hidden state is used to generate indeed a new hidden state and so on. 

![image1](images/image1.png)

Note that $h_0$ is the initial hidden state of the network. Typically, it is a vector of zeros, but it can have other values also. One method is to encode the presumptions about the data into the initial hidden state of the network. For example, for a problem to determine the tone of a speech given by a renowned person, the person's past speeches' tones may be encoded into the initial hidden state. Another technique is to make the initial hidden state a trainable parameter. Although these techniques add little nuances to the network, initializing the hidden state vector to zeros is typically an effective choice. 

## Some usecases of RNNs

Video Reference: 

1. Auto-Completion: RNNs can predict the next word in a sentence based on the previous words, making them useful for text input suggestions.
2. Sentiment Analysis: RNNs can analyze the sentiment of a sequence of words, such as determining if a sentence is positive or negative.
3. Language Translation: RNNs can translate sentences from one language to another by understanding the context and structure of the input sentence.
4. Name Entity Recognition: RNNs can identify and classify named entities in a text, such as names of people, organizations, or locations.