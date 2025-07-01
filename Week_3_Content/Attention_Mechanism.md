# Attention Mechanism

Reference: [Attention Mechanism](https://erdem.pl/2021/05/introduction-to-attention-mechanism)

Continuing with the [Encoder-Decoder Architecture](Encoder_Decoder.md), the attention mechanism was introduced to help the decoder focus on specific parts of the input sequence when generating each word in the output sequence. This is particularly useful in tasks like machine translation, where certain words in the input may be more relevant to the current output word being generated.

For long sentences, like T=100, it is highly probable that our context vector c is not going to be able to hold all meaningful information from the encoded sequence.

We could create longer and longer context vectors but because RNNs are sequential that won’t scale up. That’s where the Attention Mechanism comes in. The idea is to create a new context vector every timestep of the decoder which attends differently to the encoded sequence. 