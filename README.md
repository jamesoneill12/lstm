# RNN Tutorial

There doesn't seem to be an lstm implementation from scratch (at least the forward pass) in the pytorch tutorials so I decided
to upload a simple implementation for learning purposes. 


## Findings

* LSTM and GRU tend towards the RNN performance. 
* For long sequences (s > 40) LSTM is less stables and exhibits spikes while GRU stays smooth.
* For large class number, LSTM randomly spike to very large value
*  
