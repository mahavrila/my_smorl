
## SMORL replication preparation
This repo contains data and code for pre-processing and post-processing of data from SMORL paper

## Where to start
RC15 dataset preprocessing is described in clicks.ipynb and retail rocket dataset  
in retail_rocket.ipynb.

### Created Datasets
Original SMORL papers used also zero-length sequences (embedding only) in all (training,  
testing and validation) datasets. Note, that they still used sessions of minimal length  
of 3 interactions, but each session was used to generate training data by "sliding window"  
that was implemented in a way that in first step it took 0 items as input, and then 1 item,  
2 items etc. Thus empty sequences (embedding only) appeared in frame buffer - for more  
details see clicks.ipynb file. 

I decided to create 3 different dataset versions named _skip_0, _skip_1 and _skip_2:
  - _skip_0 corresponds to original datasets with minimal sequence_length = 0
  - _skip_1 corresponds to modified dataset where 0-item-long sequences are removed
  - _skip_2 corresponds to modified dataset where  also 1-item-long sequences are removed

### Stsandard dataset

Initiall benchmark showed that **_skip_1** behave much better than original dataset and  
we consider them as standard for further work.