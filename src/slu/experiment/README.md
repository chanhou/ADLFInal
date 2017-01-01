# exp1

## Naive approach:
1. use top hypo as translation
2. use the first one label as label data
3. no pruning
4. default parameter of rnn-nlu

### model:  (xx,  9000.ckpt )
speech_act f1 : 0.3649
semantic_tagged f1 : 0.3613
Test_Acc: 0.3637

### model:  (18,  5400.ckpt )
speech_act f1 : 0.3770
semantic_tagged f1 : 0.3278
Test_Acc: 0.3606


# exp2
- diff with exp1: Include tourists into training data

## model (6000 ckpt)

# exp3
Diff with exp2, training using translated chinease to train and testing no need to translate

# exp4
Diff with exp2, prune the semantic tag data

# exp5
Diff with exp4, prune the semantic tag data and add the sequence information i.e. opening information into data

# exp6
Separate training of intent and tagging:
- For intent prediction, 
  - 'Okay' may reflect different meaning, i.e. Closing, Opening, ACK, etc...
  - How to incorporate the sequence information is the most important thing
- For tagging filling,
  - prune the training data

