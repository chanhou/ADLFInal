# exp1

## Naive approach:
1. use top hypo as translation
2. use the first one label as label data
3. no pruning
4. default parameter of rnn-nlu

### model:  (xx,  9000.ckpt )
- speech_act f1 : 0.3649
- semantic_tagged f1 : 0.3613
- Test_Acc: 0.3637

### model:  (18,  5400.ckpt )
- speech_act f1 : 0.3770
- semantic_tagged f1 : 0.3278
- Test_Acc: 0.3606


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
    - concate the previous word
- For tagging filling,
  - prune the training data
- Translated suck:
  - use http://cmusphinx.sourceforge.net/wiki/download provided lm model
  - Ex1
    - Basic ally all know it .
    - bas ic ally all i know .
    - basic ally all know .
    - bas ically, i know .
  - Ex2
    - t hank you
    - thank you

### trying1
- use default 0 as top hyp
- separate training of intent and tagging
  - intent: use only guide information but include previous content
  - tagging: pruninig 
- model
  - tagging 8400
  - intent 3300
- performance
  - speech_act f1 : 0.3858
  - semantic_tagged f1 : 0.3455
  - Test_Acc: 0.3724

### trying2
- diff with trying1, use lm to choose best hyp
- preformance
  - speech_act f1 : 0.3786
  - semantic_tagged f1 : 0.
  - Test_Acc: 0.

### trying3
- using language model to choose best hypo
- tagging
    - valid using all data (trying1 use prune set)
    - using 15 (4800, 0.3476) 
- intent
    - figure out how to perform multilabel classification
    - concate the label with '|', result as 833 (original 65) unique labels ><
    - ignore the above option, duplicate data to multiple based on the label
    - using 9 (3000, 0.4683) 
- performance
    - speech_act f1 : 0.4471
    - semantic_tagged f1 : 0.3467
    - Test_Acc: 0.4137

### trying4
- tagging
    - use all the dataset without prunning
    - 8100 ckpt, f1 0.407855
- intent
    - same ass trying2 
- performance
    - speech_act f1 : 0.4469
    - semantic_tagged f1 : 0.3691
    - Test_Acc: 0.4210
### trying5
- first split the semantic tagged by using label information => data cleaning stage
- applying the previous stage method
- tagging
    - 8100 ckpt, 0.424958
- intent
    - 8100 ckpt, 0.467489
- preformance
    - speech_act f1 : 0.4359
    - semantic_tagged f1 : 0.3741
    - Test_Acc: 0.4153
- intent, 2100
- tagging, 6000
- performance
    - speech_act f1 : 0.4478
    - semantic_tagged f1 : 0.3701
    - Test_Acc: 0.4219
- ensemble 17 algs
-performance
    - speech_act f1 : 0.4558
    - semantic_tagged f1 : 0.3674
    - Test_Acc: 0.4263

