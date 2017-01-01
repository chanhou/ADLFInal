import os
import sys
reload(sys)
sys.path.insert(0, '../dstc5/scripts')
sys.setdefaultencoding('utf-8')


import nltk
import argparse, dataset_walker, time, json 

from semantic_tag_parser import SemanticTagParser
import re
import lm

#####
# which data set to chooce
####
targetF = 'train' # train, valid
testF = 'dev' # dev, test_slu

os.system('rm '+'./rnn-nlu-tagging/data/slu/*.txt')
os.system('rm '+'./rnn-nlu-tagging/data/slu/'+targetF+'/*')

fin = open('./rnn-nlu-tagging/data/slu/'+targetF+'/'+targetF+'.seq.in','w')
fout = open('./rnn-nlu-tagging/data/slu/'+targetF+'/'+targetF+'.seq.out','w')
flabel = open('./rnn-nlu-tagging/data/slu/'+targetF+'/'+targetF+'.label','w')
flabel2 = open('./rnn-nlu-tagging/data/slu/'+targetF+'/'+targetF+'.label2','w')


####
# original function provided by baseline_slu.py
####
def add_instance(utter, speech_act, semantic_tagged):
    tokenized = __tokenize(utter, semantic_tagged)
    if tokenized is None:
        return False

    semantic_instance = []
    original_sent = []
    IOB_tag = []
    for word, (bio, tag, attrs) in tokenized:
        if bio is None:
            sem_label = 'O'
        else:
            cat = None
            for attr, val in attrs:
                if attr == 'cat':
                    cat = val
            sem_label = '%s-%s_%s' % (bio, tag, cat)
        ####
        # get what we want into list
        ####
        original_sent.append(unicode(word.lower()))
        IOB_tag.append(unicode(sem_label))

    sa_label_list = []
    for sa in speech_act:
        sa_labels = ['%s_%s' % (sa['act'], attr) for attr in sa['attributes']]
        sa_label_list += sa_labels
    sa_label_list = sorted(set(sa_label_list))
        
    word_feats = ' '.join([word.lower() for word, _ in tokenized])
        

    #### 
    # dump data out
    ####
    '''
    #### 
    # Duplicate method for multiple label
    ####
    for llll in sa_label_list:
        fin.write(' '.join(original_sent)+'\n')
        fout.write(' '.join(IOB_tag)+'\n')
        flabel.write(llll+'\n')
    '''
    cccc = 0
    for dddd in range(len(IOB_tag)):
        if IOB_tag[dddd]=='O': cccc += 1
    if cccc != len(IOB_tag):
        fin.write(' '.join(original_sent)+'\n')
        fout.write(' '.join(IOB_tag)+'\n')
        # choose the first one as the label
        flabel.write( sa_label_list[0]  +'\n')
        # merge all the label into a unique label
        flabel2.write( '_'.join(sa_label_list)+'\n')

    return True

def __tokenize(utter, semantic_tagged=None):
    result = None
    if semantic_tagged is None:
        result = [(word, None) for word in nltk.word_tokenize(utter)]
    else:
        parser_raw = SemanticTagParser(False)
        parser_tagged = SemanticTagParser(False)

        segmented = ' '.join(nltk.word_tokenize(utter))
        tagged = ' '.join(semantic_tagged)

        parser_raw.feed(segmented)
        parser_tagged.feed(tagged)

        raw_chr_seq = parser_raw.get_chr_seq()
        raw_space_seq = parser_raw.get_chr_space_seq()

        tagged_chr_seq = parser_tagged.get_chr_seq()
        tagged_space_seq = parser_tagged.get_chr_space_seq()

        if raw_chr_seq == tagged_chr_seq:
            merged_space_seq = [
                x or y for x, y in zip(raw_space_seq, tagged_space_seq)]
            word_seq = parser_tagged.tokenize(merged_space_seq)
            tag_seq = parser_tagged.get_word_tag_seq()

            result = [(word, tag) for word, tag in zip(word_seq, tag_seq)]

    return result

####
# generate training data set
####
dataroot = '../../data_for_stu/'+targetF
trainset = 'dstc5_'+targetF

trainset = dataset_walker.dataset_walker(trainset, dataroot=dataroot, labels=True, translations=True)

ngram = lm.ArpaLM()
#ngram.read('../dstc5/scripts/dstc5.cn.3.lm')
ngram.read('en-70k-0.2-pruned.lm.gz')

count = 0 
for call in trainset:
    print(count)
    for (log_utter, translations, label_utter) in call:
        if (log_utter['speaker'] == 'Guide' or log_utter['speaker'] == 'Tourist' ):
        #if (log_utter['speaker'] == 'Guide'):
            #print(log_utter['transcript'],label_utter['speech_act'],label_utter['semantic_tagged'])
            add_instance(log_utter['transcript'], label_utter['speech_act'], label_utter['semantic_tagged'])
    #break
    count += 1

fin.close()
fout.close()
flabel.close()

###################
# generate test set
##################

dataroot = '../../data_for_stu/'+testF
testset = 'dstc5_'+testF
testset = dataset_walker.dataset_walker(testset, dataroot=dataroot, labels=False, translations=True)

testF = 'test'
os.system('rm '+'./rnn-nlu-tagging/data/slu/'+testF+'/*')
ftest = open('./rnn-nlu-tagging/data/slu/'+testF+'/'+testF+'.seq.in','w')
fhyp = open('./predict-sep/hyp','r')

for call in testset:
    for (log_utter, translations, label_utter) in call:
        if (log_utter['speaker'] == 'Guide'):
            if len(translations['translated']) > 0:
                '''
                best_hyp = []
                best_score = -100000
                best_ind = -1
                for qqqq in range(len(translations['translated'])):
                    top_hyp = translations['translated'][qqqq]['hyp']
                    tokenized = __tokenize(top_hyp)
                    score = 0
                    for i in range(len(tokenized)):
                        if i==0: 
                            score += ngram.score((tokenized[i]))
                        else: 
                            score += ngram.score((tokenized[i-1],tokenized[i]))
                    if score > best_score:
                        best_score = score
                        best_hyp = tokenized
                        best_ind = qqqq
                    #print top_hyp
                #print best_ind, best_hyp
                '''
                best_hyp = translations['translated'][int(fhyp.readline())]['hyp']
                best_hyp = __tokenize(best_hyp)
                tokenized = best_hyp 
            #tokenized = __tokenize(log_utter['transcript'])
                word_feats = ' '.join([word.lower() for word, _ in tokenized])
                ftest.write(word_feats+'\n')
                #fhyp.write(str(best_ind)+'\n')

ftest.close()
#if __name__ == "__main__":
#    main()