# -*- coding: utf-8 -*-
__author__ = "Seokhwan Kim"

"""
Dataset converter for SLG pilot task of DSTC5.

It converts the training and development datasets for DSTC5 into the formats for SLG pilot task.
"""

import argparse
import sys
import dataset_walker
import json
import os
from semantic_tag_parser import SemanticTagParser

def main(argv):
    parser = argparse.ArgumentParser(description='Dataset Converter for SAP pilot task.')
    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True, help='The target dataset to be converted')
    parser.add_argument('--dataroot', dest='dataroot', action='store', required=True, metavar='PATH',  help='Will look for corpus in <destroot>/...')
    parser.add_argument('--roletype', dest='roletype', action='store', choices=['GUIDE',  'TOURIST'], required=True,  help='Target role')
    args = parser.parse_args()

    """original

    dataset = dataset_walker.dataset_walker(args.dataset, dataroot=args.dataroot, labels=True, translations=False)


    for call in dataset:
        session_id = call.log["session_id"]

        input_guide = {u'session_id': session_id, u'utterances': [], u'roletype': u'Guide'}
        output_guide = {u'session_id': session_id, u'utterances': [], u'roletype': u'Guide'}

        input_tourist = {u'session_id': session_id, u'utterances': [], u'roletype': u'Tourist'}
        output_tourist = {u'session_id': session_id, u'utterances': [], u'roletype': u'Tourist'}


        for (log_utter, _, label_utter) in call:
            speaker = log_utter['speaker']
            utter_index = log_utter['utter_index']
            transcript = log_utter['transcript']

            speech_act = label_utter['speech_act']

            mention_words = []
            curr_cat = None
            curr_attrs = None

            semantic_tags = []

            for semantic_tagged in label_utter['semantic_tagged']:
                parser = SemanticTagParser(False)
                parser.feed(semantic_tagged)

                for word, (bio, cat, attrs) in zip(parser.get_word_seq(), parser.get_word_tag_seq()):
                    if bio == 'I':
                        mention_words.append(word)
                    else:
                        if curr_cat is not None:
                            semantic_tags.append({
                                u'main': curr_cat,
                                u'attributes': curr_attrs,
                                u'mention': ' '.join(mention_words)
                            })

                        mention_words = []
                        curr_cat = None
                        curr_attrs = None

                        if bio == 'B':
                            mention_words = [word]
                            curr_cat = cat
                            curr_attrs = {}
                            for key, value in attrs:
                                curr_attrs[key] = value

                if curr_cat is not None:
                    semantic_tags.append({
                        u'main': curr_cat,
                        u'attributes': curr_attrs,
                        u'mention': ' '.join(mention_words)
                    })
            if speaker == 'Guide':
                input_guide[u'utterances'].append({
                    u'utter_index': utter_index,
                    u'speaker': speaker,
                    u'semantic_tags': semantic_tags,
                    u'speech_act': speech_act
                })
                output_guide[u'utterances'].append({
                    u'utter_index': utter_index,
                    u'transcript': transcript
                })
                input_tourist[u'utterances'].append({
                    u'utter_index': utter_index,
                    u'speaker': speaker,
                    u'transcript': transcript,
                    u'semantic_tags': semantic_tags,
                    u'speech_act': speech_act
                })
                # transcript is guide says
            elif speaker == 'Tourist':
                input_tourist[u'utterances'].append({
                    u'utter_index': utter_index,
                    u'speaker': speaker,
                    u'semantic_tags': semantic_tags,
                    u'speech_act': speech_act
                })
                output_tourist[u'utterances'].append({
                    u'utter_index': utter_index,
                    u'transcript': transcript
                })
                input_guide[u'utterances'].append({
                    u'utter_index': utter_index,
                    u'speaker': speaker,
                    u'transcript': transcript,
                    u'semantic_tags': semantic_tags,
                    u'speech_act': speech_act
                })
                #transcript is tourist says

        path = os.path.join(os.path.abspath(args.dataroot), '%03d' % (session_id,))

        with open(os.path.join(path, 'slg.guide.in.json'), 'w') as fp:
            json.dump(input_guide, fp)
        with open(os.path.join(path, 'slg.guide.label.json'), 'w') as fp:
            json.dump(output_guide, fp)
        with open(os.path.join(path, 'slg.tourist.in.json'), 'w') as fp:
            json.dump(input_tourist, fp)
        with open(os.path.join(path, 'slg.tourist.label.json'), 'w') as fp:
            json.dump(output_tourist, fp)

    original"""


    trainset = dataset_walker.dataset_walker(args.dataset, dataroot=args.dataroot, labels=True, translations=True, task='SLG', roletype=args.roletype.lower())
    sys.stderr.write('Loading training instances ... ')

    for call in trainset:
        session_id = call.log["session_id"]

        inputPath = os.path.join(os.path.abspath(args.dataroot), '%03d' % (session_id,))
        inputFile = open(os.path.join(inputPath, args.roletype.lower()), 'w')

        inputSplitPath = os.path.join(os.path.abspath(args.dataroot), '%03d' % (session_id,))
        inputSplitPath = os.path.join(inputSplitPath, args.roletype.lower() + "_split")
        inputSplitFile = open(inputSplitPath, 'w')

        outputPath = os.path.join(os.path.abspath(args.dataroot), '%03d' % (session_id,))
        outputPath = os.path.join(outputPath, args.roletype.lower() + "_out")
        outputFile = open(outputPath, 'w')

        for (log_utter, translations, label_utter) in call:
            if log_utter['speaker'].lower() == args.roletype.lower():
                instance = {'semantic_tags': log_utter['semantic_tags'], 'speech_act': log_utter['speech_act']}
                
                string = ""
                split = ""
                element = []
                for dictionary in log_utter['speech_act']:
                    elementSplit = []
                    for sub in dictionary[u'attributes']:
                        element.append(sub)
                        elementSplit.append(sub)
                    element.append(dictionary[u'act'])
                    elementSplit.append(dictionary[u'act'])
                    intentSplit = "_".join(elementSplit)
                    split += " "
                    split += intentSplit
                intent = "_".join(element)
                string += intent
                    
                #print string
                string += " "
                split += " "
                
                tag = ""
                #print log_utter['semantic_tags']
                # tag
                for dictionary in log_utter['semantic_tags']:
                    element = []
                    element.append(dictionary[u'main'])
                    subdict = dictionary[u'attributes']
                    for key in subdict:
                        element.append(subdict[key])
                    tag += "_".join(element)
                    tag += " "
                split += tag
                string += tag
                #print len(translations['translated'])
                #print string
                



#############        output        #############
#############        output        #############
#############        output        #############
#############        output        #############

                
                tag = ""
                mentionList = []
                for dictionary in log_utter['semantic_tags']:
                    element = []
                    element.append(dictionary['main'])
                    subdict = dictionary['attributes']
                    for key in subdict:
                        element.append(subdict[key])
                    tag += "_".join(element)
                    tag += " "
                    mentionList.append(dictionary['mention'])
                tag = tag.strip()
                tagList = tag.split(" ")
                # print log_utter['semantic_tags']
                # print mentionList
                # print tag.split(" ")
                # print "================"

                if len(tag) == 0:
                    if log_utter['utter_index'] != translations['utter_index']:
                        print log_utter['utter_index'], translations['utter_index']
                    for i in range(len(translations['translated'])):
                        # print translations['translated'][i]
                        outString = translations['translated'][i]['hyp']
                        indexList = []
                        # print translations['translated'][i]['align']
                        for j in range(len(translations['translated'][i]['align'])):
                            # print len(translations['translated'][i]['align'][j][1])
                            #translations['translated'][i]['align'][j][0] => mention
                            # print args.roletype.lower()
                            # print translations['translated'][i]
                            # print translations['translated'][i]['align'][j]
                            # print translations['translated'][i]['align'][j][1]
                            if len(translations['translated'][i]['align'][j][1]) > 0:
                                indexList.append(max(translations['translated'][i]['align'][j][1]) + 1)
                            indexList.sort()

                        #print indexList
                        #print outString
                        for j in range(len(indexList)):
                            outString = outString[:indexList[j]] + ' ' + outString[indexList[j]:]
                            for k in range(len(indexList)):
                                indexList[k] += 1

     #############      file I/O        #############                   
                        inputFile.write(string.strip() + "\n")
                        inputSplitFile.write(split.strip() + "\n")

                        outString = outString.strip()
                        # print outString
                        outputFile.write(outString.strip().encode('utf-8') + "\n")

                        #print outString.split(" ")
                        # print "next hyp"
    #############      file I/O        #############


                else:
                    if log_utter['utter_index'] != translations['utter_index']:
                        print log_utter['utter_index'], translations['utter_index']
                    for i in range(len(translations['translated'])):
                    # for i in range(1):
                        # print len(translations['translated'])  1 ~ 5
                        # print translations['translated'][i]
                        trash = 0
                        find = 0
                        # print translations['translated'][i]['align']
                        outString = translations['translated'][i]['hyp']
                        minimum = 2147483647
                        maximum = -1
                        for j in range(len(mentionList)):
                            words = mentionList[j].split(" ")
                            for k in range(len(words)):
                                find = 0
                                for l in range(len(translations['translated'][i]['align'])):
                                    if len(translations['translated'][i]['align'][l][1]) == 0:
                                        trash = 1
                                        break
                                    elif translations['translated'][i]['align'][l][0] == words[k]:
                                        minimum = min(min(translations['translated'][i]['align'][l][1]), minimum)
                                        maximum = max(max(translations['translated'][i]['align'][l][1]), maximum)
                                        find = 1
                                if find == 0:
                                    trash = 1
                                    break
                            # print maximum, minimum, len(words) * 3
                            if maximum - minimum > len(words) * 3:
                                # print maximum, minimum, len(words) * 3
                                trash = 1
                                break
                        words = " ".join(mentionList).split(" ")

                        # print words
                        # print translations['translated'][i]['align']
                        for j in range(len(translations['translated'][i]['align'])):
                            same = 0
                            for k in range(len(words)):
                                # in mentionList
                                if translations['translated'][i]['align'][j][0] == words[k]:
                                    same = 1
                                    break
                            if same == 1:
                                continue
                            # not in mentionList
                            # no alignment
                            if len(translations['translated'][i]['align'][j][1]) == 0:
                                trash = 1
                                break
                            for k in range(len(translations['translated'][i]['align'][j][1])):
                                if translations['translated'][i]['align'][j][1][k] >= minimum and translations['translated'][i]['align'][j][1][k] <= maximum:
                                    # print translations['translated'][i]['align'][j][1][k], minimum, maximum
                                    trash = 1
                                    break

                        if trash == 1:
                            # print translations['translated'][i]['align']
                            # print mentionList
                            # print "============================"
                            break
                        outStringList = list(outString)
                        # print translations['translated'][i]['align']
                        added = []
                        for j in range(len(translations['translated'][i]['align'])):
                            same = 0
                            for k in range(len(words)):
                                # in mentionList
                                if translations['translated'][i]['align'][j][0] == words[k]:
                                    same = 1
                                    break
                            # not in mentionList
                            for k in range(len(added)):
                                # no need to add ' ' again
                                if added[k] == max(translations['translated'][i]['align'][j][1]):
                                    same = 1
                            if same == 1:
                                continue
                            outStringList[max(translations['translated'][i]['align'][j][1])] += ' '
                            added.append(max(translations['translated'][i]['align'][j][1]))
                        # print outString
                        # outString = "".join(outStringList)
                        # print outString
                        # print "=============="
                        for j in range(len(mentionList)):
                            start = 2147483647
                            end = -1
                            words = mentionList[j].split(" ")
                            for k in range(len(words)):
                                for l in range(len(translations['translated'][i]['align'])):
                                    if translations['translated'][i]['align'][l][0] == words[k]:
                                        start = min(min(translations['translated'][i]['align'][l][1]), start)
                                        end = max(max(translations['translated'][i]['align'][l][1]), end)
                            # print outString
                            # print mentionList[j]
                            # print outString[start:end+1]
                            # print tagList[j]
                            # print "==========================="
                            outStringList[start] = ' ' + tagList[j] + ' '
                            for k in range(start + 1, end + 1):
                                outStringList[k] = ''
                        # print mentionList
                        # print outString
                        outString = ''.join(outStringList).replace('  ', ' ')
                        # print outString
                        # print "==================="

     #############      file I/O        #############                   
                        inputFile.write(string.strip() + "\n")
                        inputSplitFile.write(split.strip() + "\n")

                        outString = outString.strip()
                        # print outString
                        outputFile.write(outString.strip().encode('utf-8') + "\n")

                        #print outString.split(" ")
                        # print "next hyp"
    #############      file I/O        #############

#############        output        #############
#############        output        #############
#############        output        #############
#############        output        #############                  
    
if __name__ == "__main__":
    main(sys.argv)
