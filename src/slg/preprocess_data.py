import json
import os
import sys

data_path = './natural_language_generation/data'

# ====== deal with testing data ==========
#test_data = os.path.join(data_path, 'test.txt')
test_data = sys.argv[1].replace('\r', '')
test_in_sentences_list = list()
with open(test_data) as file_id:
    for input1 in file_id:
        idx1 = input1.find('(')
        idx2 = input1.find(')')
        intent = input1[:idx1]
        dict_str = input1[idx1+1:idx2]
        print ('dict_str:\n%s' % dict_str)
        input_sentence = intent
        if dict_str:
            # build the inquire dictionary
            if '=' in dict_str:
                temp_dict = dict()
                pairs = dict_str.split(';')
                for pair in pairs:
                    idx = pair.find('=')
                    value_str = pair[:idx]
                    value_str = value_str.strip('\'')
                    value_str = value_str.strip('\"')

                    key_str = pair[idx+1:]
                    key_str = key_str.strip('\'')
                    key_str = key_str.strip('\"')

                    temp_dict[key_str] = value_str
                for _, value in temp_dict.iteritems():
                    input_sentence = input_sentence + ' \"' + value + '\"'
            else:
                input_sentence = input_sentence + ' ' + dict_str
        # repeat because one input corresponds to two outputs
        test_in_sentences_list.append(input_sentence)

parsed_test_file = os.path.join(data_path, 'parsed_test.txt')
with open(parsed_test_file, 'w') as file_id:
    for sentence in test_in_sentences_list:
        file_id.write(sentence + '\n')

# ====== deal with training data or validation data ==========
# val_data_json = os.path.join(data_path, 'train.json')
# valid_in_sentences_list = list()
# valid_out_sentences_list = list()
# with open(val_data_json) as valid_file_id:
#     val_data = json.load(valid_file_id)
#     #val_data = val_data[:5]
#     for data in val_data:
#         input1 = str(data[0])
#         outputsence1 = str(data[1])
#         outputsence2 = str(data[2])
#         print('\nbefore repalcing')
#         print(input1)
#         print(outputsence1)
#         print(outputsence2)
#         idx1 = input1.find('(')
#         idx2 = input1.find(')')
#         intent = input1[:idx1]

#         dict_str = input1[idx1+1:idx2]
#         print ('dict_str:\n%s' % dict_str)
#         input_sentence = intent
#         if dict_str:
#             # build the inquire dictionary
#             if '=' in dict_str:
#                 temp_dict = dict()
#                 pairs = dict_str.split(';')
#                 for pair in pairs:
#                     idx = pair.find('=')
#                     value_str = pair[:idx]
#                     value_str = value_str.strip('\'')
#                     value_str = value_str.strip('\"')

#                     key_str = pair[idx+1:]
#                     key_str = key_str.strip('\'')
#                     key_str = key_str.strip('\"')

#                     temp_dict[key_str] = value_str

#                 for key, value in temp_dict.iteritems():
#                     input_sentence = input_sentence + ' \"' + value + '\"'
#                     if key in outputsence1:
#                         outputsence1 = outputsence1.replace(key, '\"'+value+'\"')
#                     if key in outputsence2:
#                         outputsence2 = outputsence2.replace(key, '\"'+value+'\"')

#             else:
#                 input_sentence = input_sentence + ' ' + dict_str
#         # repeat because one input corresponds to two outputs
#         valid_in_sentences_list.append(input_sentence)
#         valid_in_sentences_list.append(input_sentence)

#         valid_out_sentences_list.append(outputsence1)
#         valid_out_sentences_list.append(outputsence2)

#         print('after repalcing')
#         print(outputsence1)
#         print(outputsence2)

# valid_in_file = os.path.join(data_path, 'train.in')
# with open(valid_in_file, 'w') as file_id:
#     for sentence in valid_in_sentences_list:
#         file_id.write(sentence + '\n')

# valid_out_file = os.path.join(data_path, 'train.out')
# with open(valid_out_file, 'w') as file_id:
#     for sentence in valid_out_sentences_list:
#         file_id.write(sentence + '\n')