from random import shuffle

tourist_in_data = './data3/tourist_better.in'
tourist_out_data = './data3/tourist_better.out'
guide_in_data = './data3/guide_better.in'
guide_out_data = './data3/guide_better.out'

tourist_in_list = []
tourist_out_list = []
guide_in_list = []
guide_out_list = []

with open(tourist_in_data, 'r') as file:
    for line in file.readlines():
        tourist_in_list.append(line.strip())
print len(tourist_in_list)

with open(tourist_out_data, 'r') as file:
    for line in file.readlines():
        tourist_out_list.append(line.strip())
print len(tourist_out_list)

with open(guide_in_data, 'r') as file:
    for line in file.readlines():
        guide_in_list.append(line.strip())
print len(guide_in_list)

with open(guide_out_data, 'r') as file:
    for line in file.readlines():
        guide_out_list.append(line.strip())
print len(guide_out_list)

# shuffle(tourist_in_list)
# shuffle(tourist_out_list)
# shuffle(guide_in_list)
# shuffle(guide_out_list)

train_in_data = './data3/train.in'
train_out_data = './data3/train.out'
valid_in_data = './data3/valid.in'
valid_out_data = './data3/valid.out'

tourist_train_num = len(tourist_in_list)*9/10
guide_train_num = len(guide_in_list)*9/10
print tourist_train_num
print guide_train_num

with open(train_in_data, 'w') as file:
    for line in tourist_in_list[:tourist_train_num]:
        file.write(line + '\n')
    for line in guide_in_list[:guide_train_num]:
        file.write(line + '\n')

with open(train_out_data, 'w') as file:
    for line in tourist_out_list[:tourist_train_num]:
        file.write(line + '\n')
    for line in guide_out_list[:guide_train_num]:
        file.write(line + '\n')

with open(valid_in_data, 'w') as file:
    for line in tourist_in_list[tourist_train_num:]:
        file.write(line + '\n')
    for line in guide_in_list[guide_train_num:]:
        file.write(line + '\n')

with open(valid_out_data, 'w') as file:
    for line in tourist_out_list[tourist_train_num:]:
        file.write(line + '\n')
    for line in guide_out_list[guide_train_num:]:
        file.write(line + '\n')