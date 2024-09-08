import json
import random
import math
# Open the file for reading
with open(r'data/mimic/imp_and_find.json', 'r') as f:
    # Load the contents of the file into a variable
    data = json.load(f)
filtered_train = [entry for entry in data['train'] if entry['impression'] != '']
data['train']=filtered_train

filtered_val = [entry for entry in data['val'] if entry['impression'] != '']
data['val']=filtered_val

filtered_test = [entry for entry in data['test'] if entry['impression'] != '']
data['test']=filtered_test

filtered_ft = [entry for entry in data['finetune'] if entry['impression'] != '']
data['finetune']=filtered_ft

print('train size:', len(data['train'])*0.3)
print('validation size:', len(data['val'])*0.3)
print('Finetune size:', len(data['finetune'])*0.3)
print('Test size:', len(data['test'])*0.3)

len_train=0
for i in range(len(data['train'])):
    len_train+=len(data['train'][i]['image_path'])
print("Number of images:", len_train)
    
len_val=0
for i in range(len(data['val'])):
    len_val+=len(data['val'][i]['image_path'])
print("Number of images:", len_val)
    
len_ft=0
for i in range(len(data['finetune'])):
    len_ft+=len(data['finetune'][i]['image_path'])
print("Number of images:", len_ft)
    
len_surr=0
for i in range(len(data['test'])):
    len_surr+=len(data['test'][i]['image_path'])
print("Number of images:", len_surr)

#########################################################



train_split_size= math.ceil(0.40*len(data['train']))
#print(train_split_size)
ft_split_size= math.ceil(0.40*len(data['finetune']))
val_split_size= math.ceil(1*len(data['val']))
test_split_size= math.ceil(1*len(data['test']))

train_sampled_values = random.sample(data['train'], train_split_size)
val_sampled_values = random.sample(data['val'], val_split_size)
ft_sampled_values = random.sample(data['finetune'], ft_split_size)
test_sampled_values = random.sample(data['test'], test_split_size)

ann_file={'train': train_sampled_values,
          'val' : val_sampled_values,
          'finetune': ft_sampled_values,
          'test':test_sampled_values}
out_file = open(r"/home/debodeep.banerjee/R2Gen/data/mimic/imp_n_find_split.json", "w")

json.dump(ann_file, out_file, indent = 6)

out_file.close()

print('details of new splitted data...')
with open(r"/home/debodeep.banerjee/R2Gen/data/mimic/imp_n_find_split.json", 'r') as f:
    # Load the contents of the file into a variable
    data = json.load(f)
#for i in data['test']:
 #   i['split'] = 'test'
#filename = r'/home/debodeep.banerjee/R2Gen/data/mimic/split_find_or_imps.json'
#with open(filename, "w") as file:
 #   json.dump(data, file)
    
print('train size:', len(data['train']))
print('validation size:', len(data['val']))
print('Finetune size:', len(data['finetune']))
print('Test size:', len(data['test']))
