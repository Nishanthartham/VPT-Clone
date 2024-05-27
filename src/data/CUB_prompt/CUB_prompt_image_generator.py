import json

# Specify the path to your JSON file
json_file_path = 'train.json'

# Read the JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

di = {str(i).zfill(3): 0 for i in range(1, 201)}
new_data={}
for image_path, label in data.items():
    if di[label] >= 1:
        print(di[label])
    else:
        new_data[image_path] = label
        di[label]+=1

with open("prompt_train.json",'w') as f:
    json.dump(new_data,f,indent=2)
