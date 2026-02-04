
import json

with open('facescrub_uncropped_features_list.json', 'r') as f:
  c = f.read()
  data = json.loads(c)
  data = data['id']
  for _id in data:
    print(_id)
