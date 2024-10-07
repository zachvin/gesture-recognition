import json

with open('video-list.json', 'r') as f:
    json_data = json.load(f)

videos = {}

for entry in json_data:
    gloss = entry['gloss']
    for instance in entry['instances']:
        videos[gloss] = videos.get(gloss, []) + [instance['video_id']]

with open('video-lookup.json', 'w') as f:
    json.dump(videos, f)

print(videos.keys())