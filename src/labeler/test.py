import json
import track_labels as tl


track = tl.TrackLabels(100)
track.label_behavior(0,5)
track.label_not_behavior(25,50)
track.label_behavior(51,60)
print(track.encode())
json.dumps(track.encode())


