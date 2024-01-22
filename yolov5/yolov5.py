import torch
import sys
import re

def getTag(imagePath):
    class __Autonomy__(object):
        def __init__(self):
            self._buff = ""
        def write(self, out_stream):
            self._buff += out_stream

    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

    # Inference
    results = model(imagePath) # 结果


    ## Get Output
    current = sys.stdout
    a = __Autonomy__()
    sys.stdout = a

    # Results
    results.print()  

    sys.stdout = current

    pattern = re.compile('([1-9])\s([a-z]+)')
    str_test = a._buff
    m = re.findall(pattern,str_test)
    tag = []
    for i in range(len(m)):
        tag.append(m[i][1])
    return tag