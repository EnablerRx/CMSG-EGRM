from reverse import reverse_valence
from retrieve import retrieveCommonSense
from rank import rankContext , getRoberta
import pickle
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
test_data = pickle.load(open("sen_att_pos_01.pik", 'rb'), encoding='iso-8859-1') # first text for all images
roberta = getRoberta()


resultlist = [] 
filename = 'result_CMSG_woI.json'
isNew = True # first generated item
for index in range(len(test_data['des_sen'])): 
    # generate sarcasm caption
    resultlist = []
    utterance = ' '.join(test_data['des_sen'][index]) + '.' # first text
    img = './testCMSG_503/' + test_data['img_files'][index] # input image
    rov = reverse_valence(utterance).capitalize() # reverse the rank of the input sentence
    op = retrieveCommonSense(utterance) # commonsense inference and commonsense sentences retrive 
    commonsense, extra = op[0], op[1]
    mostincongruent = rankContext(roberta, rov, commonsense, img, extra) # 新加入img
    sarcasm = rov + ' ' + mostincongruent # generated sarcasm caption

    # save result image-text pairs
    dic = {'img_file': test_data['img_files'][index], 'caption': sarcasm}
    resultlist.append(dic)


    # save result in json file
    f = open(filename)
    exist_data = json.load(f) 
    f.close()

    exist_data.append(resultlist[0])
    jsObj = json.dumps(exist_data, indent=4)
    with open(filename, "w") as fw:
        fw.write(jsObj)
        fw.close()
    isNew = False




	
