import numpy as np
from matplotlib.image import imread
from bs4 import BeautifulSoup
import requests
import os
from PIL import Image
import numpy as np
import face_recognition
def load_data():
    data = []
    labels = []
    onehot_labels = []
    path = './data/smiles/smile_facegray'
    length = 4800
    for filename in os.listdir('./data/smiles/smile_facegray'):
        img = imread(path+"/"+filename)
        data.append(img)
        labels.append(0)
        onehot_labels.append([1, 0])
        if(len(data) == length):
            break
    path = './data/smiles/no_smile_facegray'
    for filename in os.listdir('./data/smiles/no_smile_facegray'):
        img = imread(path+"/"+filename)
        data.append(img)
        labels.append(1)
        onehot_labels.append([0, 1])
        if(len(data) == length*2):
            break
    data = np.asarray(data, dtype=np.float32)
    data = np.expand_dims(data,axis=1)
    labels = np.asarray(labels)
    onehot_labels = np.asarray(onehot_labels)
    return data, labels, onehot_labels


def gather_data():
    '''
    gather smiles and no smiles images
    '''
    base_url = "https://www.gettyimages.com/photos/{}"
    terms = ["no smile", "smile"]
    PAGES = 100
    for term in terms:
        url = base_url.format(term)
        temp = term.replace(" ", "_")
        path = './data/'+temp
        img_num = 0
        os.makedirs(path)
        for page in range(PAGES):
            params = {
                'page': f'{page}',
                'numberofpeople': 'one',
                'recency': 'anydate',
                'sort': 'mostpopular',
            }
            req = requests.get(url, params=params)
            soup = BeautifulSoup(req.content)
            for img in soup.find_all('img', {'class': 'gallery-asset__thumb gallery-mosaic-asset__thumb'}):
                img_url = img['src']
                filepath = path+"/"+str(img_num) + ".jpg"
                with open(filepath, "wb") as file:
                    file.write(requests.get(img_url).content)
                img_num += 1
            print("Finished: Term: {} Page: {}".format(term, page))

def face_grayscale():
    '''
    find the face and gray scale
    '''
    height = 100
    width = 100
    search_terms = ["no smile", "smile"]
    for term in search_terms:
        temp = term.replace(" ", "_")
        path = './data/'+temp
        new_path = './data/'+temp+"_facegray"
        os.makedirs(new_path)
        for filename in os.listdir(path):
            img = Image.open(path+"/"+filename)
            img = img.convert("L")
            image = face_recognition.load_image_file(path+"/"+filename)
            faces = face_recognition.face_locations(image)
            if(len(faces) == 1):
                top, right, bottom, left = faces[0]
                box = (left, top, right, bottom)
                resize_image = img.resize((width, height), box = box)
                resize_image.save(new_path+"/"+filename)
        print("Finished: Term: {}".format(term))
