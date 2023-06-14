#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import json
import numpy as np
from tensorflow.keras.models import load_model
import requests
from flask import Flask, request, jsonify


# In[7]:


app = Flask(__name__)


# In[9]:


model = load_model('rating_recommendation_model.h5')


# In[14]:


@app.route("/tourify/predict/<int:id>", methods=["GET"])
def predict(id):
    id_place = range(1,20)
    tourism_data = np.array(list(set(id_place)))
    tourism_data[:10]
    
    id_user = id

    user = np.array([id_user for i in range(len(tourism_data))])
    user[:10]

    precitions = model.predict([user, tourism_data])

    predictions = np.array([a[0] for a in precitions])

    recommended_tourist_ids = (predictions).argsort()[:10]
    list_recommended_tourist_ids = recommended_tourist_ids.tolist()

    place_id = []
    for i in list_recommended_tourist_ids:
            r = requests.get(
            'https://tourify-314510.et.r.appspot.com/idtoplaceid/' + str(i))
            s = r.json()
            place_id.append(s['data'])
        
    response_json = {
            "data": id_user,
            "prediction": place_id
    }
        
    return jsonify(response_json)

