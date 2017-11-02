# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render_to_response
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser

import numpy as np
import tensorflow as tf
import json
import sys
sys.path.append("..")

from mnist import model

x = tf.placeholder("float", [None, 784])
sess = tf.Session()

# restore trained data
with tf.variable_scope("regression"):
    y1, variables = model.regression(x)
saver = tf.train.Saver(variables)
saver.restore(sess, "../mnist/data/regression.ckpt")


with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, "../mnist/data/convolutional.ckpt")


def regression(input):
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()


def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()  

#@csrf_exempt
def mnist(request):
    #print("receive Request: ", JSONParser().parse(request))
    input = ((255 - np.array(JSONParser().parse(request), dtype=np.uint8)) / 255.0).reshape(1, 784)
    #print("Input: ", input)
    output1 = regression(input)
   # print("Output1: ",output1)
    output2 = convolutional(input)
   # print("Output2: ", output2)
    res = HttpResponse(json.dumps({"results": [output1,output2]}), status = 200, content_type="application/json")
    print("Response:",res)
    return res

def main(request):
    return render_to_response('index.html')
