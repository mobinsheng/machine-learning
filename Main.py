#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

import run_minist2
import plane_fitting
import arcnn
import num_sum
import simple_lstm

if __name__ == '__main__':
    #run_minist2.run_minist()
    #plane_fitting.run_plane_fitting()
    #arcnn.run_arcnn()
    #num_sum.run_num_sum()
    simple_lstm.run_simple_lstm()