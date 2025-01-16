

import numpy as np
import json
import pickle
import os
from nltk.corpus import words, wordnet # type: ignore
from nltk.tag import pos_tag # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import random
from ..core.constant import data_path


