import requests, zipfile, os, errno
import nltk 
from nltk.tokenize import sent_tokenize

ALICE_URL = 'https://llds.ling-phil.ox.ac.uk/llds/xmlui/bitstream/handle/20.500.14106/1476/alice28-1476.txt'
WIZARD_URL = 'https://llds.ling-phil.ox.ac.uk/llds/xmlui/bitstream/handle/20.500.14106/1740/wizoz10-1740.txt'

