import pickle
import numpy

m0 = []
m1 = []
m2 = []

with open("kerryn_dec_M0.pickle", "rb") as file:
    m0 = pickle.load(file)

with open("kerryn_dec_M1.pickle", "rb") as file:
    m1 = pickle.load(file)    
    
with open("kerryn_dec_M2.pickle", "rb") as file:
    m2 = pickle.load(file)    

all = m0 + m1 + m2

outfile = open("kerryn_dec_all.pickle",'wb')
pickle.dump(all, outfile)