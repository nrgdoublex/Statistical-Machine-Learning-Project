import numpy as np
from Polynomial import polyfeature
from sklearn.preprocessing import PolynomialFeatures


#polyfeature.poly2_feature("test.txt","testoutput.txt")
# polyfeature.poly3_feature("test.txt","testoutput.txt")
# 
# file = open("test.txt")
# lines = file.readlines()
# for line in lines:
#     data_list = line.rstrip('\n').split(',')
#     data_list.pop()
#     poly = PolynomialFeatures(3,include_bias=False)
#     print(poly.fit_transform(data_list))


arr = [1,2,3]
print(np.sum([item > 1 for item in arr]))