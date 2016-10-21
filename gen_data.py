import numpy as np
import clear_data

clear_data.clear_rawdata("2013",False)
clear_data.clear_rawdata("2014",False)
clear_data.clear_rawdata("2015",False)
clear_data.clear_rawdata("2016",True)

clear_data.gen_data()