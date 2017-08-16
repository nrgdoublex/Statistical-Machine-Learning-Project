import numpy as np

def match_player_salary(player,salary):
    name_player = player[0].split(',')[0]
    name_salary = salary[0].split(' ')[1]
    if name_player != name_salary:
        return False
    if player[1] != salary[1]:
        return False
    if player[2] != salary[2]:
        return False
    return True



def clear_rawdata(year,test):
    out_file = open(year + "/"+ year + "stats.txt",'w')
    
    #read 2016 stats
    for i in range(0,25):
        file = year + "/stat" + str(i+1) + ".txt"
        with open(file) as f:
            content = f.readlines()
            for line in range(1,len(content)):
                stat = content[line].rstrip('\n').split('\t')
                if stat[3] != 'P' or stat[0]=='RK':
                    for idx in range(1,len(stat)):
                        out_file.write("%s\t" %stat[idx])
                    out_file.write('\n')
                    
    out_file.close()
    
    #read 2015 salaries
    out_file = open(year + "/" + year + "salaries.txt",'w')
    in_file = open(year + "/salary.txt")
    lines = in_file.readlines()
    idx = 1
    while idx < len(lines):
        line = lines[idx+2].rstrip('\n').split('\t')
        if test==True:
            if line[1] != 'RP' and line[1] != 'SP' and line[1] != 'P' and line[3] == '1 (2016)':
                out_file.write("%s\t" %lines[idx+1].rstrip('\n'))
                out_file.write("%s\t%s\t%s" %(line[0],line[1],line[5].strip(' $').replace(',','')))
                out_file.write('\n')
        else:
            if line[1] != 'RP' and line[1] != 'SP' and line[1] != 'P':
                out_file.write("%s\t" %lines[idx+1].rstrip('\n'))
                out_file.write("%s\t%s\t%s" %(line[0],line[1],line[5].strip(' $').replace(',','')))
                out_file.write('\n')
        idx += 3
    in_file.close()
    out_file.close()  
    
    #match player and salary
    player = open(year + "/" + year + "stats.txt")
    salary = open(year + "/" + year + "salaries.txt")
    total = open(year + "/" + year + ".txt",'w')
    player_lines = player.readlines()
    salary_lines = salary.readlines()
    player_matrix = []
    salary_matrix = []
    for line in player_lines:
        list = line.rstrip('\t\n').replace(' ','').split('\t')
        player_matrix.append(list)
    for line in salary_lines:
        list = line.rstrip('\t\n').split('\t')
        salary_matrix.append(list)
    
    for player in player_matrix:
        for salary in salary_matrix:
            if match_player_salary(player,salary)==True:
                player.append(salary[3])
                for item in player:
                    total.write("%s\t" %item)
                total.write('\n')

def extract_data(year,output):
    rawdata = open(year + "/" + year + ".txt")
    for lines in rawdata:
        list = lines.rstrip('\n\t').split('\t')
        for idx in range(3,len(list)):
            if idx != len(list)-1:
                output.write("%s," %list[idx])
            else:
                output.write("%s" %list[idx])
        output.write("\n")

def gen_data():
    #output training data
    training = open("training_data.txt","w")
    testing = open("testing_data.txt","w")
    
    extract_data("2013",training)
    extract_data("2014",training)
    extract_data("2015",training)
    extract_data("2016",testing)
    
    training.close()
    testing.close()
    
    
    