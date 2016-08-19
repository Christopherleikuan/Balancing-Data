# coding=UTF-8
from numpy import  *
import random
import operator
import string
bug_rate = 0.20
read_file = "C:/Users/Chris/Desktop/7.lang3.0.1_all.csv"
save_file = "C:/Users/Chris/Desktop/7.lang3.0.1_all.csv"
def chang_array(inputs):
   # print "str:",inputs

    a1 = inputs
    arr = []
    j=0
    for i in range((len(a1))):
        if i == len(a1)-2:
         arr.append(int(a1[i]))
        if a1[i] == ',':
           if j == 0:
            j = i + 1
           else:

               arr.append(float(a1[j:i]))
               j = i + 1
    return array(arr)
'''开始计算欧式距离'''
def get_deances(inputs1,inputs2):
    arr1 = inputs1
    arr2 = inputs2
    arr_difference = arr1 - arr2
    arr_difference = arr_difference ** 2
    arr_difference = arr_difference.sum()
    arr_difference = arr_difference ** 0.5
    return arr_difference
def made_bug(bug_inputs_vector,all_inputs_vector):
     str_list = []
     bug_vector = bug_inputs_vector[random.randrange(0,len(bug_inputs_vector))]
     arr_deances_list = []
     for i in range(len(all_inputs_vector)):
         deances = get_deances(bug_vector,all_inputs_vector[i])
         arr_deances_list.append(deances)
    # for i in range(len(arr_deances_list)):
         #print  arr_deances_list[i]
     #print min(arr_deances_list)
     for i in range(len(arr_deances_list)):
         if arr_deances_list[i] == min(arr_deances_list):
             min_index = i
     '''至此实现了与选中的的bug样本欧式距离最小的正立样本
        下面开始实现算法进行新的样本的改造
        根据公式：Yi = Xi +- (Xi - Zi) * r
        r取0.15
     '''
     different_vector = bug_vector - all_inputs_vector[min_index]
     new_vector = bug_vector + different_vector * 0.15
     new_vector = new_vector ** 2
     new_vector = new_vector ** 0.5

     return new_vector
def MORPH_function(bug_rate,read_file,save_file):
    f = open(read_file, 'a+')
    bug_inputs = []  # 装入有bug的行数，是str列表
    bug_inputs_vector = []
    all_inputs = []  # 装入每一行，是str形的列表
    all_inputs_vector = []
    str1 = [500]
    line_num = 1
    line_num_1 = 1
    count_bug = 0
    count = 0
    inputs_2 = []
    for line in f:
        if line_num_1 == 1:
            line_num_1 = line_num_1 + 1
        else:

            line_num_1 = line_num_1 + 1
            # print line

        if line_num == 1:
            line_num = line_num + 1
        else:
            str1 = line
            i = len(str1) - 2

            count = count + 1
            if str1[i] != '0':
                char1 = str1[i]
                #print char1
                j = int('1')
                count_bug = count_bug + 1
                bug_inputs.append(str1)
                arr = chang_array(line)
                bug_inputs_vector.append(arr)
            else:
                all_inputs.append(line)
                arr = chang_array(line)
                all_inputs_vector.append(arr)
    new_bug_list = []
  #  print round(float(len(bug_inputs_vector)) / len(all_inputs_vector), 2)
    while round(float(len(bug_inputs_vector)) / len(all_inputs_vector), 2) != bug_rate:
        new_bug = made_bug(bug_inputs_vector, all_inputs_vector)

        bug_inputs_vector.append(new_bug)
        new_bug_list.append(new_bug)
     #   print len(bug_inputs_vector)
     #   print len(all_inputs_vector)
      #  print round(float(len(bug_inputs_vector)) / len(all_inputs_vector), 2)

    for i in range(len(new_bug_list)):
        str_list = []
        for j in range(len(new_bug_list[i])):
            str_list.append(new_bug_list[i][j])
        for k in range(0, len(str_list)):
            str_list[k] = round(str_list[k], 2)
        new_str = str(str_list)
     #   print type(new_str)
        out_bug_str = new_str[1:len(new_str) - 1]
        bug_name_str = "lang\WordUtils.java\org.apache.commons.lang.WordUtils,"
        new_bug_1 = bug_name_str + out_bug_str + '\n'
        w = open(save_file, 'a+')
        w.write(new_bug_1)
        #print new_bug_1
        w.close()
    return


#MORPH_function(0.2,read_file,save_file)
