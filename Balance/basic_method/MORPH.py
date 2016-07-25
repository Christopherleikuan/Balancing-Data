# coding=UTF-8
from numpy import  *
import random
import operator
import string
f = open("C:/Users/Chris/Desktop/1.lang2.2_all.csv",'a+')
bug_inputs = [] #装入有bug的行数，是str列表
bug_inputs_vector = []
all_inputs = [] #装入每一行，是str形的列表
all_inputs_vector = []
str1 = [500]
line_num = 1
line_num_1 = 1
count_bug =0
count = 0
inputs_2 = []
print " [][][\n",type(all_inputs)
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
for line in f :
    if line_num_1 == 1:
        line_num_1 = line_num_1 + 1
    else:

        line_num_1 = line_num_1 + 1
        #print line


    if line_num == 1:
        line_num = line_num + 1
    else:
        str1 = line
        i = len(str1) - 2

        count = count + 1
        if str1[i] != '0':
            char1 = str1[i]
            print char1
            j = int('1')
            count_bug = count_bug + 1
            bug_inputs.append(str1)
            arr = chang_array(line)
            arr = arr[0:len(arr) - 1] #这是为了将每一行数组中的lable去掉
            bug_inputs_vector.append(arr)
        else:
            all_inputs.append(line)
            arr = chang_array(line)
            arr = arr[0:len(arr) - 1] #这是为了将每一行数组中的lable去掉
            all_inputs_vector.append(arr)


print "数据的读入以及转换测试："
print "字符串列表的长度:"
print len(all_inputs)
print len(bug_inputs)
print "输出示例:"
print all_inputs[1]
print bug_inputs[1]
#print all_inputs[1]
print "向量列表长度:"
print len(all_inputs_vector)
print len(bug_inputs_vector)
print "输出示例:"
print all_inputs_vector[1]
print bug_inputs_vector[1]


'''
arr1 = chang_array(inputs[1])
#print "array:",type(arr1[0])
#print "array:",arr1
arr2 = chang_array(inputs[0])
#print "array:",arr2
'''
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
   #  bug_vector = bug_inputs_vector[2]
     arr_deances_list = []
     for i in range(len(all_inputs_vector)):
         deances = get_deances(bug_vector,all_inputs_vector[i])
         arr_deances_list.append(deances)
     for i in range(len(arr_deances_list)):
         print  arr_deances_list[i]
     print min(arr_deances_list)
     for i in range(len(arr_deances_list)):
         if arr_deances_list[i] == min(arr_deances_list):
             min_index = i
     print min_index
     '''至此实现了与选中的的bug样本欧式距离最小的正立样本
        下面开始实现算法进行新的样本的改造
        根据公式：Yi = Xi +- (Xi - Zi) * r
        r取0.15
     '''
     different_vector = bug_vector - all_inputs_vector[min_index]
  #   print different_vector

     new_vector = bug_vector + different_vector * 0.15
     new_vector = new_vector ** 2
     new_vector = new_vector ** 0.5
     print'1111\n', new_vector
     print '222\n', len(new_vector)
     print '333\n', new_vector[0]

     for i in range(len(new_vector)):
         str_list.append(new_vector[i])
        # i = i + 1

     print '+++',str_list
     print "---",type(str_list[1])
     print '~~~',round(str_list[0],2)#这个函数实现了具体小数点之间的转换
     for i in range(0,len(str_list)):
         str_list[i] = round(str_list[i], 2)

     print "***",str_list
     new_str = str(str_list)
 #    new_str = ','.join(str(i) for i in b)
     print type(new_str)
     new_str = new_str[1:len(new_str)-1] #数组转换为字符的时候有中括号，要去掉
     f = open("C:/Users/Chris/Desktop/1.lang2.2_all.csv", 'a') #写入的是没有文件名的新bug信息
     f.write(new_str)
     f.close()
     return new_str
'''

'''

bug_name_str = "lang\WordUtils.java\org.apache.commons.lang.WordUtils，"
new_bug = bug_name_str+out_bug_str;
print new_bug


'''
for i in range(0,6):
    print new_str[i]
    '''
out_bug_list = new_str.split(',')

#w = open("C:/Users/Chris/Desktop/1.lang2.2_all.csv",'a')
#w.write(new_bug)
