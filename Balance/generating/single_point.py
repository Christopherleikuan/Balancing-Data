# coding=UTF-8
from numpy import  *
import random
import operator
import string
def single_point(vector1,vector2):
    leng = len(vector1)
    random_point = random.randint(int(leng * 0.3),int(leng * 0.6))
    print random_point
    new_bug_1 = []
    new_bug_2 = []
    new_bug_point = []
    array(new_bug_1)
    array(new_bug_2)
    new_bug_1 = vector1[0:random_point]
    print new_bug_1
    new_bug_2 = vector2[random_point+1: leng]
    print new_bug_2

    new_bug_1 = list(new_bug_1)
    new_bug_2 = list(new_bug_2)


    new_bug_point= new_bug_1+new_bug_2
    print new_bug_point
    array(new_bug_point)

    return new_bug_point
