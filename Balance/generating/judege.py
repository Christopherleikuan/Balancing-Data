# coding=UTF-8
from basic_method import MORPH
'''在各种算法子中使用的lable生成函数，son的距离跟父母谁近那么lable就跟谁
    本来想着是直接添加的但是array不支持 + 重载，于是只好直接修改'''
def judge_lable(vector1,vector2,vector_son):
    if MORPH.get_deances(vector1,vector_son) > MORPH.get_deances(vector2,vector_son): #儿子跟vector2姓
        v2_len = len(vector2)
        vson_len = len(vector_son)
        vector_son[vson_len - 1] = vector2[v2_len - 1]
    else:
        v1_len = len(vector1)
        vson_len = len(vector_son)
        vector_son[vson_len - 1] = vector1[v1_len - 1]
    return vector_son

print "judge:\n",judge_lable(MORPH.all_inputs_vector[2],MORPH.all_inputs_vector[3],MORPH.all_inputs_vector[4])
print "len(judge):\n",len(judge_lable(MORPH.all_inputs_vector[2],MORPH.all_inputs_vector[3],MORPH.all_inputs_vector[4]))
