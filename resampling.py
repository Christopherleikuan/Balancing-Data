# coding=UTF-8
"""
SDA数据集对比试验
通过SDA这篇文献，最其中的数据集进行rs处理然后再与论文中的数据作比较
对数据集的处理流程：
1.对数据进行预处理（比如去掉模块名）
2.通过cv.py对文件中的数据进行cv划分
3.然后生成两个文件，个是训练集，一个而是测试集
4.文件命名格式：文件名_版本号_bug比例数_测试集/训练集_预处理方法（add，del，smote，morph）_学习器（rf，updata，无监督）
5.然后将训练集导入学习器中，用测试集合来测试结果
6.学习器可在Rstudio上使用
"""
from numpy import *
import random
import operator
import string
import csv

_metaclass_ = type  #


class resampling():
    """

    """

    def __init__(self, rate):
        self.rate = rate
        self.new_bug_list_morph = []
        self.new_bug_list_add = []
        self.new_bug_list_del = []
        return

    def get_deances(self, inputs1, inputs2):
        arr1 = inputs1
        arr2 = inputs2
        arr_difference = arr1 - arr2
        arr_difference **= 2
        arr_difference = arr_difference.sum()
        arr_difference **= 0.5
        return arr_difference

    def made_bug(self, bug_inputs_vector, all_inputs_vector):
        str_list = []
        bug_vector = bug_inputs_vector[random.randrange(0, len(bug_inputs_vector))]
        arr_deances_list = []
        for i in range(len(all_inputs_vector)):
            deances = self.get_deances(bug_vector, all_inputs_vector[i])
            arr_deances_list.append(deances)
            # for i in range(len(arr_deances_list)):
            # print  arr_deances_list[i]
        # print min(arr_deances_list)
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

    def make_bug_rate_number(self, all_list, bug_list, bug_rate):
        self.make_bug_num = (bug_rate * len(all_list) - len(bug_list)) + 1

    def MORPH(self, bug_rate, all_list, bug_list):
        """
        通过给定的参数列表生，生成新的bug
        :param bug_rate: 需要生成的bug比例数
        :param all_list: 这个是文件的所有数据list，包含了bug与非bug
        :param bug_list: 这个是文件中的所有的bug的list
        :return: 一个含有新bug数据的列表,self.new_bug_list_morph
        """
        self.make_bug_rate_number(all_list, bug_list, bug_rate)
        print"should make bug number is :", self.make_bug_num
        count = 1
        new_bug_list = []
        while count <= self.make_bug_num:
            new_bug = self.made_bug(bug_list, all_list)

            # bug_list.append(new_bug)
            new_bug_list.append(new_bug)
            count += 1
        self.new_bug_list_morph = new_bug_list

    def ADD(self, bug_rate, all_list, bug_list):
        self.make_bug_rate_number(all_list, bug_list, bug_rate)
        count = 1
        new_bug_list = []
        while count <= self.make_bug_num:
            bug_random_num = random.randint(0, len(bug_list) - 1)
            new_bug = bug_list[bug_random_num]
            new_bug_list.append(new_bug)
            count += 1
        self.new_bug_list_add = new_bug_list

    def DEL(self, bug_rate, unbug_list, bug_list):
        """
        跟add算法不同，这个是删减正例的，所以只要将文件中的bug数除以
        bug_rate就可以知道要生成的中数据，减去原来的bug数就是要生成
        正例数了
        :param bug_rate:bug比例
        :param unbug_list:在file模块中，self.unbyg_list中有全是正例的数据
        :param bug_list:全是负例的数据
        :return:self.new_bug_list_del 表示生成的新的正例集合，与bug_list
        合并就是新的数据
        """
        count = 1
        new_unbug_list = []
        new_file_num = len(bug_list) / bug_rate
        unbug_num = new_file_num - len(bug_list)
        while count <= unbug_num:
            unbug_random = random.randint(0, len(unbug_list) - 1)
            new_unbug = unbug_list[unbug_random]
            new_unbug_list.append(new_unbug)
            count += 1
        self.new_bug_list_del = new_unbug_list


class File(resampling):
    def __init__(self, name, read_file_address):
        self.train_list = []
        self.test_list = []
        self.train_bug_list = []
        self.train_unbug_list = []
        self.name = name
        self.read_file_address = read_file_address
        return

    def chang_array(self, inputs):
        # print "str:",inputs

        a1 = inputs
        arr = []
        j = 0
        for i in range((len(a1))):
            if i == len(a1) - 2:
                """
                看文件中的数据格式，每一行字符串末尾两位都是换行符，所以不显示出来
                然后有的数据末尾还有一个逗号，那就是len（a1）-4，如果不是那就是
                len（a1）-3，这个根据不同的文件有不同的改变
                """
                arr.append(int(a1[i]))
            if a1[i] == ',':
                if j == 0:
                    arr.append(float(a1[j:i]))
                    j = i + 1
                else:

                    arr.append(float(a1[j:i]))
                    j = i + 1
        return array(arr)

    def read_csv(self):  # 读入给定的csv文件,做预处理，生成指定的数据结构
        """
        读取文件生成相应的数据结构
        :param self.read_file_address: 文件的地址
        :return: self.list :就是文件中所有信息
                 self。bug_list :文件中的所有bug信息
        """
        all_data_inputs = []
        f = open(self.read_file_address, 'a+')
        sss = f.readline()
        all_data_inputs.append(sss)  # 把文件的模块名装入list中
        bug_inputs = []  # 装入有bug的行数，是str列表
        bug_inputs_vector = []
        all_inputs = []  # 装入每一行，是str形的列表
        all_inputs_vector = []
        line_num = 1
        line_num_1 = 1
        count_bug = 0
        count = 0
        for line in f:
            all_data_inputs.append(line)

            if line_num_1 == 1:
                line_num_1 += 1
            else:

                line_num_1 += 1
                # print line

            if line_num == 1:
                line_num += 1
            else:
                str1 = line
                i = len(str1) - 2
                '''
                    同上面的def chang_array(inputs)函数一样，要看文件的末尾是什么结束的
                    有的在bug信息后面加了“，”所以要多两位这个参数要随着文件的改变而改变
                '''

                count += 1
                if str1[i] != '0':
                    # print str1
                    # print char1
                    count_bug += 1
                    bug_inputs.append(str1)
                    arr = self.chang_array(line)
                    # print '1111', arr
                    bug_inputs_vector.append(arr)
                else:
                    all_inputs.append(line)
                    arr = self.chang_array(line)
                    all_inputs_vector.append(arr)
        self.unbug_list = all_inputs_vector
        self.list = all_inputs_vector + bug_inputs_vector
        self.bug_list = bug_inputs_vector
        return

    def csv_cv(self, cv_rate):  # 将文件进行cv划分
        """
        对类中的实例进行数据划分
        :param cv_rate: 将文件划分的比例，一般是随机对半分
        :return:
        """
        for vec in range(len(self.list)):
            ran = random.uniform(0, 1)
            if ran >= cv_rate:
                self.test_list.append(self.list[vec])
            else:
                self.train_list.append(self.list[vec])
        '''由于resampling需要将bug部分分离出来，故在此操作'''
        for j in self.train_list:
            if j[len(j) - 1] != 0:
                self.train_bug_list.append(j)
            else:
                self.train_unbug_list.append(j)
        return

    def check_file_bug_num(self, file_list):
        count_bug = 0
        for j in file_list:
            if j[len(j) - 1] != 0:
                count_bug += 1
        print count_bug

    def csv_cross_version(self):  # 对文件做跨版本处理
        return

    def csv_cross_project(self):  # 对文件做跨项目处理

        return

    def csv_resampling_MORPH(self, bug_rate, cv_or_cross):
        """
        通过输入bug的比例数来生成行的bug数
        :param bug_rate: bug比例数
        :cv_or_cross: 输入cv就是做cv的，输出cross就是做跨版本的，用的list不同
        :return: 在resampling类中引用了函数self.MORPH函数中定义了一个
        self.new_bug_list_morph对象数据来装载生成的新的bug。
        所以最后的输出是self.new_bug_list_morph + self.list
        """
        if cv_or_cross == 'cv':
            self.MORPH(bug_rate, self.train_list, self.train_bug_list)
        else:
            self.MORPH(bug_rate, self.list, self.bug_list)

    def csv_resampling_add(self, bug_rate, cv_or_cross):
        """
        输入bug比例数和控制选项来生成相对应的bug
        :param bug_rate: bug比例数
        :param cv_or_cross: 输入cv就是做cv的，输出cross就是做跨版本的，用的list不同
        :return: 在resampling类中引用了函数self.add函数中定义了一个
        self.new_bug_list_add对象数据来装载生成的新的bug。
        所以最后的输出是self.new_bug_list_add + self.list
        """
        if cv_or_cross == 'cv':
            self.ADD(bug_rate, self.train_list, self.train_bug_list)
        else:
            self.ADD(bug_rate, self.list, self.bug_list)

    def csv_resampling_del(self, bug_rate, cv_or_cross):
        if cv_or_cross == 'cv':
            self.DEL(bug_rate, self.train_unbug_list, self.train_bug_list)
        else:
            self.DEL(bug_rate, self.unbug_list, self.bug_list)

    def csv_out_file(self, file_name, out_list):
        """
        输出文件函数，用于
        :file_name: 通过本程序文档中的文件命名法建立名字
        :out_list: 要输出的列表
        :return:
        """
        save_file = file_name
        """
        w = open(save_file, 'a')
        for j in range(len(out_list)):
            print out_list[j]
            w.write(str(out_list[j]))
        """
        csvfile = file(save_file, 'wb')
        writer = csv.writer(csvfile)
        # writer.writerow(['loc', 'woc']) # 如果文件要添加属性名就用用这方法
        for j in range(len(out_list)):
            writer.writerow(out_list[j])
        return

    def print_csv(self):

        """检测读文件功能"""
        print "all list num:", len(self.list)
        print "check all bug list:", self.check_file_bug_num(self.list)
        print "bug list num:", len(self.bug_list)

        """检测cv功能"""
        print "train list:", len(self.train_list)
        print "train list bug num:", self.check_file_bug_num(self.train_list)
        print "test list:", len(self.test_list)
        print "test list bug num:", self.check_file_bug_num(self.test_list)

        """检测resampling功能"""
        print "new_bug_list_morph :", len(self.new_bug_list_morph)
        print "write_list :", len(self.new_bug_list_morph) + len(self.list)
        print "new_bug_list_add :", len(self.new_bug_list_add)
        print "write_list :", len(self.new_bug_list_add) + len(self.list)
        print "new_bug_list_del :", len(self.new_bug_list_del)
        print "write_list :", len(self.new_bug_list_del) + len(self.bug_list)
        print "bug make num :", self.make_bug_num


"""
jdt = File("jed")
jdt.read_csv('C:/Users/Chris/Desktop/JDT.csv')
list_1 = jdt.list
list_2 = jdt.bug_list
jdt.csv_cv(0.5)
print list_1[2]
print len(jdt.list)
print len(jdt.train_list)
print len(jdt.test_list)
print jdt.name
"""

'''
list_str = ['C:/Users/Chris/Desktop/JDT.csv', 'C:/Users/Chris/Desktop/LC.csv']
for li in list_str:
    f = File(str(li))
    f.read_csv(li)
    f.csv_cv(0.5)
    print len(f.list)
    print len(f.train_list)
    print len(f.test_list)
    print f.name
'''

'''
f = File("jdt", 'C:/Users/Chris/Desktop/JDT_TF.csv')
f.read_csv()
print len(f.list[4])
print f.list[6]
# f.csv_cv(0.5)

# f.csv_resampling_MORPH(0.2, 'cv')
file_name_morph = 'C:/Users/Chris/Desktop/JDT_1.0_20_test_morph.csv'
# f.csv_out_file(file_name_morph, f.new_bug_list_morph + f.train_list)
# print f.new_bug_list_morph + f.list

# f.csv_resampling_add(0.2)
# f.csv_resampling_del(0.2)
#f.print_csv()


'''

import glob
file_list = {'nasa'}
for vec in file_list:
    file_add = 'C:/Users/Chris/Desktop/' + vec + "/*.csv"

    for filename in glob.glob(r"%s" % file_add):
        # print filename  # filename:C:/Users/Chris/Desktop/aeeem\JDT.csv
        # filename可以直接当成文件的读入路径使用
        file_add_org = filename[0:len(filename) - 4]  # C:/Users/Chris/Desktop/aeeem\JDT
        for j in range(len(file_add_org)):
            if file_add_org[j] == '\\':
                file_name = file_add_org[j + 1:]
        # print file_name

        '''生成存放训练集合路径'''
        import os
        file_add_train = file_add_org + '_' + 'train'
        os.mkdir(r'%s' % file_add_train)  # 生成训练集文件夹

        file_add_train_org = file_add_train + '/' + 'org'
        os.mkdir(r'%s' % file_add_train_org)  # 生成训练集原始文件夹

        file_add_train_add = file_add_train + '/' + 'add'
        os.mkdir(r'%s' % file_add_train_add)  # 生成训练集add文件夹

        file_add_train_del = file_add_train + '/' + 'del'
        os.mkdir(r'%s' % file_add_train_del)  # 生成训练集del文件夹

        file_add_train_morph = file_add_train + '/' + 'morph'
        os.mkdir(r'%s' % file_add_train_morph)  # 生成训练集morph文件夹

        file_add_train_smote = file_add_train + '/' + 'smote'
        os.mkdir(r'%s' % file_add_train_smote)  # 生成训练集原始文件夹
        '''生成测试集路径'''
        import os
        file_add_test = file_add_org + '_' + 'test'
        os.mkdir(r'%s' % file_add_test)
        '''开始批量生成csv文件'''
        '''做20次cv，然后bug比例是20%'''

        for i in range(1, 21):
            print file_name, i
            f = File('this', filename)
            f.read_csv()
            f.csv_cv(0.5)

            '''morph方法'''
            f.csv_resampling_MORPH(0.2, 'cv')
            file_train_name_morph = file_name + '_1.0_20_train_morph_' + "%d" % i + ".csv"
            file_train_name_morph_adder = file_add_train_morph + '/' + file_train_name_morph
            # print file_train_name_morph_adder
            f.csv_out_file(file_train_name_morph_adder, f.new_bug_list_morph + f.train_list)

            '''add方法'''
            f.csv_resampling_add(0.2, 'cv')
            file_train_name_add = file_name + '_1.0_20_train_add_' + "%d" % i + ".csv"
            file_train_name_add_adder = file_add_train_add + '/' + file_train_name_add
            f.csv_out_file(file_train_name_add_adder, f.new_bug_list_add + f.train_list)

            '''del方法'''
            f.csv_resampling_del(0.2, 'cv')
            file_train_name_del = file_name + '_1.0_20_train_del_' + "%d" % i + ".csv"
            file_train_name_del_adder = file_add_train_del + '/' + file_train_name_del
            f.csv_out_file(file_train_name_del_adder, f.new_bug_list_del + f.train_bug_list)

            '''org写入'''
            file_train_name_org = file_name + '_1.0_20_train_org_' + "%d" % i + ".csv"
            file_train_name_org_adder = file_add_train_org + '/' + file_train_name_org
            f.csv_out_file(file_train_name_org_adder, f.train_list)

            '''test写入'''
            file_test_name_org = file_name + '_1.0_20_test_org_' + "%d" % i + ".csv"
            file_test_name_org_adder = file_add_test + '/' + file_test_name_org
            f.csv_out_file(file_test_name_org_adder, f.test_list)































