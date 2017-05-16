
from cv_head import File
from cv_head import supervised
from cv_head import resampling
from cv_head import Control


"""
    以下代码是对单个小的文件夹进行操作，例如nasa，aeeem等
"""

C = Control()
C.file_analysis(Multilayer_folder=False,Multilayer_folder_adder='null',
                folder_name='nasa-uniformed')
C.cv_and_learn_control(10, 0.20)
C.cv_ang_learn_oog_data_control(10, 0.20)

f = File('this', 'null')
f.csv_out_file('C:/Users/Chris/Desktop/nasa_result.csv', C.all_result)
