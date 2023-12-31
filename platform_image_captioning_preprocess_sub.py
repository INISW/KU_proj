# 파일명: image_captioning_preprocess_sub.py

import os
import zipfile
import logging

def exec_process(pm):

    logging.info('[hunmin log]  the start line of the function [exec_process]')

    logging.info('[hunmin log] pm.source_path : {}'.format(pm.source_path))

    # 저장 파일 확인
    list_files_directories(pm.source_path)
    
    # pm.source_path의 dataset.zip 파일을 
    # pm.target_path의 dataset 폴더에 압축을 풀어준다.
    # 이미지, 캡션(라벨), preprocessor zip파일
    my_zip_path = os.path.join(pm.source_path,'dataset.zip') 
    extract_zip_file = zipfile.ZipFile(my_zip_path)
    #extract_zip_file.extractall(os.path.join(pm.target_path, 'image_dataset'))
    extract_zip_file.extractall(pm.target_path)
    extract_zip_file.close()
    
    # 저장 파일 확인
    list_files_directories(pm.target_path)

    logging.info('[hunmin log]  the finish line of the function [exec_process]')



# 저장 파일 확인
def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[hunmin log] Files and directories in {} :'.format(path))
    logging.info('[hunmin log] dir_list : {}'.format(dir_list))


