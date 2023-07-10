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
    # 1. 이미지 데이터
    my_zip_path_1 = os.path.join(pm.source_path,'image_augmented.zip') 
    extract_zip_file_1 = zipfile.ZipFile(my_zip_path_1)
    extract_zip_file_1.extractall(os.path.join(pm.target_path, 'image_dataset'))
    extract_zip_file_1.close()

    # 2. 캡션(라벨) 데이터
    my_zip_path_2 = os.path.join(pm.source_path, 'annotations.zip') 
    extract_zip_file_2 = zipfile.ZipFile(my_zip_path_2)    
    extract_zip_file_2.extractall(os.path.join(pm.target_path, 'annotations'))
    extract_zip_file_2.close()

    # 3. preprocessor
    my_zip_path_3 = os.path.join(pm.source_path, 'preprocessor.zip') # 전처리 수행 모듈
    extract_zip_file_3 = zipfile.ZipFile(my_zip_path_3)    
    extract_zip_file_3.extractall(os.path.join(pm.target_path, 'preprocessor'))
    extract_zip_file_3.close()

    
    # 저장 파일 확인
    list_files_directories(pm.target_path)

    logging.info('[hunmin log]  the finish line of the function [exec_process]')



# 저장 파일 확인
def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[hunmin log] Files and directories in {} :'.format(path))
    logging.info('[hunmin log] dir_list : {}'.format(dir_list))


