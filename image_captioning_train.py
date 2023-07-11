# 파일명: image_classification_train.py

from platform_image_captioning_train_sub import exec_train, exec_inference, exec_init_svc

import logging


def train(tm):
    
    exec_train(tm)
    logging.info('[hunmin log] the end line of the function [train]')


def init_svc(im):
    
    params = exec_init_svc(im)
    logging.info('[hunmin log] the end line of the function [init_svc]')
    
    return { **params }


def inference(df, params, batch_id):
    
    result = exec_inference(df, params, batch_id)
    logging.info('[hunmin log] the end line of the function [inference]')
    
    return { **result }


# if __name__=='__main__':
#     class TM:
#         param_info = {}
#         def __init__(self):
#             self.train_data_path = './meta_data/image'
#             self.model_path = './meta_data/model'
#             self.label_path = './meta_data/annotations'
#             self.output_path = './mata_data/output'
#             # 사용자 param 사용시 입력
#             self.param_info['batch_size'] = 16
#             self.param_info['epoch'] = 30
#             self.param_info['learning_rate'] = 1e-6
#             self.param_info['optimizer'] = 0

#     tm = TM()
#     exec_train(tm)