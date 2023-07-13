# 파일명: image_captioning_preprocess.py

from platform_image_captioning_preprocess_sub import exec_process
import logging


logging.basicConfig(level=logging.INFO)

# 학습 전 데이터 전처리
def process_for_train(pm):
    # pm : 플랫폼 사용에 필요한 객체 - 사용법은 exec_process에 있다.
    
    # 데이터 전처리 알고리즘 호출 - sub파일의 함수 사용
    exec_process(pm)
    
    # return : return값은 없으나 전처리 완료된 데이터를 반드시 pm.target_path에 저장하여
    #          학습 모델의 train함수의 tm.target_path로 불러와야 학습을 할 수 있다.
    


def init_svc(im, rule):
    
    return {}



def transform(df, params, batch_id):
    
    return df