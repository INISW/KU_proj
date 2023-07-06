# 파일명: image_captioning_preprocess.py

from platform_image_captioning_preprocess_sub import exec_process

from transformers import Swin2SRImageProcessor, BlipProcessor
import numpy as np
import os
import cv2
import pandas as pd
from PIL import Image
import torch

import logging




logging.basicConfig(level=logging.INFO)

# 학습 전 데이터 전처리
def process_for_train(pm):
    # pm : 플랫폼 사용에 필요한 객체 - 사용법은 exec_process에 있다.
    
    # 데이터 전처리 알고리즘 호출 - sub파일의 함수 사용
    exec_process(pm)
    
    # return : return값은 없으나 전처리 완료된 데이터를 반드시 pm.target_path에 저장하여
    #          학습 모델의 train함수의 tm.target_path로 불러와야 학습을 할 수 있다.
    


'''
-----------------------------------만들어야 하는 함수 하나---------------------------------------------

'''

# 데이텨 변환시 메모리에 standby 시켜놓을 데이터 반환
# 1. 추론할 데이터의 폴더에서 데이터를 꺼내 추론에 필요한 형태로 메모리에 올려두는 역할
# 2. 추론에 쓰일 모델 메모리에 올려두는 역할

def init_svc(im, data):
    # im : 플랫폼 사용에 필요한 객체
    # rule : 전처리 규칙정보 -> 생략
    # im.meta_path : process_for_train의 pm.meta_path경로를 호출하여 학습때 사용한 전처리 모듈 및 메타 데이터를 불러온다.
    
    # 사용할 모델 및 preprocessor 불러오기
    processor_blip = BlipProcessor.from_pretrained(im.preprocessor_path)
    model_blip = torch.load(os.path.join(im.model_path, 'blip_all_1e6.pt'), map_location=torch.device('cpu'))
    pro_sr = Swin2SRImageProcessor.from_pretrained(os.path.join(im.sr_path, 'preprocessor'))
    model_sr = torch.load(os.path.join(im.sr_path, 'swinsr.pt'), map_location=torch.device('cpu'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_sr.to(device)

    # 비디오에서 이미지 크롭한 후 메모리에 스탠바이
    video = "./yolov4_deepsort/data/video/demo4.mp4"
    cap =cv2.VideoCapture(video)
    images = []
    for _, info in data.iterrows(): # 이미지 전체 다 가져오기
        image = Image.fromarray(cropping(cap, info['frame_id'], [info['x1'],info['y1'],info['x2'],info['y2']]))
        # 작은 이미지의 경우 SR 적용 
        image = super_resolution(image,model_sr,pro_sr,device) if image.size[0]<50 or image.size[1]<100 else image 
        images.append(image)

    # return : dict형태로 반환하며 이는 transform의 params가 된다.
    # ex) init_svc(im, rule): return {'meta_path' : im.meta_path}
    #     transform(df, params, batch_id): meta_path = params['meta_path']
    return {'model_blip': model_blip, 'processor_blip': processor_blip,'cropped_images':images}


def cropping(cap, frame_id, box):
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id) # 이미지 불러오기
    T, image = cap.read()
    image = image[box[1]:box[3],box[0]:box[2]] if T else print("error")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if T else None


def super_resolution(image,model_sr,pro_sr,device):
    inputs = pro_sr(image, return_tensors="pt").to(device)

    # forward pass
    with torch.no_grad():
        outputs = model_sr(**inputs)

    output = outputs.reconstruction.data.squeeze().cpu().float().clamp_(0, 1).numpy()
    output = np.moveaxis(output, source=0, destination=-1)
    output = (output * 255.0).round().astype(np.uint8)
    return Image.fromarray(output)

'''
---------------------------------만들어야 하는 함수 둘------------------------------------------
'''



# 추론 이전 데이터 변환
# 위 inin_svc에서 올려둔 데이터를 super_resol 및 preprocessor 
def transform(df, params, batch_id):
    # df => pd.DataFrame : 학습한 모델로 추론할 때 입력한 데이터
    # params => Dict : init_svc의 반환값
    # batch_id : 사용하지 않음
    
    # meta_path = params['meta_path'] : init_svc(im, rule)의 반환값중 'meta_path'를 불러온다.
    # rule = params['rule'] : 전처리 규칙 불러오기
    # use_cols = rule['source_column'] : 전처리 규칙중 사용한 컬럼명
    # inner_df = df.loc[:, use_cols]
    
    # return : df 형태가 적절하며 이는 학습 모델에서 inference 함수의 df가 된다.
    return df