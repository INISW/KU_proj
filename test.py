from platform_image_captioning_inference_service_sub import video_tracking, IM, multi_image_caption
from platform_image_captioning_preprocess import init_svc
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

result_list = video_tracking(input_type='A', table=None, input_keyword='')
result = pd.DataFrame(result_list)

table = result.drop_duplicates(subset='object_id', keep='first')
table = table.reset_index(drop=True)

print("*" * 70)
print(table)
print("*" * 70)

# 비디오에서 사람 객체 이미지 크롭 및 SR 진행
# params: 사용할 모델(blip), preprocessor 및 전처리된 images 반환
im = IM()
params = init_svc(im, table) # {'model_blip': model_blip, 'processor_blip': processor_blip, 'cropped_images':images}

# 캡션 생성 및 table에 객체 id별 캡션 넣어주기
captions = multi_image_caption(params)
table['caption'] = captions

print("*" * 70)
print(table)
print("*" * 70)

# input_keyword = [""] 키워드 0개
# input_keyword = ["woman"] 키워드 1개
input_keyword=["man", "black"]  # 키워드 2개
result_list = video_tracking(input_type='B', table=table, input_keyword=["woman", "black"])
print("keyword: ", input_keyword)
print(result_list)
print("*" * 70)