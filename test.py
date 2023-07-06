from platform_image_captioning_inference_service_sub import video_tracking
import pandas as pd

result_list = video_tracking(input_type='A')
result = pd.DataFrame(result_list)

table = result.drop_duplicates(subset='object_id', keep='first')
table = table.reset_index(drop=True)



print("*" * 70)
print(table)
print("*" * 70)