# 파일명: image_classification_train_sub.py

# 사용할 gpu 번호를 적는다.
# os.environ["CUDA_VISIBLE_DEVICES"]='0'

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_visible_devices(gpus, 'GPU')
#         logging.info('[hunmin log] gpu set complete')
#         logging.info('[hunmin log] num of gpu: {}'.format(len(gpus)))
    
#     except RuntimeError as e:
#         logging.info('[hunmin log] gpu set failed')
#         logging.info(e)


import logging
from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import json

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text=item["text"], padding="max_length", return_tensors="pt")

        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding
def image_list(dir, captions):
    imagelist = []
    for item in captions:
        image_name = item['image']
        path = os.path.join(dir, image_name)
        imagelist.append(Image.open(path))
    return imagelist
def super_reso(image,pro_sr,model_sr):
    import numpy as np
    inputs = pro_sr(image, return_tensors="pt").to(device)

    # forward pass
    with torch.no_grad():
        outputs = model_sr(**inputs)

    output = outputs.reconstruction.data.squeeze().cpu().float().clamp_(0, 1).numpy()
    output = np.moveaxis(output, source=0, destination=-1)
    output = (output * 255.0).round().astype(np.uint8)
    return Image.fromarray(output)     
def gen_captions(captions,filename):
    gen = []
    for i in range(len(captions)):
        gen.append({'image_id': i+1, 'caption': captions[i]})

    with open(filename,'w') as f:
      json.dump(gen,f)



def exec_train(tm):
    import torch
    import json
    from transformers import BlipProcessor, Swin2SRImageProcessor, Swin2SRForImageSuperResolution
    from torch.utils.data import DataLoader

    ## 1. 데이터셋 준비(Data Setup)
    with open(tm.label_path,'r',encoding='utf-8' or 'cp949') as f: # caption 불러오기
        captions = json.load(f)   
    logging.info('[hunmin log] :caption load ok')
    imagelist = image_list(tm.train_data_path,captions)
    ## 2. 데이터 전처리
    pro_sr = Swin2SRImageProcessor.from_pretrained("caidas/swin2SR-lightweight-x2-64")
    model_sr = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-lightweight-x2-64")
    model_sr.to(device)
    images = []
    for image in imagelist:
        image = super_reso(image,pro_sr,model_sr) if image.size[0]<50 or image.size[1]<100 else image
        images.append(image)
    model_sr.to('cpu')

    data = [{'text':captions[i]['label'],'image':images[i]} for i in range(len(images))] # 최종 학습을 위한 데이터셋
    train_dataset = ImageCaptioningDataset(data[:int(0.8*25000)], processor)
    val_dataset = ImageCaptioningDataset(data[int(0.2):], processor)
    train_dataloader = DataLoader(train_dataset,shuffle=False,batch_size = option['batch_size'])
    val_dataloader = DataLoader(val_dataset,shuffle=False,batch_size = option['batch_size'])

    model = torch.load(tm.model_path) # 모델 불러오기
    processor = BlipProcessor.from_pretrained(tm.preprocessor_path)
    batch_size = int(tm.param_info['batch_size'])
    epochs = int(tm.param_info['epoch'])
    lr = float(tm.param_info['learning_rate'])

    #train 진행
    train_hist = []
    val_hist = []

    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for epoch in range(epochs):
        model.train()
        Loss = 0

        for idx, batch in enumerate(train_dataloader):
            model.train()
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)
            outputs = model(input_ids=input_ids,pixel_values=pixel_values, labels=input_ids)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            Loss+=loss.tolist()
        train_hist.append(Loss/len(train_dataloader))


        #validation진행
        val = 0
        val_caption =[]
        with torch.no_grad():
            model.eval()
            for idx, batch in enumerate(val_dataloader):
                input_ids = batch.pop("input_ids").to(device)
                pixel_values = batch.pop("pixel_values").to(device)
                outputs = model(input_ids=input_ids,pixel_values=pixel_values, labels=input_ids)

                #성능을 보기위한 작업
                val_caption+=processor.batch_decode(model.generate(pixel_values=pixel_values,max_length = 300),skip_special_tokens=True)
                val+=outputs.loss.tolist()

        val_hist.append(val/len(val_dataloader))

        #checkpoint
        if val_hist[-1]==min(val_hist):
            torch.save(model,'/content/drive/MyDrive/mjae/model_zoo/blip_all_1e6_final.pt')

        #Epoch 출력
        logging.info("Epoch {}회차 - val_Loss:{}, ".format(epoch+1,val/313))

        gen_captions(val_caption,val_cpath+'/'+str(epoch+1)+'.json')
        #train_eval.append(coco_caption_eval(train_rpath,train_cpath+'/'+str(epoch+1)+'.json').eval.items())
        scheduler.step()

    
    # 학습 결과 표출 함수화
    plot_metrics(tm, history, model, X_test, Y_test)
    torch.save(model,os.path.join(tm.model_path, 'model.h5'))
    


# def exec_init_svc(im):

#     logging.info('[hunmin log] im.model_path : {}'.format(im.model_path))
    
#     # 저장 파일 확인
#     list_files_directories(im.model_path)
    
#     ###########################################################################
#     ## 학습 모델 준비
#     ########################################################################### 
    
#     # load the model
#     model = load_model(os.path.join(im.model_path, 'cnn_model.h5'))
    
#     return {'model' : model}



def exec_inference(df, params, batch_id):
    
    ###########################################################################
    ## 4. 추론(Inference)
    ###########################################################################
    
    logging.info('[hunmin log] the start line of the function [exec_inference]')
    
    ## 학습 모델 준비
    model = params['model']
    logging.info('[hunmin log] model.summary() :')
    model.summary(print_fn=logging.info)
    
    dataset=['ant','apple', 'bus', 'butterfly', 'cup', 'envelope','fish', 'giraffe', 'lightbulb','pig']
    
    # image preprocess
    img_base64 = df.iloc[0, 0]
    image_bytes = io.BytesIO(base64.b64decode(img_base64))
    image = Image.open(image_bytes).convert('L')
    image = image.resize((28, 28))
    image = np.invert(image).astype('float32')/255.
    image = image.reshape(-1, 28, 28 , 1)
    
    # data predict
    y_pred = model.predict(image)
    y_pred_idx=np.argmax(y_pred, axis=1)
    
    # inverse transform
    result = {'inference' : dataset[y_pred_idx[0]]}
    logging.info('[hunmin log] result : {}'.format(result))

    return result



# 저장 파일 확인
def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[hunmin log] Files and directories in {} :'.format(path))
    logging.info('[hunmin log] dir_list : {}'.format(dir_list))



###########################################################################
## exec_train(tm) 호출 함수 
###########################################################################
    
# 시각화
def plot_metrics(tm, history, model, x_test, y_test):
    from sklearn.metrics import confusion_matrix
    
    accuracy_list = history.history['accuracy']
    loss_list = history.history['loss']
    
    for step, (acc, loss) in enumerate(zip(accuracy_list, loss_list)):
        metric={}
        metric['accuracy'] = acc
        metric['loss'] = loss
        metric['step'] = step
        tm.save_stat_metrics(metric)

    predict_y = np.argmax(model.predict(x_test), axis = 1).tolist()
    actual_y = np.argmax(y_test, axis = 1).tolist()
    
    eval_results={}
    eval_results['predict_y'] = predict_y
    eval_results['actual_y'] = actual_y
    eval_results['accuracy'] = history.history['val_accuracy'][-1]
    eval_results['loss'] = history.history['val_loss'][-1]

    # calculate_confusion_matrix(eval_results)
    eval_results['confusion_matrix'] = confusion_matrix(actual_y, predict_y).tolist()
    tm.save_result_metrics(eval_results)
    logging.info('[hunmin log] accuracy and loss curve plot for platform')
    