# 파일명: image_classification_train_sub.py

class TM:
    param_info = {}
    def __init__(self):
        self.train_data_path = './meta_data/image'
        self.model_path = './meta_data'
        self.label_path = './meta_data/annotations'
        self.output_path = './mata_data/output'
        # 사용자 param 사용시 입력
        self.param_info['batch_size'] = 16
        self.param_info['epoch'] = 30
        self.param_info['learning_rate'] = 1e-6
        self.param_info['optimizer'] = 0

tm = TM
import logging
from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import json
import base64
import numpy as np

def exec_train(tm):
    import torch
    import json
    from transformers import BlipProcessor, Swin2SRImageProcessor, Swin2SRForImageSuperResolution
    from torch.utils.data import DataLoader

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
            path = dir+'/'+image_name
            imagelist.append(Image.open(path))
        return imagelist

    def super_reso(image,pro_sr,model_sr):
        import numpy as np
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


    ####################################################
    # 본격 시작 
    ###################################################
    ## 1. 데이터셋 준비(Data Setup)
    with open(tm.label_path+'/shuffled_captions.json','r',encoding='utf-8' or 'cp949') as f: # caption 불러오기
        captions = json.load(f)
    
    logging.info('[hunmin log] :caption load ok')
    imagelist = image_list(tm.train_data_path,captions) # train_data_path로 불러오기 

    ## 2. 데이터 전처리
    pro_sr = Swin2SRImageProcessor.from_pretrained(tm.model_path+'/super_resol/preprocessor')
    model_sr = Swin2SRForImageSuperResolution.from_pretrained(tm.model_path+'/super_resol/swinsr.pt')
    model_sr.to(device)
    images = []
    for image in imagelist:
        image = super_reso(image,pro_sr,model_sr) if image.size[0]<50 or image.size[1]<100 else image
        images.append(image)
    model_sr.to('cpu')

    data = [{'text':captions[i]['label'],'image':images[i]} for i in range(len(images))] # 최종 학습을 위한 데이터셋
    train_dataset = ImageCaptioningDataset(data[:int(0.8*25000)], processor)
    val_dataset = ImageCaptioningDataset(data[int(0.2):], processor)


    model = torch.load(tm.model_path+'init.pt') # 모델 불러오기
    processor = BlipProcessor.from_pretrained(tm.preprocessor_path)
    batch_size = int(tm.param_info['batch_size'])
    epochs = int(tm.param_info['epoch'])
    lr = float(tm.param_info['learning_rate'])
    val_cpath = tm.caption_path

    train_dataloader = DataLoader(train_dataset,shuffle=False,batch_size = batch_size)
    val_dataloader = DataLoader(val_dataset,shuffle=False,batch_size = batch_size)

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
            torch.save(model,tm.model_path+'/model.pt')

        #Epoch 출력
        logging.info("Epoch {}회차 - val_Loss:{}, ".format(epoch+1,val/313))

        gen_captions(val_caption,val_cpath+'/'+str(epoch+1)+'.json')
        #train_eval.append(coco_caption_eval(train_rpath,train_cpath+'/'+str(epoch+1)+'.json').eval.items())
        scheduler.step()

    history = [train_hist,val_hist]    
    # 학습 결과 표출 함수화 일단 하지 말자. 
    plot_metrics(tm, history, model,val_dataloader) # epoch별로 보여줄 예정 

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
    ##########################################################################
    result=0
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
def plot_metrics(tm, history, model,val_dataloader, processor):
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    def coco_caption_eval(annotation_file, results_file):

        coco = COCO(annotation_file)
        coco_result = coco.loadRes(results_file)

        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.evaluate()

        return coco_eval
    
    acc = []
    y_test = tm.label_path+'/'+'gt.json'
    for epoch in range(len(tm.param_info['epoch'])):
        y_predict = tm.label_path+'/'+str(epoch+1)+'.json'
        acc.append(coco_caption_eval(y_test,y_predict).eval['METEOR'])
     
    for epoch, (acc,loss) in enumerate(zip(acc,history[0])):
        metric={}
        metric['accuracy'] = acc[epoch]
        metric['loss'] = loss
        metric['step'] = epoch
        tm.save_stat_metrics(metric)   

    # 최종모델에 대한 성능보기 
    val = 0
    val_caption =[]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        model.eval()
        for idx, batch in enumerate(val_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)
            outputs = model(input_ids=input_ids,pixel_values=pixel_values, labels=input_ids)

            #성능을 보기위한 작업
            val_caption+=processor.batch_decode(model.generate(pixel_values=pixel_values,max_length = 300),skip_special_tokens=True)
            val+=outputs.loss.tolist()
    
    # 이건 할지 안할지 고민 중 => confusion matrix를 사용할 수 없음 
    eval_results={}
    eval_results['accuracy'] = coco_caption_eval(y_test,y_predict).eval['METEOR']
    eval_results['loss'] = val/len(val_dataloader)
    tm.save_result_metrics(eval_results)
    logging.info('[user log] accuracy and loss curve plot for platform')