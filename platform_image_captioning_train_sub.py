# 파일명: image_classification_train_sub.py

import logging
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import torch
import json
import numpy as np
from PIL import Image
import base64
import io
from transformers import BlipProcessor, BlipForConditionalGeneration, Swin2SRImageProcessor, Swin2SRForImageSuperResolution

def exec_train(tm):
    
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
    with open(os.path.join(tm.train_data_path, 'annotations/shuffled_captions_2.json'),'r',encoding='utf-8' or 'cp949') as f: # caption 불러오기
        captions = json.load(f)

    logging.info('[hunmin log] :caption load ok')
    imagelist = image_list(os.path.join(tm.train_data_path, 'image_dataset/image_augmented_2'),captions) # train_data_path로 불러오기 
    
    ## 2. 데이터 전처리
    pro_sr = Swin2SRImageProcessor.from_pretrained("caidas/swin2SR-lightweight-x2-64")
    model_sr = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-lightweight-x2-64")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_sr.to(device)
    images = []
    for image in imagelist:
        image = super_reso(image,pro_sr,model_sr) if image.size[0]<50 or image.size[1]<100 else image
        images.append(image)
    model_sr.to('cpu')

    data = [{'text':captions[i]['label'],'image':images[i]} for i in range(len(images))] # 최종 학습을 위한 데이터셋
    processor = BlipProcessor.from_pretrained(os.path.join(tm.train_data_path, 'preprocessor/preprocessor'))

    train_dataset = ImageCaptioningDataset(data[:int(0.8*100)], processor)
    val_dataset = ImageCaptioningDataset(data[int(0.8*100):], processor)


    # 모델 불러오기
    mode = "Salesforce/blip-image-captioning-base"
    model = BlipForConditionalGeneration.from_pretrained(mode)
    batch_size = int(tm.param_info['batch_size'])
    epochs = int(tm.param_info['epoch'])
    lr = float(tm.param_info['learning_rate'])
    val_cpath = tm.model_path

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
    #plot_metrics(tm, history, model,val_dataloader, preprocessor) # epoch별로 보여줄 예정 



def exec_init_svc(im):
    from transformers import BlipProcessor
    logging.info('[hunmin log] im.model_path : {}'.format(im.model_path))
    
    # 저장 파일 확인
    list_files_directories(im.model_path)
    
    ###########################################################################
    ## 학습 모델 준비
    ########################################################################### 
    
    # load the model
    mode = "Salesforce/blip-image-captioning-base"
    processor_blip = BlipProcessor.from_pretrained(mode)
    model_blip = torch.load(os.path.join(im.model_path, 'model.pt'), map_location=torch.device('cpu'))
    
    return {'model' : model_blip, 'processor': processor_blip}


def super_reso(image,pro_sr,model_sr):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    inputs = pro_sr(image, return_tensors="pt").to(device)

    # forward pass
    with torch.no_grad():
        outputs = model_sr(**inputs)

    output = outputs.reconstruction.data.squeeze().cpu().float().clamp_(0, 1).numpy()
    output = np.moveaxis(output, source=0, destination=-1)
    output = (output * 255.0).round().astype(np.uint8)
    return Image.fromarray(output)   

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


###########################################################################
## exec_inference(df, params, batch_id)함수, 하위 호출 함수 
###########################################################################

import json, os, cv2, time, sys
import pandas as pd
from absl import flags
from absl.flags import FLAGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("len(physical_devices): ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import yolov4_deepsort.core.utils as utils
from yolov4_deepsort.core.yolov4 import filter_boxes
from yolov4_deepsort.core.config import cfg
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from yolov4_deepsort.deep_sort import preprocessing, nn_matching
from yolov4_deepsort.deep_sort.detection import Detection
from yolov4_deepsort.deep_sort.tracker import Tracker
from yolov4_deepsort.tools import generate_detections as gdet

import torch

def exec_inference(df, params, batch_id):

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
    pro_sr = Swin2SRImageProcessor.from_pretrained("caidas/swin2SR-lightweight-x2-64")
    model_sr = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-lightweight-x2-64")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_sr.to(device)

    video = df
    cap =cv2.VideoCapture(video)
    images = []
    for _, info in table.iterrows(): # 이미지 전체 다 가져오기
        image = Image.fromarray(cropping(cap, info['frame_id'], [info['x1'],info['y1'],info['x2'],info['y2']]))
        # 작은 이미지의 경우 SR 적용 
        image = super_reso(image,model_sr,pro_sr,device) if image.size[0]<50 or image.size[1]<100 else image 
        images.append(image)

    # 캡션 생성 및 table에 객체 id별 캡션 넣어주기
    captions = multi_image_caption(params, images)
    table['caption'] = captions

    print("*" * 70)
    print(table)
    print("*" * 70)

    input_keyword = [""]  # 키워드 0개
    # input_keyword = ["woman"]  # 키워드 1개
    # input_keyword=["man", "black"]  # 키워드 2개
    result_list = video_tracking(input_type='B', table=table, input_keyword=input_keyword)
    print("keyword: ", input_keyword)
    print(result_list)
    print("*" * 70)    

    return result_list



def draw_box(table, input_keyword):
    table_len = len(table)

    print('-'*70)
    print('input_keyword: ', input_keyword)
    try:
        selected_row = []

        if (len(input_keyword) == 0):
            for i in range(table_len):
                row = table.iloc[i]
                selected_row.append(row)
        else:
            for i in range(table_len):
                row = table.iloc[i]
                object_cap = row['caption']
                matched_keywords = [True for keyword in input_keyword if keyword in object_cap]
                if sum(matched_keywords) == len(input_keyword):
                    selected_row.append(row)
                    print("-"*70)
                    print("keyword를 포함하는 객체 id")
                    print(row['object_id'])
                    print("해당 캡션 문장")
                    print(row['caption'])
    except:
        print('draw_box Error')

    return selected_row


def video_tracking(input_type, table, input_keyword):
    for name in list(flags.FLAGS):
        delattr(flags.FLAGS, name)

    # db_handler = DBHandler()
    # video_id = int(video_id)
    # video_name = video_name['video_name']
    # output_video_name = str(video_name[:(len(video_name)-4)])+'_output.mp4'

    flags.DEFINE_string('framework', 'tf', 'tf, tflite, trt')
    flags.DEFINE_string('weights', './yolov4_deepsort/checkpoints/yolov4-tiny-416','path to weights file')
    flags.DEFINE_integer('size', 416, 'resize images to')
    flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
    flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
    flags.DEFINE_string('output_format', 'H264', 'codec used in VideoWriter when saving video to file')
    flags.DEFINE_float('iou', 0.45, 'iou threshold')
    flags.DEFINE_float('score', 0.50, 'score threshold')
    flags.DEFINE_boolean('dont_show', False, 'dont show video output')
    flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
    flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
    flags.DEFINE_string('video', './yolov4_deepsort/data/video/demo4.mp4', 'path to input video or set to 0 for webcam')
    flags.DEFINE_string('output', 'demo4_test.mp4', 'path to output video')

    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    # model_filename = 'C:\\Users\\Sihyun\\Desktop\\INISW\\project\\sihyun_track_test\\yolov4-deepsort-master\\model_data\\mars-small128.pb'
    model_filename = './yolov4_deepsort/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        infer = keras.models.load_model(FLAGS.weights)

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    total_frame = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0

    result_list = []

    if input_type == 'B':
        draw_box_res = draw_box(table, input_keyword)
        if len(draw_box_res) == 0:
            return json.dumps({'status': 'fail', 'message': 'No object to track'})
        else:
            print("## 실제로 Tracking이 되야 하는 객체 ID 리스트 ##")
            
            draw_box_res = pd.DataFrame(draw_box_res)
            data_rs = draw_box_res.drop_duplicates(subset='object_id',keep='first') 

            result_list = data_rs[['object_id', 'caption']]

            result_obj_id = data_rs['object_id'].tolist()
            print("result_obj_id: ", result_obj_id)
            print("-" * 70)
    
    # while video is running
    # result_list = []
    cnt = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            session.close()
            break
        frame_num +=1
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer.predict(batch_data)

            for value in pred_bbox:
                temp_value = np.expand_dims(value, axis=0)
                boxes = temp_value[:, :, 0:4]
                pred_conf = temp_value[:, :, 4:]

                
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            if input_type == 'A':
                fps = vid.get(cv2.CAP_PROP_FPS)

                time_in_seconds = frame_num / fps

                minutes = int(time_in_seconds // 60)
                seconds = int(time_in_seconds % 60)

                bbox_0 = int(bbox[0]) if int(bbox[0]) > 0 else 0
                bbox_1 = int(bbox[1]) if int(bbox[1]) > 0 else 0
                bbox_2 = int(bbox[2]) if int(bbox[2]) > 0 else 0
                bbox_3 = int(bbox[3]) if int(bbox[3]) > 0 else 0
                
                print("*"*70)
                print("객체 좌표값")
                print(bbox_0, bbox_1, bbox_2, bbox_3)
                # insert_result = db_handler.insert_video_info(frame_num-1, track.track_id, bbox_0, bbox_1, bbox_2, bbox_3, video_id, minutes, seconds)
                result_list.append({"frame_id": frame_num-1, "object_id": track.track_id, "x1": bbox_0, "y1": bbox_1, "x2": bbox_2, "y2": bbox_3, "minutes": minutes, "seconds": seconds})
                cnt += 1

                print("DB Input Result: ", result_list)
                # print("DB Input Result: ", insert_result)
                print("*"*70)

            elif input_type =='B':
                if (int(track.track_id) in result_obj_id):
                    print(int(track.track_id) in result_obj_id)

                    color = colors[int(track.track_id) % len(colors)]
                    color = [i * 255 for i in color]
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                else:
                    continue

        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if FLAGS.output and input_type == 'B':
            print("*"*70)
            print("Tracking이 되어야 하는 Object만 영상에 표시")
            print("*"*70)
            out.write(result)

    return result_list


# 캡션 생성 함수
def multi_image_caption(params, images): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = params["model"].to(device)
    model.eval()
    vis_processors = params["processor"].image_processor
    decode = params["processor"].batch_decode

    captions = []
    for image in images:
        captions.append(single_image_caption(image,model,vis_processors,decode,device)) # 이미지 captioning
    return captions

def single_image_caption(image, model, vis_processors, decode, device):
    generated_ids = model.generate(pixel_values= vis_processors(images=image, return_tensors="pt").to(device).pixel_values,max_length=300)
    caption= decode(generated_ids, skip_special_tokens=True)[0]
    return caption

def cropping(cap, frame_id, box):

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id) # 이미지 불러오기
    T, image = cap.read()
    image = image[box[1]:box[3],box[0]:box[2]] if T else print("error")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if T else None