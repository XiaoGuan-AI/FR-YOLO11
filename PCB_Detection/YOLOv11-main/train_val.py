import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# 调用的YOLO库，在 timm 的最新版本中，models.layers 模块已经被弃用

#model_path = r"D:\StudyRoom\Medical_Imaging\CTB_and_CS\YOLOv11-main\ultralytics\cfg\models\addv11\yolov11n_C3k2_CMUNeXtBlock.yaml"
# YAML starts training from scratch, while PT uses the trained weight model for further training
# Generally, YAML is used for training, and PT obtained from training is used for validation and prediction
from ultralytics import YOLO

if __name__ == '__main__':
    # 定义训练数据路径和其他参数
    data_yaml_path = 'data/data.yaml'
    #model_path = r'D:\StudyRoom\PCB_Detection\YOLOv11-main\ultralytics\cfg\models\addv11\yolov11n.yaml'
    # model_path = 'best.pt'
    epochs = 10
    batch_size = 4
    img_size = 640
    workers = 0   #Win10:works=0，Win11:works=4
    device = '0'  # 使用第一个 GPU


    # 创建 YOLO 实例并开始训练
    model = YOLO(model_path)

    # 开始训练
    model.train(data=data_yaml_path,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                workers=workers,
                device=device,
                project='runs/detect/train/',
                name='train')