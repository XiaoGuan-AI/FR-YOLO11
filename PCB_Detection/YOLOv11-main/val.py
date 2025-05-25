import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# 调用的YOLO库，在 timm 的最新版本中，models.layers 模块已经被弃用

from ultralytics import YOLO
# 模型配置文件
model_path = 'runs/detect/train/train/weights/best.pt'
# 数据集配置文件
data_yaml_path = 'data/data.yaml'

if __name__ == '__main__':
    model = YOLO(model_path)
    model.val(data=data_yaml_path,
              split='val',
              imgsz=640,
              batch=8,
              project='runs/detect/val/',
              name='val',
              save_json=True,
              )