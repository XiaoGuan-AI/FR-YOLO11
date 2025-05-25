import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# 调用的YOLO库，在 timm 的最新版本中，models.layers 模块已经被弃用

from ultralytics import YOLO

if __name__ == '__main__':
    # 定义模型路径和源数据路径
    model_path = 'runs/detect/train/train3/weights/best.pt'
    source_path = 'data/test/images'

    # 创建 YOLO 实例
    model = YOLO(model_path)

    # 开始预测
    model.predict(source=source_path,
                  project='runs/detect/predict/',
                  name='predict',
                  save=True,
                  save_txt=True
                  )


