import json
import os

# 读取源文件
with open(r'E:\yolo\yolov8v10\runs\val\exp\predictions.json', 'r') as file:
    data = json.load(file)

# 对每个字典进行处理
for item in data:
    # 如果字典中包含 'image_id' 键
    if 'image_id' in item:
        # 将 'image_id' 值转换为6位数形式，并在前面用0填充
        item['image_id'] = str(item['image_id']).zfill(6)

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建新文件的路径
new_file_path = os.path.join(current_dir, 'new_pre_file.json')

# 将修改后的内容写入到新文件中
with open(new_file_path, 'w') as file:
    json.dump(data, file, indent=4)