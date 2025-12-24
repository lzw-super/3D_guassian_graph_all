import json

def extract_unique_categories(json_file_path):
    """
    从给定的JSON文件中提取所有唯一的物体类别
    
    参数:
    json_file_path: JSON文件的路径
    
    返回:
    set: 包含所有唯一物体类别的集合
    """
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 提取所有节点的类别
    categories = set()
    for node in data.get('nodes', []):
        category = node.get('category')
        if category:
            categories.add(category)
    
    return categories
categories = extract_unique_categories('/home/zhengwu/Desktop/3D-Gaussian/office_scene_graph_70000.json') 
print(categories)
print('finish')