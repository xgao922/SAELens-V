import pandas as pd
import os

# 注意：每个样本需要是包含列表的列表
sample_data = {
    # images列：每个元素是一个包含图像路径的列表
    'images': [
        [["/data/wyx/datasets/COCO/train2017/000000000009.jpg"]],
        [["/data/wyx/datasets/COCO/train2017/000000000009.jpg"]],
        [["/data/wyx/datasets/COCO/train2017/000000000009.jpg"]],
        [["/data/wyx/datasets/COCO/train2017/000000000009.jpg"]],
        [["/data/wyx/datasets/COCO/train2017/000000000009.jpg"]]
    ],
    # texts列：每个元素是一个包含文本的列表，与images中的列表一一对应
    'texts': [
        [["A cute orange cat sitting on a windowsill looking outside."]],
        [["A cute orange cat sitting on a windowsill looking outside."]],
        [["A cute orange cat sitting on a windowsill looking outside."]],
        [["A cute orange cat sitting on a windowsill looking outside."]],
        [["A cute orange cat sitting on a windowsill looking outside."]]
    ]
}

# 创建DataFrame
df = pd.DataFrame(sample_data)

# 保存为parquet文件
output_path = "/data/xgao/code/interpretability/SAELens-V/data/test/test.parquet"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_parquet(output_path, index=False)

print("📊 数据集已创建并保存到:", output_path)
print("\nDataFrame内容:")
print(df)
print("\nDataFrame结构:")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# 验证数据结构
images_sample = df['images'].iloc[0]
texts_sample = df['texts'].iloc[0]
print(f"\nImages列第一个元素: {images_sample}")
print(f"Images元素类型: {type(images_sample)}")
print(f"Images元素长度: {len(images_sample)}")
print(f"Images第一个路径: {images_sample[0]}")

print(f"\nTexts列第一个元素: {texts_sample}")
print(f"Texts元素类型: {type(texts_sample)}")
print(f"Texts元素长度: {len(texts_sample)}")
print(f"Texts第一个内容: {texts_sample[0]}")

print(f"\n验证配对关系:")
print(f"图像数量: {len(images_sample)}")
print(f"文本数量: {len(texts_sample)}")
print("配对正确!" if len(images_sample) == len(texts_sample) else "配对错误!")