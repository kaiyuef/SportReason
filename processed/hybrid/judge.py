import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/tweet-topic-21-multi")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/tweet-topic-21-multi")

# 定义判断是否与sports相关的函数
def is_sports_related(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits).item()

    # 预测类别与标签
    labels = ["arts_&_culture", "business_&_entrepreneurs", "celebrity_&_pop_culture", "diaries_&_daily_life", "family", 
              "fashion_&_style", "film_tv_&_video", "fitness_&_health", "food_&_dining", "gaming", 
              "learning_&_educational", "music", "news_&_social_concern", "other_hobbies", "relationships", 
              "science_&_technology", "sports", "travel_&_adventure", "youth_&_student_life"]

    # 获取预测的类别名称
    predicted_label = labels[predicted_class_id]

    # 打印预测信息
    print(f"Sentence: '{sentence}' -> Predicted label: {predicted_label}")
    
    # 判断是否为'sports'
    return predicted_label == "sports"

# 读取JSON文件
with open('processed/hybrid/new_reformatted_questions.json', 'r') as f:
    data = json.load(f)

# 使用'seed_question'进行筛选
filtered_data = [item for item in data if is_sports_related(item.get('seed_question', ''))]

# 将筛选后的数据保存为新的JSON文件
with open('filtered_output.json', 'w') as f:
    json.dump(filtered_data, f, indent=4, ensure_ascii=False)

print("筛选完成，已保存与sports相关的数据。")
