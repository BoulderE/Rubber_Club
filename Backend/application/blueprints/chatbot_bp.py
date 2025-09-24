import requests
import os
from flask import Blueprint, request, jsonify
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("HKBU_API_KEY")
base_url = "https://genai.hkbu.edu.hk/general/rest"
model_name = "gpt-4.1"
api_version = "2024-12-01-preview"

chatbot_bp = Blueprint('chatbot_bp', __name__)

def get_school_api_response(user_message, conversation_history):
    if not api_key:
        print("严重错误: 环境变量 HKBU_API_KEY 未设置！")
        return "抱歉，AI助手未配置，请联系管理员。"
    
    system_prompt = """
    You are a professional fitness coach AI assistant. Your task is to help users select the coaching style that best suits them.
    There are two styles:
    1. The Motivator: Suitable for users with no exercise background or those new to this type of tech product.
    2. The Guide: Suited for users prioritizing safety, preferring steady guidance, and seeking clear feedback. This is the default style.
    Your conversation strategy is: Use open-ended questions to understand user preferences, 
    then naturally recommend a style. Responses should be concise, friendly, and professional.
    """
    messages = [{"role": "system", "content": system_prompt}]
    for msg in conversation_history:
         messages.append({"role": msg['role'], "content": msg['content']})
    messages.append({"role": "user", "content": user_message})

    url = f"{base_url}/deployments/{model_name}/chat/completions?api-version={api_version}"

    headers = {
        # "accept": "application/json",
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    payload = {"messages": messages, "temperature": 0.7, "max_tokens": 150, "top_p": 1, "stream": False}
    response = requests.post(url, json=payload, headers=headers)

    try:
        # 发送 POST 请求
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # 如果请求失败 (如 4xx 或 5xx 错误), 会抛出异常

        response_data = response.json()
        
        bot_reply = response_data['choices'][0]['message']['content']
        return bot_reply

    except requests.exceptions.RequestException as e:
        print(f"API 请求失败: {e}")
        return "抱歉，连接学校的AI助手时网络出现问题，请稍后再试。"
    except (KeyError, IndexError) as e:
        print(f"解析API响应失败: {e}")
        print(f"收到的数据: {response.json()}")
        return "抱歉，无法理解AI助手的回答格式，我需要检查一下代码。"
    
    
@chatbot_bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Message is required'}), 400

    user_message = data.get('message')
    conversation_history = data.get('history', []) 

    bot_response = get_school_api_response(user_message, conversation_history)

    return jsonify({'reply': bot_response})