# 步骤 1: 选择一个官方的、轻量的 Python 基础镜像
FROM python:3.9-slim

# 步骤 2 (最关键的一步): 安装 mediapipe 运行所需的底层系统依赖库
# Buildpack 环境缺的就是这些东西！
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 步骤 3: 在容器内创建一个工作目录
WORKDIR /app

# 步骤 4: 复制依赖文件，并安装所有 Python 包
# 因为上一步安装了系统库，这里的 pip install mediapipe 就会成功
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 步骤 5: 将你所有的后端代码复制到容器的工作目录中
COPY . .

# 步骤 6: 定义容器启动时要执行的命令
# Gunicorn 是生产级的 Python Web 服务器运行器
# 请确保 "app:app" 和你的文件名及Flask/FastAPI实例名一致
# 例如，如果你的主文件是 main.py，实例是 my_app，就写 "main:my_app"
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "app:app"]