FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# 必须安装官方的 gaussian rasterizer
RUN cd EGSTalker/submodules/custom-gaussian-rasterization && python setup.py install
RUN cd EGSTalker/submodules/simple-knn && python setup.py install

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]