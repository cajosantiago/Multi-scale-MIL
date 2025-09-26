FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
	git wget && \
	git clone https://github.com/cajosantiago/Multi-scale-MIL.git /home && \
    cd /home && \
    pip install -r docker/requirements.txt && \
    wget -P checkpoints 'https://huggingface.co/shawn24/Mammo-CLIP/resolve/main/Pre-trained-checkpoints/b2-model-best-epoch-10.tar?download=true'

WORKDIR /home