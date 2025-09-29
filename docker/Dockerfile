#FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
#FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
	git wget && \
	git clone https://github.com/cajosantiago/Multi-scale-MIL.git /home && \
    cd /home && \
    pip install -r docker/requirements.txt && \
    mkdir checkpoints && cd checkpoints && \
    wget 'https://huggingface.co/shawn24/Mammo-CLIP/resolve/main/Pre-trained-checkpoints/b2-model-best-epoch-10.tar?download=true' -O 'b2-model-best-epoch-10.tar' && \
    gdown 'https://drive.google.com/uc?id=1pcr5wa8cI7R8L-7MfkXBEBB2IE02NmMI'

WORKDIR /home