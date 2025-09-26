FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
	git && \
	git clone https://github.com/marianamourao-37/Multi-scale-Attention-based-MIL.git /home

WORKDIR /home

CMD ['python run_inference.py']