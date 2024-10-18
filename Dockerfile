FROM nvcr.io/nvidia/pytorch:24.07-py3

RUN apt update
RUN apt install -y python3.10-venv
RUN apt install -y libgl1
#when in the venv
#RUN pip install polygraphy
#RUN pip install cuda-python
