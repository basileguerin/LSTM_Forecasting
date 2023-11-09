FROM continuumio/miniconda3

WORKDIR /home/app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD streamlit run app.py