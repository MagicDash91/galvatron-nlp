FROM python:3.9
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY . .
EXPOSE 8201
ENTRYPOINT ["streamlit", "run"]
CMD ["app/Home.py"]
