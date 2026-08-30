FROM python:3.13

WORKDIR /app

COPY flask_app/ /app/
COPY models/ /app/models/

RUN pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
