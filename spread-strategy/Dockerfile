FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt
RUN pip install -r ./pluto/requirements.txt

RUN useradd -ms /bin/bash user
RUN chown -R user /app
USER user

CMD ["sh", "-c", "python main.py --config ${CONFIG}"]