FROM python:3.9

RUN useradd -ms /bin/bash user
USER user

WORKDIR /app

COPY pluto/utils /app/pluto/utils
COPY pluto/requirements.txt /app
COPY main.py /app

RUN pip install -r requirements.txt

ENV SYMBOL "ETH-USDT"
ENV PLATFORM "bingx"
ENV OUTPUT "./data"

CMD ["sh", "-c", "python main.py --symbol ${SYMBOL} --platform ${PLATFORM} --output ${OUTPUT}"]
