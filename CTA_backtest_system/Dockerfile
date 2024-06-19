FROM python:3.8.10

# Set the working directory.
WORKDIR /app

# Copy the file from your host to your current location.
COPY requirements2.txt .

# https://stackoverflow.com/questions/68673221/warning-running-pip-as-the-root-user
ENV PIP_ROOT_USER_ACTION=ignore 

RUN python -m pip install --upgrade pip

RUN python -m pip install -r requirements2.txt

# Copy the rest of your app's source code from your host to your image filesystem.
COPY . .

RUN cd backtest_system && python3 backtest_engine/setup_core.py build_ext
