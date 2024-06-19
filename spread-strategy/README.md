# spread-strategy
Dual exchange spread trade.

## Get start

Set API keys

```bash
cp .env.example .env
```

## Usage

```bash
sudo docker build -t spreadbot .
sudo docker run -i -t --env-file .env -e CONFIG="./configs/xrpusdt.ini" --rm --name spreadbot-xrpusdt spreadbot
```