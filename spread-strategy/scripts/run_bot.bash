symbol="ldousdt"
touch "$(pwd)/$symbol-log.txt"
chown ubuntu "$(pwd)/$symbol-log.txt"
docker run -i -t --env-file .env -e CONFIG="./configs/$symbol.ini" --rm --name "spreadbot-$symbol" -v "$(pwd)/$symbol-log.txt":/app/log.txt -v "$(pwd)/configs":"/app/configs/" spreadbot