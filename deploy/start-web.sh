#!/bin/sh
set -eu

pnpm run start -- --host 127.0.0.1 --port 3001 &
web_pid=$!
nginx -g 'daemon off;' &
nginx_pid=$!

shutdown() {
    kill "$web_pid" "$nginx_pid" 2>/dev/null || true
    wait "$web_pid" "$nginx_pid" 2>/dev/null || true
}

trap shutdown INT TERM EXIT

while kill -0 "$web_pid" 2>/dev/null && kill -0 "$nginx_pid" 2>/dev/null; do
    sleep 1
done

exit 1
