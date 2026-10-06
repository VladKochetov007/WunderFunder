#!/bin/bash
# Refresh IAM token on Mac (browser-login profile) and push it to beleriand for datasphere jobs.
T=$(/Users/nb/yandex-cloud/bin/yc iam create-token --no-user-output 2>/dev/null) || exit 1
[ -n "$T" ] && printf "%s" "$T" | ssh -o ConnectTimeout=30 beleriand 'umask 077; cat > ~/.config/yc_iam_token' && echo "$(date) pushed"
