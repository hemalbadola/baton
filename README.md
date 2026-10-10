# Baton

Run one large language model across several laptops on the same Wi-Fi. Each laptop holds
a few layers of the model and passes activations to the next laptop, like a relay baton.

## Install

On every laptop, one line. A page opens in the browser.

```
# macOS or Linux
curl -LsSf https://raw.githubusercontent.com/hemalbadola/baton/main/install.sh | sh -s -- app

# Windows (PowerShell)
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/hemalbadola/baton/main/install.ps1))) app

# any system with Python 3.11 or newer
pip install baton-cluster
baton app

# macOS or Linux with Homebrew
brew install hemalbadola/baton/baton
baton app
```

## Use

1. Run `baton app` on every laptop. Each laptop shows the others that are nearby.
2. On one laptop, pick a model and click **Host this model**. Invite the nearby laptops.
3. On each other laptop, click **Accept**.
4. On the host, click **Start**. Chat when the cluster is ready.

The API is OpenAI compatible: `http://<host>:7700/v1`. The first start downloads the model.
Only bf16 weights load today (`--quant none`).

## Commands

```
baton app                                 # the page above
baton serve --model Qwen/Qwen2.5-0.5B-Instruct --quant none --min-workers 2
baton worker                              # joins a head found over mDNS
```

## Develop

```
pip install -e ".[dev]"
pytest
```

Design: see `TICKETS.md`. Tests need the sandbox off for sockets and mDNS.
