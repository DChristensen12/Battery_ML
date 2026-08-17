# Paste this into a Colab cell and run it. Starts a jupyter server on the Colab
# VM and puts a cloudflare tunnel in front of it so your laptop can attach.
# Set the runtime to GPU first. If you need Drive, mount it in an earlier cell,
# that needs the browser popup and won't work from the other side of the tunnel.

import os
import re
import secrets
import subprocess
import sys
import threading
import time

PORT = 8888
TOKEN = os.environ.get("COLAB_JUPYTER_TOKEN") or secrets.token_urlsafe(24)

if not os.path.exists("/usr/local/bin/cloudflared"):
    subprocess.run(
        "wget -q -O /usr/local/bin/cloudflared "
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        "cloudflared-linux-amd64 && chmod +x /usr/local/bin/cloudflared",
        shell=True,
        check=True,
    )
subprocess.run(
    [sys.executable, "-m", "pip", "-q", "install", "jupyter_server", "ipykernel"],
    check=True,
)

jupyter = subprocess.Popen(
    [
        sys.executable, "-m", "jupyter", "server",
        "--no-browser",
        f"--port={PORT}",
        "--ip=0.0.0.0",
        f"--IdentityProvider.token={TOKEN}",
        "--ServerApp.allow_origin=*",
        "--ServerApp.allow_remote_access=True",
        # non-browser clients have no xsrf cookie, the token is the auth here
        "--ServerApp.disable_check_xsrf=True",
        "--ServerApp.root_dir=/content",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

tunnel = subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel", "--no-autoupdate",
     "--url", f"http://localhost:{PORT}"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
)

url = None
deadline = time.time() + 90
while time.time() < deadline:
    line = tunnel.stderr.readline()
    if not line:
        if tunnel.poll() is not None:
            break
        continue
    m = re.search(r"https://[-\w]+\.trycloudflare\.com", line)
    if m:
        url = m.group(0)
        break

if not url:
    raise RuntimeError("cloudflared never printed a url, it probably died")

# keep draining stderr, otherwise the pipe buffer fills and the tunnel stalls
threading.Thread(target=lambda: [None for _ in tunnel.stderr], daemon=True).start()

print(url)
print(TOKEN)
print()
print(f"python tools/colab_runtime.py attach {url} {TOKEN}")
print()
print("leave this tab open, the tunnel goes down with it")

import torch
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
