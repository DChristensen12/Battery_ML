#!/usr/bin/env python
"""Run code on a Colab GPU from this repo, over a cloudflare tunnel.

Colab side: paste tools/colab_tunnel_cell.py into a cell, it prints a url and a
token. Here: attach once per session, then run/push. Quick tunnels get a new url
every time the runtime restarts, so attaching is not a one off.
"""

import argparse
import base64
import json
import pathlib
import sys
import uuid

import requests
import websocket

CONFIG = pathlib.Path(__file__).resolve().parent.parent / ".colab_runtime.json"


def load_config():
    if not CONFIG.exists():
        sys.exit("nothing attached, run `colab_runtime.py attach <url> <token>` first")
    return json.loads(CONFIG.read_text())


def save_config(cfg):
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n")


def api(cfg, method, path, **kwargs):
    resp = requests.request(
        method,
        f"{cfg['url'].rstrip('/')}/api/{path.lstrip('/')}",
        headers={"Authorization": f"token {cfg['token']}"},
        timeout=kwargs.pop("timeout", 30),
        **kwargs,
    )
    resp.raise_for_status()
    return resp


def ensure_kernel(cfg):
    """Reuse the stored kernel if it's still alive, otherwise start a fresh one."""
    live = {k["id"] for k in api(cfg, "GET", "kernels").json()}
    if cfg.get("kernel") in live:
        return cfg["kernel"]
    kernel = api(cfg, "POST", "kernels", json={"name": "python3"}).json()["id"]
    cfg["kernel"] = kernel
    save_config(cfg)
    return kernel


def execute(cfg, code):
    """Run code on the remote kernel, streaming output. Returns a shell exit status."""
    kernel = ensure_kernel(cfg)
    ws = cfg["url"].replace("https://", "wss://").replace("http://", "ws://")
    conn = websocket.create_connection(
        f"{ws.rstrip('/')}/api/kernels/{kernel}/channels",
        header=[f"Authorization: token {cfg['token']}"],
        timeout=30,
    )

    msg_id = uuid.uuid4().hex
    conn.send(json.dumps({
        "header": {
            "msg_id": msg_id,
            "username": "colab_runtime",
            "session": uuid.uuid4().hex,
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "channel": "shell",
        "content": {
            "code": code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
    }))

    status = 0
    try:
        while True:
            msg = json.loads(conn.recv())
            # the kernel is shared, so ignore traffic from anyone else's request
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            kind = msg["msg_type"]
            content = msg["content"]

            if kind == "stream":
                out = sys.stderr if content["name"] == "stderr" else sys.stdout
                out.write(content["text"])
                out.flush()
            elif kind in ("execute_result", "display_data"):
                text = content.get("data", {}).get("text/plain")
                if text:
                    print(text)
            elif kind == "error":
                print("\n".join(content["traceback"]), file=sys.stderr)
                status = 1
            elif kind == "status" and content["execution_state"] == "idle":
                break
    except KeyboardInterrupt:
        api(cfg, "POST", f"kernels/{kernel}/interrupt")
        print("\ninterrupted the remote kernel", file=sys.stderr)
        status = 130
    finally:
        conn.close()
    return status


def cmd_attach(args):
    # the cell prints a bare url but it's easy to paste the ?token= form instead
    url = args.url.rstrip("/").removesuffix("/?token=" + args.token)
    cfg = {"url": url, "token": args.token}
    kernels = api(cfg, "GET", "kernels").json()
    save_config(cfg)
    print(f"attached to {url}, {len(kernels)} kernel(s) already up")
    return execute(cfg, "import torch, os\n"
                        "print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')\n"
                        "print('drive:', os.path.isdir('/content/drive'))")


def cmd_status(args):
    cfg = load_config()
    kernels = api(cfg, "GET", "kernels").json()
    print(cfg["url"])
    for k in kernels:
        here = " <- current" if k["id"] == cfg.get("kernel") else ""
        print(f"  {k['id']}  {k['execution_state']}{here}")
    return 0


def cmd_run(args):
    cfg = load_config()
    if args.code:
        code = args.code
    elif args.file:
        code = pathlib.Path(args.file).read_text()
    else:
        code = sys.stdin.read()
    return execute(cfg, code)


def cmd_push(args):
    cfg = load_config()
    for local in args.paths:
        path = pathlib.Path(local)
        name = args.name or path.name
        remote = f"{args.dest.rstrip('/')}/{name}" if args.dest else name
        api(cfg, "PUT", f"contents/{remote}", json={
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(path.read_bytes()).decode(),
        })
        print(f"{path} -> /content/{remote}")
    return 0


def cmd_kernel(args):
    cfg = load_config()
    kernel = ensure_kernel(cfg)
    if args.restart:
        api(cfg, "POST", f"kernels/{kernel}/restart")
        print(f"restarted {kernel}")
    elif args.interrupt:
        api(cfg, "POST", f"kernels/{kernel}/interrupt")
        print(f"interrupted {kernel}")
    else:
        print(kernel)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("attach", help="save the url and token the Colab cell printed")
    p.add_argument("url")
    p.add_argument("token")
    p.set_defaults(func=cmd_attach)

    p = sub.add_parser("status", help="show what's attached and which kernels are up")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("run", help="execute code on the Colab GPU")
    p.add_argument("file", nargs="?", help="local .py file, or stdin if omitted")
    p.add_argument("-c", "--code", help="inline code")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("push", help="upload files into /content on the Colab VM")
    p.add_argument("paths", nargs="+")
    p.add_argument("-d", "--dest", default="", help="subdirectory under /content")
    p.add_argument("-n", "--name", help="rename on the way over, single file only")
    p.set_defaults(func=cmd_push)

    p = sub.add_parser("kernel", help="print, restart, or interrupt the kernel")
    p.add_argument("--restart", action="store_true")
    p.add_argument("--interrupt", action="store_true")
    p.set_defaults(func=cmd_kernel)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
