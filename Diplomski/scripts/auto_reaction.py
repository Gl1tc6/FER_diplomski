import subprocess
import json
import time
import threading
import os
import sys

if os.geteuid() != 0:
    sys.exit("Pokreni kao root (sudo).")

LOG_FILE = "zeek-logs/notice.log"
TARGET_NOTICES = {
    "SSH::Password_Guessing",
    "Custom_Port_Scan",
    "Custom_SSH_Bruteforce",   # bilo "CustomSSH::Bruteforce" -> nije se podudaralo
    "Intel::Notice",
}
WHITELIST = {"127.0.0.1", "10.0.1.1"}
BLOCKED_IPS = {}

def read_timeout():
    """Timeout seta iz zive nft konfiguracije (sekunde)."""
    try:
        out = subprocess.run(
            ["nft", "-j", "list", "set", "ip", "diplomski", "denylist"],
            capture_output=True, text=True, check=True
        ).stdout
        for item in json.loads(out)["nftables"]:
            if "set" in item and "timeout" in item["set"]:
                return int(item["set"]["timeout"])
    except Exception:
        pass
    return 60

BLOCK_DURATION = read_timeout()

def unblock_expired():
    while True:
        time.sleep(5)
        now = time.time()
        for ip in list(BLOCKED_IPS.keys()):
            if now - BLOCKED_IPS[ip] > BLOCK_DURATION:
                del BLOCKED_IPS[ip]
                print(f"[+] Istekla evidencija blokade: {ip}")

def block(ip):
    res = subprocess.run(
        ["nft", "add", "element", "ip", "diplomski", "denylist", "{ %s }" % ip],
        capture_output=True, text=True
    )
    if res.returncode == 0:
        BLOCKED_IPS[ip] = time.time()
        print(f"[!] Blokiran IP: {ip}")
    else:
        print(f"[x] nft nije blokirao {ip}: {res.stderr.strip()}")

def process_log():
    print(f"[*] Nadzor pokrenut nad {LOG_FILE}")
    while not os.path.exists(LOG_FILE):
        time.sleep(1)
    proc = subprocess.Popen(["tail", "-F", "-n", "0", LOG_FILE],
                            stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        try:
            entry = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        note = entry.get("note", "")
        msg = entry.get("msg", "")
        src_ip = entry.get("src") or entry.get("id.orig_h", "")   # custom notice -> "src"
        if note in TARGET_NOTICES and src_ip and src_ip not in WHITELIST:
            if src_ip not in BLOCKED_IPS:
                print(f"[!] Detektiran {note}.")
                print(f"[Info] {msg}.")
                block(src_ip)

if __name__ == "__main__":
    threading.Thread(target=unblock_expired, daemon=True).start()
    process_log()