import subprocess
import json
import time
import threading
import os

LOG_FILE = "zeek-logs/notice.log"
BLOCK_DURATION =  60 # 30 sekundi    # 1800  # 30 minuta
BLOCKED_IPS = {}
TARGET_NOTICES = {"Scan::Port_Scan", "SSH::Password_Guessing", "Custom_Port_Scan", "Intel::Notice", "CustomSSH::Bruteforce"}
WHITELIST = {"127.0.0.1", "10.0.1.1"}

def unblock_expired():
    while True:
        time.sleep(60)
        now = time.time()
        for ip in list(BLOCKED_IPS.keys()):
            if now - BLOCKED_IPS[ip] > BLOCK_DURATION:
                subprocess.run(["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"], stderr=subprocess.DEVNULL)
                subprocess.run(["iptables", "-D", "FORWARD", "-s", ip, "-j", "DROP"], stderr=subprocess.DEVNULL)
                del BLOCKED_IPS[ip]
                print(f"[+] Odblokiran IP: {ip}")

def process_log():
    print(f"[*] Nadzor pokrenut nad {LOG_FILE}")
    
    while not os.path.exists(LOG_FILE):
        time.sleep(1)

    proc = subprocess.Popen(
        ["tail", "-F", "-n", "0", LOG_FILE],
        stdout=subprocess.PIPE, text=True
    )
    
    for line in proc.stdout:
        try:
            log_entry = json.loads(line.strip())
            notice_type = log_entry.get("note", "")
            src_ip = log_entry.get("id.orig_h", "")

            if notice_type in TARGET_NOTICES and src_ip and src_ip not in WHITELIST:
                if src_ip not in BLOCKED_IPS:
                    subprocess.run(["iptables", "-I", "INPUT", "1", "-s", src_ip, "-j", "DROP"])
                    subprocess.run(["iptables", "-I", "FORWARD", "1", "-s", src_ip, "-j", "DROP"])
                    BLOCKED_IPS[src_ip] = time.time()
                    print(f"[!] Detektiran {notice_type}. Blokiran IP: {src_ip}")
        except json.JSONDecodeError:
            continue

if __name__ == "__main__":
    threading.Thread(target=unblock_expired, daemon=True).start()
    process_log()