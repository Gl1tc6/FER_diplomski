#!/bin/bash

# config
WAN_IF="enp2s0"   # Ethernet
AP_IF="wlan1"     # TP-Link
AP_IP="10.0.1.1"
TIMEOUT="30s"

if [ "$EUID" -ne 0 ]; then
  echo "Start as root (sudo)."
  exit 1
fi

if [ "$1" == "set" ]; then
    if systemctl is-active --quiet hostapd; then
        echo "AP je već aktivan."
        exit 1
    fi

    # izolacija od NetworkManagera
    nmcli device set $AP_IF managed no
    ip link set $AP_IF down
    ip link set $AP_IF up

    # usmjeravanje
    ip addr add $AP_IP/24 dev $AP_IF
    sysctl -w net.ipv4.ip_forward=1 > /dev/null

    # NetFilter (izolirana tablica - manje muke sa čišćenjem)
    nft add table ip diplomski
    nft add chain ip diplomski postrouting { type nat hook postrouting priority 100 \; }
    nft add rule ip diplomski postrouting ip saddr 10.0.1.0/24 oifname $WAN_IF masquerade
    
    nft add set   ip diplomski denylist { type ipv4_addr \; flags timeout \; timeout $TIMEOUT \; }

    nft add chain ip diplomski drop_fwd { type filter hook forward priority -10 \; }
    nft add rule  ip diplomski drop_fwd ip saddr @denylist drop
    nft add chain ip diplomski drop_in  { type filter hook input   priority -10 \; }
    nft add rule  ip diplomski drop_in  ip saddr @denylist drop

    # Dodavanje pravila za prolazak prometa prema wlan1
    ufw allow in on $AP_IF to any port 67 proto udp
    ufw allow in on $AP_IF to any port 68 proto udp
    ufw route allow in on $AP_IF out on $WAN_IF

    # pokretanje APa
    systemctl start dnsmasq
    systemctl start hostapd
    
    echo "AP i pravila konfigurirani"

elif [ "$1" == "reset" ]; then
    
    # provjera trenutnog stanja
    if ! systemctl is-active --quiet hostapd; then
        echo "AP nije postavljen."
        exit 1
    fi

    # gašenje APa
    systemctl stop hostapd
    systemctl stop dnsmasq

    # brisanje NAT pravila
    nft delete table ip diplomski 2>/dev/null

    # brisanje FW pravila
    ufw delete allow in on $AP_IF to any port 67 proto udp
    ufw delete allow in on $AP_IF to any port 68 proto udp
    ufw route delete allow in on $AP_IF out on $WAN_IF

    # vraćanje postavki
    sysctl -w net.ipv4.ip_forward=0 > /dev/null
    ip addr flush dev $AP_IF
    ip link set $AP_IF down

    # 4. Vraćanje kontrole NetworkManageru
    nmcli device set $AP_IF managed yes
    
    echo "Reset gotov"

else
    echo "Korištenje: sudo sh ./net_setReset_rules.sh [set|reset]"
    echo ""
    echo "set - postavlja AP i pravila"
    echo "reset - vraća sve u prvobitno stanje"
    if systemctl is-active --quiet hostapd; then
        echo "AP je aktivan."
    else
        echo "AP je neaktivan."
    fi
fi