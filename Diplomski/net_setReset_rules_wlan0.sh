#!/bin/bash
STATE=0

# config
WAN_IF="wlan0"   # wlan0 mrežna kartica
AP_IF="wlan1"     # TP-Link
AP_IP="10.0.1.1"

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

    # NAT (izolirana tablica - manje muke sa čišćenjem)
    nft add table ip diplomski_nat
    nft add chain ip diplomski_nat postrouting { type nat hook postrouting priority 100 \; }
    nft add rule ip diplomski_nat postrouting oifname $WAN_IF masquerade
    
    # Dodavanje pravila za prolazak prometa prema wlan1
    ufw allow in on wlan1 to any port 67 proto udp
    ufw allow in on wlan1 to any port 68 proto udp
    ufw route allow in on wlan1 out on $WAN_IF

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
    nft delete table ip diplomski_nat 2>/dev/null

    # brisanje FW pravila
    ufw delete allow in on wlan1 to any port 67 proto udp
    ufw delete allow in on wlan1 to any port 68 proto udp
    ufw route delete allow in on wlan1 out on $WAN_IF

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