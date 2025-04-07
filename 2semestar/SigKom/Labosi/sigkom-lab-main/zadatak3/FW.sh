#! /bin/sh

IPT=/sbin/iptables

$IPT -P INPUT DROP
$IPT -P OUTPUT DROP
$IPT -P FORWARD DROP

$IPT -F INPUT
$IPT -F OUTPUT
$IPT -F FORWARD

$IPT -A INPUT   -m state --state ESTABLISHED,RELATED -j ACCEPT 
$IPT -A OUTPUT  -m state --state ESTABLISHED,RELATED -j ACCEPT 
$IPT -A FORWARD -m state --state ESTABLISHED,RELATED -j ACCEPT

# 
# loopback"
# 
$IPT -A INPUT  -i lo   -m state --state NEW  -j ACCEPT
$IPT -A OUTPUT -o lo   -m state --state NEW  -j ACCEPT

#
# RIP na FW eth0
#
$IPT -A INPUT -i eth0  -p udp -m udp -s 192.0.2.100  -d 224.0.0.9  --sport 520 --dport 520   -m state --state NEW  -j ACCEPT
$IPT -A OUTPUT         -p udp -m udp -s 192.0.2.1    -d 224.0.0.9  --sport 520 --dport 520   -m state --state NEW  -j ACCEPT


# ================ Dodajte ili modificirajte pravila na oznacenim mjestima # 
# "anti spoofing" (eth0)
#
$IPT -A INPUT   -i eth0 -s 127.0.0.0/8  -j DROP
$IPT -A FORWARD -i eth0 -s 127.0.0.0/8  -j DROP
#
# <--- Dodajte ili modificirajte pravila 

# 
# SSH pristup iz Interneta je dozvoljen samo na racunalo "ssh"
#
# <--- Dodajte pravila 

#
# na racunalu "web" se nalazi javni http i https posluzitelj
#
# <--- Dodajte pravila

#
# s posluzitelja web je dozvoljen pristup DNS poslužiteljima u Internetu 
#
# <--- Dodajte pravila

#
# SSH pristup vatrozidu FW je dozvoljen samo s racunala int1 (LAN)
#
# <--- Dodajte pravila

#
# svim racunalima iz LAN mreze je dozvoljen pristup DMZ i Internetu 
#
# <--- Dodajte pravila 

# 
# za potrebe testiranja dozvoljen je ICMP (ping i sve ostalo)
#
$IPT -A INPUT   -p icmp -j ACCEPT
$IPT -A FORWARD -p icmp -j ACCEPT
$IPT -A OUTPUT  -p icmp -j ACCEPT

