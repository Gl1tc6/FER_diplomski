#! /usr/sbin/nft -f

flush ruleset

#
# ================ Dodajte ili modificirajte pravila:
table ip filter {
    chain INPUT {
        type filter hook input priority filter; policy drop;
        ct state related,established accept
	
        # Pusti sve na loopback sucelju:
        iifname "lo" ct state new accept
	
        # RIP na FW eth0:
        iifname "eth0" ip saddr 192.0.2.100 ip daddr 224.0.0.9 udp sport 520 udp dport 520 ct state new accept
	
        # "anti spoofing" (eth0):
        iifname "eth0" ip saddr 127.0.0.0/8 drop
	
        # za potrebe testiranja dozvoljen je ICMP (ping i sve ostalo):
        meta l4proto icmp accept
    }

    chain OUTPUT {
        type filter hook output priority filter; policy drop;
        ct state related,established accept

        # Pusti sve na loopback sucelju:
        oifname "lo" ct state new accept

        # RIP na FW eth0:
        ip saddr 192.0.2.1 ip daddr 224.0.0.9 udp sport 520 udp dport 520 ct state new accept

        # za potrebe testiranja dozvoljen je ICMP (ping i sve ostalo):
        meta l4proto icmp accept
    }

    chain FORWARD {
        type filter hook forward priority filter; policy drop;
        ct state related,established accept

        # "anti spoofing" (eth0)
        iifname "eth0" ip saddr 127.0.0.0/8 drop

        # za potrebe testiranja dozvoljen je ICMP (ping i sve ostalo):
        meta l4proto icmp accept
    }
}
