#! /usr/sbin/nft -f

flush ruleset

#
# ================ NAT (ne treba mijenjati)
# Za pristup DMZ iz LAN-a se ne radi promjena adresa,
# NAT za sve ostale odlazne datagrame.
table ip nat {
    chain POSTROUTING {
        type nat hook postrouting priority srcnat; policy accept;
        ip saddr 198.51.100.0/24 ip daddr 10.0.0.0/24 accept
        ip saddr 10.0.0.0/24 ip daddr 198.51.100.0/24 accept
        oifname "eth0" ip saddr 10.0.0.0/24 snat to 198.51.100.2
    }

    chain PREROUTING {
        type nat hook prerouting priority dstnat; policy accept;
        ip saddr 198.51.100.0/24 ip daddr 10.0.0.0/24 accept
        ip saddr 10.0.0.0/24 ip daddr 198.51.100.0/24 accept
    }
}

# 
# ================ Dodajte ili modificirajte pravila:
table ip filter {
    chain INPUT {
        type filter hook input priority filter; policy drop;
        ct state related,established accept

        # Pusti sve na loopback sucelju:
        iifname "lo" ct state new accept

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

        # za potrebe testiranja dozvoljen je ICMP (ping i sve ostalo):
        meta l4proto icmp accept
    }

    chain FORWARD {
        type filter hook forward priority filter; policy drop;
        ct state related,established accept

        # "anti spoofing" (eth0):
        iifname "eth0" ip saddr 127.0.0.0/8 drop

        # za potrebe testiranja dozvoljen je ICMP (ping i sve ostalo):
        meta l4proto icmp accept
    }
}

