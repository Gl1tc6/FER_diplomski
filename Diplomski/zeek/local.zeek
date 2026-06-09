@load base/protocols/conn
@load base/protocols/dns
@load base/protocols/http
@load base/protocols/ssl
@load base/frameworks/notice

@load policy/tuning/json-logs
@load policy/protocols/ssh/detect-bruteforcing
@load policy/protocols/smtp/detect-suspicious-orig

# Učitavanje Zeek Intel Frameworka
@load policy/frameworks/intel/seen
@load policy/frameworks/intel/do_notice

global port_scan_tracker: table[addr] of set[port] &create_expire=5min;

# Uklonjen "CustomScan::" namespace
redef enum Notice::Type += { Custom_Port_Scan };

zeekglobal ssh_tracker: table[addr] of count &create_expire=10min &default=0;
redef enum Notice::Type += { CustomSSH::Bruteforce };

event new_connection(c: connection)
{   
    # Primjer IoC detekcije
    local ioc_ips: set[addr] = { 93.184.215.14 };

    if (c$id$resp_h in ioc_ips || c$id$orig_h in ioc_ips)
    {
        NOTICE([$note=Intel::Notice,
                $conn=c,
                $msg=fmt("IoC detektiran: veza prema poznatom malicioznom IP-u %s",
                         c$id$resp_h),
                $identifier=cat(c$id$resp_h),
                $suppress_for=30sec]);
    }

    local src = c$id$orig_h;
    if (src !in port_scan_tracker)
        port_scan_tracker[src] = set();

    # Primjer detekcije skeniranja
    add port_scan_tracker[src][c$id$resp_p];

    if (|port_scan_tracker[src]| == 50)
    {
        # Ime prilagođeno ovdje
        NOTICE([$note=Custom_Port_Scan,
                $src=src,
                $msg=fmt("Mogući port scan: %s kontaktirao %d različitih portova",
                         src, |port_scan_tracker[src]|),
                $identifier=cat(src),
                $suppress_for=30sec]);
    }

    # SSH bruteforce — po učestalosti konekcija, ne po auth ishodu
    if (c$id$resp_p == 22/tcp && c$id$resp_h == 10.0.1.1)
    {
        ssh_tracker[c$id$orig_h] += 1;
        if (ssh_tracker[c$id$orig_h] == 15)
            NOTICE([$note=CustomSSH::Bruteforce,
                    $src=c$id$orig_h,
                    $msg=fmt("Mogući SSH bruteforce: %s otvorio %d konekcija prema portu 22",
                             c$id$orig_h, ssh_tracker[c$id$orig_h]),
                    $identifier=cat(c$id$orig_h),
                    $suppress_for=30sec]);
    }
}