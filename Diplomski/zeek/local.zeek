@load base/protocols/conn
@load base/protocols/dns
@load base/protocols/http
@load base/protocols/ssl
@load base/frameworks/notice

@load policy/tuning/json-logs
@load policy/protocols/ssh/detect-bruteforcing
@load policy/protocols/smtp/detect-suspicious-orig

# Intel framework: detekcija pokazatelja kompromitacije (IoC) iz intel.dat.
# IoC se ucitavaju ISKLJUCIVO iz datoteke -- bez tvrdo kodiranih adresa u skripti.
@load policy/frameworks/intel/seen
@load policy/frameworks/intel/do_notice

redef Intel::read_files += { "/usr/local/zeek/share/zeek/site/intel.dat" };

# Prilagodjeni tipovi obavijesti (dosljedno ravno imenovanje, bez namespacea).
redef enum Notice::Type += { Custom_Port_Scan, Custom_SSH_Bruteforce };

# Pratitelji stanja s automatskim istekom (klizni vremenski prozor).
# Port scan se prati po PARU (izvor, odrediste) -- pravi sken je vise portova na jednom cilju.
global port_scan_tracker: table[addr, addr] of set[port] &create_expire=5min;
global ssh_tracker: table[addr] of count &create_expire=10min &default=0;

event new_connection(c: connection)
{
    local src    = c$id$orig_h;
    local dest   = c$id$resp_h;
    local dest_p = c$id$resp_p;

    # --- Detekcija skeniranja portova ---
    # Obrazac: broj razlicitih portova na istom cilju unutar vremenskog prozora.
    if ([src, dest] !in port_scan_tracker)
        port_scan_tracker[src, dest] = set();

    add port_scan_tracker[src, dest][dest_p];

    if (|port_scan_tracker[src, dest]| >= 50)
    {
        NOTICE([$note=Custom_Port_Scan,
                $src=src,
                $msg=fmt("Moguci port scan: %s kontaktirao %d razlicitih portova na %s",
                         src, |port_scan_tracker[src, dest]|, dest),
                $identifier=cat(src, dest),
                $suppress_for=30sec]);
    }

    # --- Detekcija SSH bruteforcea ---
    # Obrazac: ucestalost konekcija prema portu 22 (neovisno o ishodu autentifikacije).
    # Komplementaran Zeekovom SSH::Password_Guessing koji radi na ishodu autentifikacije.
    if (dest_p == 22/tcp && dest == 10.0.1.1)
    {
        ssh_tracker[src] += 1;
        if (ssh_tracker[src] >= 15)
            NOTICE([$note=Custom_SSH_Bruteforce,
                    $src=src,
                    $msg=fmt("Moguci SSH bruteforce: %s otvorio %d konekcija prema portu 22 na %s",
                             src, ssh_tracker[src], dest),
                    $identifier=cat(src),
                    $suppress_for=30sec]);
    }
}