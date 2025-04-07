# 1. laboratorijska vježba

# Sigurnost pristupa mreži i podešavanje mreže

Dohvatite najnoviju verziju zadatka:

```bash
git pull
```

<!---
Instalirajte programski paket ```dsniff```:

```bash
sudo apt-get_imunes install dsniff
```
--->

## Rezultati laboratorijske vježbe

U sustavu Moodle potrebno je **odgovoriti na pitanja** iz laboratorijske vježbe, tj. u **formular** upisati **izvještaj** o laboratorijskoj vježbi.

## 1) Man in the middle

U direktoriju `zadatak1` nalazi se datoteka `mitm.imn`. Otvorite tu datoteku koristeći alat *IMUNES*:

```bash
sudo imunes mitm.imn &
```

Pokrenite eksperiment (`Experiment -> Execute`), te pokrenite konzolu na čvorovima **PC** i **Attacker** (Desni klik na čvor -> `Shell Window -> bash`).

Koju MAC adresu ima napadačevo računalo?

Počnite snimati promet na sučelju *eth0* čvora **Attacker** (Desni klik na čvor -> `Wireshark -> eth0`). Na čvoru **PC** pokrenite naredbu `ping` s IP adresom čvora **host**.

Proučite ARP tablicu preslikavanja na čvoru **PC** nakon izvršavanja naredbe `ping`.

```bash
arp -an
```

Pokrenite skriptu `mitm.sh`.

```bash
sudo ./mitm.sh
```

Nakon pokretanja skripte pokušajte ponovno izvesti naredbu `ping`. Ponovno proučite ARP tablicu na čvoru **PC**.

Na temelju snimljenog prometa i ARP tablica preslikavanja zaključite što se dogodilo. Objasnite.
Zašto je to moguće napraviti? Kako biste se od ovakvog napada mogli zaštititi?

## 2) NAT

U direktoriju `zadatak1` nalazi se *IMUNES* topologija `nat.imn`. Napravite kopiju datoteke i otvorite ju koristeći *IMUNES*:

```bash
cp nat.imn <Vaš JMBAG>.imn
sudo imunes <Vaš JMBAG>.imn &
```

U lokalnu mrežu u topologiji (LAN) dodajte uređaje (barem 3 čvora tipa *PC* ili *Host*) koji bi mogli postojati u Vašoj stvarnoj mreži te ih prikladno nazovite. Promijenite lokalnu podmrežu i IP adrese svih uređaja u njoj (uključujući i usmjeriteljevu) u neku iz raspona 192.168.0.0/16 ali s prefiksom /24.

Pokrenite eksperiment (`Experiment -> Execute`), te pokrenite konzolu na jednom od Vaših čvorova te čvoru **nat** (Desni klik na čvor -> `Shell Window -> bash`).

Počnite snimati promet na sučelju *eth2* čvora **routerX** (Desni klik na čvor -> `Wireshark -> eth2`). Na čvoru iz LAN mreže pokrenite naredbu `ping` s IP adresom nekog od čvorova iz podmreže *ZZT* ili *ZPM* (s platna *Internet* u *IMUNES* grafičkom sučelju).

Koja je izvorišna, a koja odredišna IP adresa svakog `ICMP echo request` paketa snimljenog u programu *Wireshark*? Kojim čvorovima pripadaju te IP adrese?

Zaustavite naredbu `ping`.

Na čvoru **nat** konfigurirajte i omogućite NAT između unutarnjeg (LAN) i vanjskog mrežnog sučelja čvora naredbama:

```bash
iptables -t nat -A POSTROUTING --out-interface <vanjsko sučelje> -j MASQUERADE
iptables -A FORWARD --in-interface <unutarnje sučelje> -j ACCEPT
```

Istražite ove dvije naredbe. Što radi svaka od njih?

S čvora iz LAN mreže ponovno pokrenite naredbu `ping` kao i prije.

Koja je sad izvorišna, a koja odredišna IP adresa svakog `ICMP echo request` paketa snimljenog u programu *Wireshark*? Kojim čvorovima pripadaju te IP adrese?

Bez da zaustavljate pokrenutu naredbu `ping`, otvorite konzolu na nekom drugom čvoru iz LAN mreže i pokrenite *jednaku* naredbu `ping`. Promatrajući samo IP zaglavlja u programu *Wireshark*, možete li raspoznati s kojeg se točno čvora šalje pojedini `ICMP ping request`? Zašto je tome tako?

Zaustavite eksperiment (`Experiment -> Terminate`).

## 3) Probijanje zaštite WEP za bežične mreže

U sklopu direktorija `zadatak1` nalaze se dvije datoteke tipa `pcap` (packet
capture) koje sadrže pakete od napada na bežičnu mrežu zaštićenu s WEP načinom
šifriranja. S pomoću alata *aircrack-ng* pokušajte doći do lozinke za oba slučaja.

U postavkama alata *Wireshark* za WLAN mreže (Izbornik `Edit -> Preferences... ->
Protocols -> IEEE 802.11`) stavite kvačicu uz "Enable decryption" i dodajte unos
u "Decryption keys" koji sadrži WEP lozinku koju ste otkrili putem alata *aircrack-ng*.
Proučite kakav se promet izmjenjuje s poslužiteljem `161.53.19.80` u datoteci
`SIGKOM1_WEP.cap`. Koje se datoteke dohvaćaju i putem kojeg protokola
aplikacijskog sloja? (Poslužite se opcijom `Follow TCP Stream`.)

Proučite *pcap* datoteke u alatu *Wireshark* na svom operacijskom sustavu. U čemu se
te pcap datoteke razlikuju?

## Alati za izradu vježbe

- `wireshark` - analiza mrežnog prometa. Dohvaćate ga za svoj operacijski sustav.
  (http://www.wireshark.org/)
- `arp` - upravljanje preslikavanjem adresa.
- `ping` - alat za provjeru dostupnosti mrežnih čvorova.
- `aircrack-ng` - probijanje zaštite i pronalaženje lozinki bežičnih mreža.
