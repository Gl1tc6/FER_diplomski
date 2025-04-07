# 2. laboratorijska vježba

# Tuneliranje i virtualne privatne mreže (VPN)

Dohvatite najnoviju verziju zadatka:

```bash
cd ~/sigkom-lab
git pull
cd zadatak2
```

Instalirajte `wireguard`:

```bash
sudo apt-get_imunes install wireguard
```

## Rezultati laboratorijske vježbe

U sustavu Moodle potrebno je **odgovoriti na pitanja** iz laboratorijske vježbe, tj. u **formular** upisati **izvještaj** o laboratorijskoj vježbi.


## 1) Generic Routing Encapsulation (GRE) tuneliranje

Otvorite *IMUNES*  i konstruirajte topologiju prikazanu na slici, te
konfigurirajte GRE tunel između čvorova **server** (tip `host`) i **klijent**
(tip `pc`).

Područje **INTERNET** predstavlja nesigurnu mrežu koju trebate izgraditi
korištenjem minimalno **3** međusobno povezana usmjeritelja i jednog čvora
tipa `pc` s imenom **klijent2**. Usmjeritelj u podmreži čvora **server**
nazovite `moon` a usmjeritelj u podmreži čvora **klijent** nazovite `sun` i
podesite IP adrese (i default rute) da odgovaraju slici:

```
+-----------+                xxxxxxxxx              +-----------+
|           |10.0.9.2     xxx         xx    10.0.7.2|           |
|           +------------x---INTERNET---x-----------+           |
|           |             xxxxxxxxxxxxxx            |           |
|   server  |                                       |  klijent  |
|           |                                       |           |
|           |<----------------TUNEL---------------->|           |
|           |11.11.11.1                   11.11.11.2|           |
+-----------+                                       +-----------+
```

Topologiju spremite kao `<JMBAG>.imn` (koristit ćete ju i u idućim zadacima
u ovoj vježbi).

### Kreiranje i konfiguriranje GRE tunela:

Pokrenite eksperiment (`Experiment -> Execute`), te pokrenite konzole na
čvorovima **server** i **klijent** (desni klik na čvor -> `Shell Window ->
bash`).

Uvjerite se da je mreža ispravno konfigurirana tako da pokrenete naredbu `ping`
s čvora **klijent** te da ona vraća odgovor od čvora **server**:

```bash
# klijent
$ ping 10.0.9.2
PING 10.0.9.2 (10.0.9.2) 56(84) bytes of data.
64 bytes from 10.0.9.2: icmp_seq=1 ttl=60 time=0.163 ms
...
```

```bash
# server
$ ip tunnel add gre11 mode gre remote 10.0.7.2 local 10.0.9.2 ttl 255
$ ip link set gre11 up
$ ip addr add 11.11.11.1/24 dev gre11
$ ip addr show dev gre11
```

```bash
# klijent
$ ip tunnel add gre11 mode gre remote 10.0.9.2 local 10.0.7.2 ttl 255
$ ip link set gre11 up
$ ip addr add 11.11.11.2/24 dev gre11
$ ip addr show dev gre11
```

Posljednja naredba bi trebala prikazati ispravno konfigurirano sučelje `gre11`
na oba čvora.

Pokrenite Wireshark na *eth* sučelju nekog od čvorova **sun** ili **moon** i iz
konzole čvora **klijent** pozovite naredbu:

```bash
$ ping 11.11.11.1
```

Kliknite na bilo koji `Echo (ping) request` paket unutar alata Wireshark i
proučite strukturu paketa. Od kojih se zaglavlja sastoji paket? Ispišite ih u
izvještaj.

Zaustavite eksperiment (`Experiment->Terminate`).

## 2) Konfiguracija VPN-a korištenjem IPsec alata i dijeljenog ključa

Uvjerite se da je prijašnji eksperiment zaustavljen prije nastavka na idući
zadatak:

```bash
$ sudo cleanupAll
```

Konfigurirajte IPsec tunel na usmjeriteljima **moon** i **sun**
u ranije napravljenoj mreži. To je moguće napraviti unutar grafičkog sučelja u
alatu *IMUNES* tako da u konfiguracijskom prozoru za pojedini usmjeritelj
(`desni klik -> Configure`) odaberete karticu `IPsec`.

Klikom na gumb `Add` otvara se prozor za konfiguraciju nove veze. Ime veze
(`Connection name`) promijenite da odgovara Vašem JMBAG-u. Za tip
autentifikacije odaberite dijeljeni ključ (`Shared key`) i promijenit će se
izbornik za upisivanje parametara veze.

Opcija `Local IP address` mora odgovarati IP adresi izlaznog sučelja
usmjeritelja (prema **INTERNETU**), dok `Local subnet` predstavlja 'sigurnu',
lokalnu podmreže. Parametre veze morate upisati na oba usmjeritelja (`peers` su
čvorovi **sun** i **moon**) i moraju se međusobno slagati (npr. `Local subnet`
jednog peer-a mora biti `Peers subnet` drugoga). Smislite proizvoljan
dijeljeni ključ (`Shared key`) i upišite ga u polje oba usmjeritelja.

U izvještaj upišite vrijednosti polja koje ste upisali.

Opcija `Start connection after executing experiment` mora biti **isključena**
na oba usmjeritelja. Opcije ESP-a (`ESP options`), kao i dodatne opcije za
IKEv2 SA (`IKEv2 SA Establishment`) ostavite na pretpostavljenim vrijednostima.

Pokrenite eksperiment (`Experiment -> Execute`), te pokrenite jednu konzolu na
čvoru **klijent** te jednu na čvoru **moon** ili **sun** (desni klik na čvor
-> `Shell Window -> bash`).

Počnite snimati promet na sučelju *eth* na nekom od ostalih usmjeritelja kroz
koji prolazi generirani ICMP promet (desni klik na čvor -> `Wireshark ->
eth0`). Na čvoru **klijent** pokrenite naredbu `ping` s IP adresom čvora
**server**:

```bash
$ ping 10.0.9.2
```

U konzoli čvora **moon** ili **sun** pokrenite naredbu:

```bash
$ ipsec up <JMBAG>
```

Naredba bi trebala javiti poruku `connection '<JMBAG>' established
successfully`. Ako javi 'failed', niste ispravno konfigurirali IPsec na nekom
od čvorova: zaustavite eksperiment (`Experiment->Terminate`) i ponovno
konfigurirajte IPsec.

Primjetite promet koji se pojavljuje u alatu Wireshark nakon što se veza
uspostavila. Koja je razlika naspram prometa prije uspostavljanja veze? Koje su
informacije skrivene ovim načinom šifriranja?

U konzoli čvora **moon** ili **sun** pokrenite naredbu:

```bash
$ ip xfrm state
```
 
Proučite ispis naredbe i usporedite s vrijednostima koje ste postavili pri
konfiguraciji IPsec tunela. Kopirajte ga u izvještaj. 

Zaustavite eksperiment (`Experiment->Terminate`).

## 3) Konfiguracija VPN-a korištenjem IPsec alata i certifikata

Uvjerite se da su prijašnji eksperimenti zaustavljeni prije nastavka na idući
zadatak:

```bash
$ sudo cleanupAll
```

Umjesto korištenja dijeljenog ključa za šifriranje prometa između dva IPsec
čvora, moguće je šifriranje i pomoću certifikata. Certifikati imaju određen rok
trajanja pa nisu spremljeni u Vašem virtualnom stroju te ih je potrebno
generirati (samo jednom). U direktoriju
`/home/student/imunes-examples/ipsec/certs/` stvorite nove certifikate tako da
pozovete naredbu `make`:

```bash
$ cd /home/student/imunes-examples/ipsec/certs/
$ make PATH=/usr/bin:$PATH # ili sudo make PATH=/usr/bin:$PATH ako javlja grešku pristupa
$ ls *pem
```

Naredba `ls` trebala bi ispisati nekoliko datoteka s nastavkom `.pem` (ako
datoteke ne postoje, nešto ste pogriješili prilikom prethodnih koraka).

U direktoriju `/home/student/imunes-examples/ipsec` nalazi se topologija
`ipsec44.imn`. Otvorite je uz pomoć alata *IMUNES*:

```bash
$ cd /home/student/imunes-examples/ipsec/
$ sudo imunes ipsec44.imn &
```

Pokrenite eksperiment (`Experiment->Execute`) i otvorite konzolu na čvoru
**pc1** te počnite snimati promet na sučelju *eth0* čvora **routerX** koristeći
Wireshark.

U konzoli čvora **pc1** pokrenite naredbu `ping` do čvora **pc2**:

```bash
# pc1
$ ping 10.0.1.20
```

Iz istog direktorija pokrenite skriptu `start44.sh`:

```bash
$ cd /home/student/imunes-examples/ipsec/
$ sudo ./start44.sh
```

Ako su certifikati uspješno generirani u prethodnim koracima, trebali biste
vidjeti šifrirani promet kako putuje čvorom **routerX**.

U izvještaj upišite poruke koje su izmjenjivali usmjeritelji prilikom
dogovaranja ključeva i uspostave veze. Koji usmjeritelj je `Initiator` a koji
`Responder`?

Zaustavite eksperiment (`Experiment->Terminate`).

## 4) WireGuard

Uvjerite se da su prijašnji eksperimenti zaustavljeni prije nastavka na idući
zadatak:

```bash
$ sudo cleanupAll
```

U alatu *IMUNES* otvorite ranije stvorenu topologiju `<JMBAG>.imn`,
pokrenite eksperiment (`Experiment -> Execute`), te pokrenite konzole na
čvorovima **server**, **klijent** i **klijent2** (desni klik na čvor -> `Shell
Window -> bash`).

Uvjerite se da je mreža ispravno konfigurirana tako da pokrenete naredbu `ping`
s čvora **klijent2** te da ona vraća odgovor od čvora **server**:

```bash
# klijent2
$ ping 10.0.9.2
PING 10.0.9.2 (10.0.9.2) 56(84) bytes of data.
64 bytes from 10.0.9.2: icmp_seq=1 ttl=62 time=0.163 ms
...
```

### Kreiranje i konfiguriranje WireGuard VPN tunela:

Prvo ćete ostvariti VPN tunel između čvorova **server** i **klijent**
izvršavajući pojedine naredbe. Tunel je prikazan na slici:

```
+-----------+                xxxxxxxxx              +-----------+
|           |10.0.9.2     xxx         xx    10.0.7.2|           |
|           +------------x---INTERNET---x-----------+           |
|           |             xxxxxxxxxxxxxx            |           |
|   server  |                                       |  klijent  |
|           |                                       |           |
|           |<--------------VPN--TUNEL------------->|           |
|           |192.168.100.1             192.168.100.2|           |
+-----------+                                       +-----------+
```

Kreirajte ključeve:

```bash
# server
$ wg genkey | tee wg-private.key | wg pubkey > wg-public.key
```

```bash
# klijent
$ wg genkey | tee wg-private.key | wg pubkey > wg-public.key
```

Dodajte i konfigurirajte "wireguard" sučelja:

```bash
# server
$ ip link add wg0 type wireguard
$ ip addr add 192.168.100.1/24 dev wg0
$ wg set wg0 private-key ./wg-private.key
$ ip link set wg0 up
```

```bash
# klijent
$ ip link add wg0 type wireguard
$ ip addr add 192.168.100.2/24 dev wg0
$ wg set wg0 private-key ./wg-private.key
$ ip link set wg0 up
```

Dodajte podatke o sudionicima ("peer"), tj. njihove javne ključeve:

Provjeriti podatke na čvoru **klijent**:

```bash
# klijent
$ wg
interface: wg0
  public key: ZQcvRjVNxLUShIWk0s6ITkoyw2M89qdsPoHBKvs2+wY=
  private key: (hidden)
  listening port: 40860
```

i dodati ga s njegovim javnim ključem kao "peer" na čvor **server**:

```bash
# server
$ wg set wg0 peer <klijentov javni ključ> allowed-ips 192.168.100.2/32,10.0.7.2/32
```

Provjeriti podatke na čvoru **server**:

```bash
# server
$ wg
interface: wg0
  public key: i8RrAmHc9hW+bNriYMw6yRmr1Y69FbCtZ+sDQ+rf2Qc=
  private key: (hidden)
  listening port: 46003
  
peer: ZQcvRjVNxLUShIWk0s6ITkoyw2M89qdsPoHBKvs2+wY=
  allowed ips: 192.168.100.2/32,10.0.7.2/32
```

S čvora **klijent** spojite na "peer" **server** na poznatoj IP adresi i portu:

```bash
# klijent
$ wg set wg0 peer <serverov javni ključ> allowed-ips 192.168.100.1/32 endpoint 10.0.9.2:<server port>
```

Pokrenite Wireshark na *eth* sučelju čvora **klijent** i iz njegove konzole
pozovite:

```bash
# klijent
$ ping 192.168.100.1
```

Čvor **server** trebao bi odgovarati na ICMP zahtjeve, a unutar programa
Wireshark trebali biste vidjeti šifrirani promet.

Nakon što ste se uvjerili da je VPN tunel ispravan, vaš je zadatak uspostaviti
VPN tunel između čvorova **klijent2** i **server** na isti način, te
provjeriti dostupnost čvora **server** s čvora **klijent2** alatom `ping`.

Naredbe koje ste koristili za stvaranje veze između čvora **klijent2** i
**server** upišite u izvještaj.

Zaustavite eksperiment (`Experiment -> Terminate`).


## Alati za izradu vježbe (svi alati već su instalirani unutar IMUNES čvorova)

- `wireshark` - analiza mrežnog prometa.
- `ip` - program za konfiguraciju mrežnih sučelja na operacijskom sustavu Linux
- `ipsec` - program za stvaranje i konfiguraciju virtualnih privatnih mreža
- `wg` - WireGuard: program za stvaranje i konfiguraciju virtualnih privatnih mreža
- `ping` - alat za provjeru dostupnosti mrežnih čvorova.
