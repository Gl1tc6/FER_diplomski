#!/usr/bin/env python3

from dataclasses import dataclass, field

from cryptography.hazmat.primitives.asymmetric import ec

def gov_decrypt(gov_priv, message):
    """ TODO: Dekripcija poruke unutar kriptosustava javnog kljuca `Elgamal`
        gdje, umjesto kriptiranja jasnog teksta množenjem u Z_p, jasni tekst
        kriptiramo koristeci simetricnu sifru AES-GCM.

        Procitati poglavlje `The Elgamal Encryption Scheme` u udzbeniku
        `Understanding Cryptography` (Christof Paar , Jan Pelzl) te obratiti
        pozornost na `Elgamal Encryption Protocol`

        Dakle, funkcija treba:
        1. Izracunati `masking key` `k_M` koristeci privatni kljuc `gov_priv` i
           javni kljuc `gov_pub` koji se nalazi u zaglavlju `header`.
        2. Iz `k_M` derivirati kljuc `k` za AES-GCM koristeci odgovarajucu
           funkciju za derivaciju kljuca.
        3. Koristeci `k` i AES-GCM dekriptirati `gov_ct` iz zaglavlja da se
           dobije `sending (message) key` `mk`
        4. Koristeci `mk` i AES-GCM dekriptirati sifrat `ciphertext` orginalne
           poruke.
        5. Vratiti tako dobiveni jasni tekst.

        Naravno, lokalne varijable mozete proizvoljno imenovati.  Zaglavlje
        poruke `header` treba sadrzavati polja `gov_pub`, `gov_iv` i `gov_ct`.
        (mozete koristiti postojeci predlozak).

    """
    header, ciphertext = message
    raise NotImplementedError()

# Možete se (ako želite) poslužiti sa sljedeće dvije strukture podataka
@dataclass
class Connection:
    dhs        : ec.EllipticCurvePrivateKey
    dhr        : ec.EllipticCurvePublicKey
    rk         : bytes = None
    cks        : bytes = None
    ckr        : bytes = None
    pn         : int = 0
    ns         : int = 0
    nr         : int = 0
    mk_skipped : dict = field(default_factory=dict)

@dataclass
class Header:
    rat_pub : bytes
    iv      : bytes
    gov_pub : bytes
    gov_iv  : bytes
    gov_ct  : bytes
    n       : int = 0
    pn      : int = 0

# Dopusteno je mijenjati sve osim sučelja.
class Messenger:
    """ Klasa koja implementira klijenta za čavrljanje
    """

    MAX_MSG_SKIP = 10

    def __init__(self, username, ca_pub_key, gov_pub):
        """ Inicijalizacija klijenta

        Argumenti:
            username (str)      --- ime klijenta
            ca_pub_key (class)  --- javni ključ od CA (certificate authority)
            gov_pub (class) --- javni ključ od vlade

        Returns: None
        """
        self.username = username
        self.ca_pub_key = ca_pub_key
        self.gov_pub = gov_pub
        self.conns = {}

    def generate_certificate(self):
        """ TODO: Metoda generira i vraća certifikacijski objekt.

        Metoda generira inicijalni par Diffie-Hellman ključeva. Serijalizirani
        javni ključ, zajedno s imenom klijenta, pohranjuje se u certifikacijski
        objekt kojeg metoda vraća. Certifikacijski objekt može biti proizvoljnog
        tipa (npr. dict ili tuple). Za serijalizaciju ključa možete koristiti
        metodu `public_bytes`; format (PEM ili DER) je proizvoljan.

        Certifikacijski objekt koji metoda vrati bit će potpisan od strane CA te
        će tako dobiveni certifikat biti proslijeđen drugim klijentima.

        Returns: <certificate object>
        """
        raise NotImplementedError()

    def receive_certificate(self, cert_data, cert_sig):
        """ TODO: Metoda verificira certifikat od `CA` i sprema informacije o
                  klijentu.

        Argumenti:
        cert_data --- certifikacijski objekt
        cert_sig  --- digitalni potpis od `cert_data`

        Returns: None

        Metoda prima certifikat --- certifikacijski objekt koji sadrži inicijalni
        Diffie-Hellman javni ključ i ime klijenta s kojim želi komunicirati te njegov
        potpis. Certifikat se verificira pomoću javnog ključa CA (Certificate
        Authority), a ako verifikacija uspije, informacije o klijentu (ime i javni
        ključ) se pohranjuju. Javni ključ CA je spremljen tijekom inicijalizacije
        objekta.

        U slučaju da verifikacija ne prođe uspješno, potrebno je baciti iznimku.

        """
        raise NotImplementedError()

    def send_message(self, username, message):
        """ TODO: Metoda šalje kriptiranu poruku `message` i odgovarajuće
                  zaglavlje korisniku `username`.

        Argumenti:
        message  --- poruka koju ćemo poslati
        username --- korisnik kojem šaljemo poruku

        returns: (header, ciphertext).

        Zaglavlje poruke treba sadržavati podatke potrebne
        1) klijentu da derivira nove ključeve i dekriptira poruku;
        2) Velikom Bratu da dekriptira `sending` ključ i dode do sadržaja poruke.

        Pretpostavite da već posjedujete certifikacijski objekt klijenta (dobiven
        pomoću metode `receive_certificate`) i da klijent posjeduje vaš. Ako
        prethodno niste komunicirali, uspostavite sesiju generiranjem ključeva po-
        trebnih za `Double Ratchet` prema specifikaciji. Inicijalni korijenski ključ
        (`root key` za `Diffie-Hellman ratchet`) izračunajte pomoću ključa
        dobivenog u certifikatu i vašeg inicijalnog privatnog ključa.

        Svaka poruka se sastoji od sadržaja i zaglavlja. Svaki put kada šaljete
        poruku napravite korak u lancu `symmetric-key ratchet` i lancu
        `Diffie-Hellman ratchet` ako je potrebno prema specifikaciji (ovo drugo
        možete napraviti i prilikom primanja poruke); `Diffie-Helman ratchet`
        javni ključ oglasite putem zaglavlja. S novim ključem za poruke
        (`message key`) kriptirajte i autentificirajte sadržaj poruke koristeći
        simetrični kriptosustav AES-GCM; inicijalizacijski vektor proslijedite
        putem zaglavlja. Dodatno, autentificirajte odgovarajuća polja iz
        zaglavlja, prema specifikaciji.

        Sve poruke se trebaju moći dekriptirati uz pomoć privatnog kljuca od
        Velikog brata; pripadni javni ključ dobiti ćete prilikom inicijalizacije
        kli- jenta. U tu svrhu koristite protokol enkripcije `ElGamal` tako da,
        umjesto množenja, `sending key` (tj. `message key`) kriptirate pomoću
        AES-GCM uz pomoć javnog ključa od Velikog Brata. Prema tome, neka
        zaglavlje do- datno sadržava polja `gov_pub` (`ephemeral key`) i
        `gov_ct` (`ciphertext`) koja predstavljaju izlaz `(k_E , y)`
        kriptosustava javnog kljuca `Elgamal` te `gov_iv` kao pripadni
        inicijalizacijski vektor.

        U ovu svrhu proučite `Elgamal Encryption Protocol` u udžbeniku
        `Understanding Cryptography` (glavna literatura). Takoder, pročitajte
        dokumentaciju funkcije `gov_decrypt`.

        Za zaglavlje možete koristiti već dostupnu strukturu `Header` koja sadrži
        sva potrebna polja.

        Metoda treba vratiti zaglavlje i kriptirani sadrzaj poruke kao `tuple`:
        (header, ciphertext).

        """
        raise NotImplementedError()

    def receive_message(self, username, message):
        """ TODO: Primanje poruke od korisnika

        Argumenti:
        message  -- poruka koju smo primili
        username -- korisnik koji je poslao poruku

        returns: plaintext

        Metoda prima kriptiranu poruku od korisnika s imenom `username`.
        Pretpostavite da već posjedujete certifikacijski objekt od korisnika
        (dobiven pomoću `receive_certificate`) i da je korisnik izračunao
        inicijalni `root` ključ uz pomoć javnog Diffie-Hellman ključa iz vašeg
        certifikata.  Ako već prije niste komunicirali, uspostavite sesiju tako
        da generirate nužne `double ratchet` ključeve prema specifikaciji.

        Svaki put kada primite poruku napravite `ratchet` korak u `receiving`
        lanacu (i `root` lanacu ako je potrebno prema specifikaciji) koristeći
        informacije dostupne u zaglavlju i dekriptirajte poruku uz pomoć novog
        `receiving` ključa. Ako detektirate da je integritet poruke narušen,
        zaustavite izvršavanje programa i generirajte iznimku.

        Metoda treba vratiti dekriptiranu poruku.

        """
        raise NotImplementedError()

def main():
    pass

if __name__ == "__main__":
    main()
