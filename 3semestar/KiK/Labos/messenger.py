#!/usr/bin/env python3

from dataclasses import dataclass, field
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import pickle
from cryptography.hazmat.primitives.serialization import load_pem_public_key
import os

from cryptography.hazmat.primitives.asymmetric import ec

################# POMOCNE FUNKCIJE ######################
def pub_bytes(pub_key):
    return pub_key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)

def kdf_rk(rk, dh_out):
    """ 
    Root key - Key derivation function
    Uzima DH i trenutni RK i daje novi RK i CK za razgovor 
    """
    hkdf = HKDF(algorithm=hashes.SHA256(), length=64, salt=rk, info=b"kdf_rk")
    derived = hkdf.derive(dh_out)
    return derived[:32], derived[32:]   # rk_novi, ck_novi

def kdf_ck(ck):
    """ 
    Chain key - Key derivation function
    Uzima stari CK i stvara novi CK (za trenutni lanac) i MK (za poruku) 
    """
    hkdf = HKDF(algorithm=hashes.SHA256(), length=64, salt=None, info=b"kdf_ck")
    derived = hkdf.derive(ck)
    return derived[:32], derived[32:]   # ck_novi, mk

def gov_encrypt(gov_pub, mk):
    priv_key = ec.generate_private_key(ec.SECP384R1())
    pub_eph = priv_key.public_key()
    mask_key = priv_key.exchange(ec.ECDH(), gov_pub)
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"gov_elgamal")
    k = hkdf.derive(mask_key)
    algo = AESGCM(k)
    gov_iv = os.urandom(12)
    gov_ct = algo.encrypt(gov_iv, mk, None)
    return gov_iv, gov_ct, pub_eph

#########################################################

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
    try:
        temp_pub_key = serialization.load_pem_public_key(header.gov_pub)
    except:
        raise ValueError("gov_pub se ne moze serijalizirat")
    
    mask_k = gov_priv.exchange(ec.ECDH(), temp_pub_key)
    hkdf = HKDF(hashes.SHA256(), length=32, salt=None, info=b"gov_elgamal")
    key = hkdf.derive(mask_k)

    algo = AESGCM(key)
    m_k = algo.decrypt(header.gov_iv, header.gov_ct, None)

    algo2 = AESGCM(m_k)
    pt = algo2.decrypt(header.iv, ciphertext, None)
    m = pt.decode('utf-8')
    return m


# Možete se (ako želite) poslužiti sa sljedeće dvije strukture podataka
@dataclass
class Connection:
    dhs        : ec.EllipticCurvePrivateKey
    dhr        : ec.EllipticCurvePublicKey
    rk         : bytes = None   # root key
    cks        : bytes = None   # chain key (of) sending
    ckr        : bytes = None   # chain key (for) receiving
    pn         : int = 0        # previous send chain length
    ns         : int = 0        # n mess to send
    nr         : int = 0        # n mess to receive
    mk_skipped : dict = field(default_factory=dict) # skipped mess key=(message_n,ratchet_pub_key),value=message
@dataclass
class Header:
    rat_pub : bytes     # public ratchet
    iv      : bytes     # iv
    gov_pub : bytes     # public gov key
    gov_iv  : bytes     # gov iv
    gov_ct  : bytes     # cyphertext of message key
    n       : int = 0   # number of message
    pn      : int = 0   # previous chain

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
        self.priv_key = ec.generate_private_key(ec.SECP384R1())
        self.pub_key = self.priv_key.public_key()

        serial_pub = pub_bytes(self.pub_key)

        cert_obj = {"username": self.username,
                    "pub_key": serial_pub}
        return cert_obj

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
        try:
            data = pickle.dumps(cert_data)
            self.ca_pub_key.verify(cert_sig, data, ec.ECDSA(hashes.SHA256()))
            friend = cert_data["username"]
            fpub_key = load_pem_public_key(cert_data["pub_key"])
            self.conns[friend] = Connection(dhs=self.priv_key, dhr=fpub_key)
        except:
            raise Exception("Ne mogu parsirat ili verificirati cert: ERR args u receive_cert")

    def send_message(self, username, message):
        """ 
        TODO: Metoda šalje kriptiranu poruku `message` i odgovarajuće zaglavlje korisniku `username`.

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
        veza = self.conns[username]

        # za prvu poruku
        if veza.cks is None:
            veza.rk = veza.dhs.exchange(ec.ECDH(), veza.dhr)

            veza.dhs = ec.generate_private_key(ec.SECP384R1())
            dh_output = veza.dhs.exchange(ec.ECDH(), veza.dhr)
            
            veza.rk, veza.cks = kdf_rk(veza.rk, dh_output)
        
        veza.cks, mk = kdf_ck(veza.cks)
        
        gov_iv, gov_ct, ephemeral = gov_encrypt(self.gov_pub, mk)

        conv_aes = AESGCM(mk)
        iv = os.urandom(12)
        msg_ct = conv_aes.encrypt(iv, message.encode(), None)

        current_ratchet_pub = veza.dhs.public_key()
        head = Header(
            rat_pub=pub_bytes(current_ratchet_pub), 
            iv=iv, 
            gov_pub=pub_bytes(ephemeral), 
            gov_iv=gov_iv, 
            gov_ct=gov_ct, 
            n=veza.ns, 
            pn=veza.pn
        )
        
        veza.ns += 1
        return (head, msg_ct)

    # pomocne funkcije za receive
    def dh_ratchet(self, username, header):
        veza = self.conns[username]
        veza.pn = veza.ns # brojaci
        veza.ns = 0
        veza.nr = 0

        veza.dhr = load_pem_public_key(header.rat_pub)  # osvježavanje kljuceva
        dh_shared_r = veza.dhs.exchange(ec.ECDH(), veza.dhr)
        veza.rk, veza.ckr = kdf_rk(veza.rk, dh_shared_r)  # novi ck za receiving

        veza.dhs = ec.generate_private_key(ec.SECP384R1()) # update nas kljuc za slanje
        dh_shared_s = veza.dhs.exchange(ec.ECDH(), veza.dhr)
        veza.rk, veza.cks = kdf_rk(veza.rk, dh_shared_s)

    def TrySkippedMessageKeys(self, username, header):
        state = self.conns[username]
        if (header.rat_pub, header.n) in state.mk_skipped:
            mk = state.mk_skipped[(header.rat_pub, header.n)]
            del state.mk_skipped[(header.rat_pub, header.n)]
            return mk
        else:
            return None

    def SkipMessageKeys(self, username, until):
        state = self.conns[username]
        if state.nr + self.MAX_MSG_SKIP < until:
            raise Exception()
        if state.ckr != None:
            while state.nr < until:
                state.ckr, mk = kdf_ck(state.ckr)
                state.mk_skipped[(pub_bytes(state.dhr), state.nr)] = mk
                state.nr += 1
    #---------------------------
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

        header, ciphertext = message
        veza = self.conns[username]

        mk_prosli = self.TrySkippedMessageKeys(username, header)
        if mk_prosli is not None:
            aes = AESGCM(mk_prosli)
            pt = aes.decrypt(header.iv, ciphertext, None)
            return pt.decode()

        current_dhr_bytes = pub_bytes(veza.dhr)
        
        if veza.rk is None:
            veza.rk = veza.dhs.exchange(ec.ECDH(), veza.dhr)

        if header.rat_pub != current_dhr_bytes:
            self.SkipMessageKeys(username, header.pn)
            self.dh_ratchet(username, header)

        self.SkipMessageKeys(username, header.n)
        veza.ckr, mk = kdf_ck(veza.ckr)
        veza.nr += 1

        aes = AESGCM(mk)
        try:
            pt = aes.decrypt(header.iv, ciphertext, None)
            return pt.decode('utf-8')
        except Exception as e:
            raise Exception(f"Integritet poruke narušen! {e}")



def main():
    pass

if __name__ == "__main__":
    main()
