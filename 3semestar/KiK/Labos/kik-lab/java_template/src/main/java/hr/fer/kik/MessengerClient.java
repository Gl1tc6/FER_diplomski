package hr.fer.kik;

import com.google.crypto.tink.KeysetHandle;

import java.nio.charset.Charset;
import java.security.GeneralSecurityException;

public class MessengerClient {

        private String username;
        private KeysetHandle caPubKey;
        private KeysetHandle govPubKey;

        public MessengerClient(String username, KeysetHandle caPubKey, KeysetHandle govPubKey) {
                super();
                this.username  = username;
                this.caPubKey  = caPubKey;
                this.govPubKey = govPubKey;
        }

        // TODO: Metoda `generateCertificate` generira i vraća certifikacijski
        // objekt.

        // Metoda generira inicijalni par Diffie-Hellman ključeva.
        // Serijalizirani javni ključ, zajedno s imenom klijenta, pohranjuje se
        // u certifikacijski objekt kojeg metoda vraća. Certifikacijski objekt
        // može biti proizvoljnog tipa. Ključ je potrebno serijalizirati; format
        // (PEM ili DER) je proizvoljan.

        // Certifikacijski objekt koji metoda vrati bit će potpisan od strane CA
        // te će tako dobiveni certifikat biti proslijeđen drugim klijentima.
        public byte[] generateCertificate() {
                // dummy implementation
                return new byte[0];
        }

        // TODO: Metoda `receiveCertificate` verificira certifikat od `CA` i
        // sprema informacije o klijentu.

        // Argumenti:
        // cert_data --- certifikacijski objekt
        // cert_sig  --- digitalni potpis od `cert_data`

        // Returns:

        // Metoda prima certifikat --- certifikacijski objekt koji sadrži
        // inicijalni Diffie-Hellman javni ključ i ime klijenta s kojim želi
        // komunicirati te njegov potpis. Certifikat se verificira pomoću javnog
        // ključa CA (Certificate Authority), a ako verifikacija uspije,
        // informacije o klijentu (ime i javni ključ) se pohranjuju. Javni ključ
        // CA je spremljen tijekom inicijalizacije objekta.

        // U slučaju da verifikacija ne prođe uspješno, potrebno je baciti iznimku.
        public void receiveCertificate(byte[] cert_data, byte[] cert_sig) {
                // needs implementation
        }

        // TODO: Metoda `sendMessage` šalje kriptiranu poruku `message` i
        //       odgovarajuće zaglavlje korisniku `username`.

        // Argumenti:
        // message  --- poruka koju ćemo poslati
        // username --- korisnik kojem šaljemo poruku

        // returns: zaglavlje i sifrat poruke

        // Zaglavlje poruke treba sadržavati podatke potrebne
        // 1) klijentu da derivira nove ključeve i dekriptira poruku;
        // 2) Velikom Bratu da dekriptira `sending` ključ i dode do sadržaja poruke.

        // Pretpostavite da već posjedujete certifikacijski objekt klijenta
        // (dobiven pomoću metode `receive_certificate`) i da klijent posjeduje
        // vaš. Ako prethodno niste komunicirali, uspostavite sesiju
        // generiranjem ključeva po- trebnih za `Double Ratchet` prema
        // specifikaciji. Inicijalni korijenski ključ (`root key` za
        // `Diffie-Hellman ratchet`) izračunajte pomoću ključa dobivenog u
        // certifikatu i vašeg inicijalnog privatnog ključa.

        // Svaka poruka se sastoji od sadržaja i zaglavlja. Svaki put kada
        // šaljete poruku napravite korak u lancu `symmetric-key ratchet` i
        // lancu `Diffie-Hellman ratchet` ako je potrebno prema specifikaciji
        // (ovo drugo možete napraviti i prilikom primanja poruke);
        // `Diffie-Helman ratchet` javni ključ oglasite putem zaglavlja. S novim
        // ključem za poruke (`message key`) kriptirajte i autentificirajte
        // sadržaj poruke koristeći simetrični kriptosustav AES-GCM;
        // inicijalizacijski vektor proslijedite putem zaglavlja. Dodatno,
        // autentificirajte odgovarajuća polja iz zaglavlja, prema
        // specifikaciji.

        // Sve poruke se trebaju moći dekriptirati uz pomoć privatnog kljuca od
        // Velikog brata; pripadni javni ključ dobiti ćete prilikom
        // inicijalizacije klijenta. U tu svrhu koristite protokol enkripcije
        // `ElGamal` tako da, umjesto množenja, `sending key` (tj. `message
        // key`) kriptirate pomoću AES-GCM uz pomoć javnog ključa od Velikog
        // Brata. Prema tome, neka zaglavlje dodatno sadržava polja `govPubKey`
        // (`ephemeral key`) i `govCt` (`ciphertext`) koja predstavljaju izlaz
        // `(k_E , y)` kriptosustava javnog kljuca `Elgamal` te `govIv` kao
        // pripadni inicijalizacijski vektor.

        // U ovu svrhu proučite `Elgamal Encryption Protocol` u udžbeniku
        // `Understanding Cryptography` (glavna literatura). Takoder, pročitajte
        // dokumentaciju funkcije `govDecrypt`.

        // Metoda treba vratiti zaglavlje i kriptirani sadrzaj poruke.

        public byte[] sendMessage(String peerUsername, String message) throws GeneralSecurityException {
                // dummy implementation
                return message.getBytes(Charset.defaultCharset());
        }

        // TODO: Metoda `receiveMessage` prima poruku od korisnika

        // Argumenti:
        // message  -- poruka koju smo primili
        // username -- korisnik koji je poslao poruku

        // returns: sadrzaj nakon dekriptiranja poruke `message`

        // Metoda prima kriptiranu poruku od korisnika s imenom `username`.
        // Pretpostavite da već posjedujete certifikacijski objekt od korisnika
        // (dobiven pomoću `receive_certificate`) i da je korisnik izračunao
        // inicijalni `root` ključ uz pomoć javnog Diffie-Hellman ključa iz
        // vašeg certifikata.  Ako već prije niste komunicirali, uspostavite
        // sesiju tako da generirate nužne `double ratchet` ključeve prema
        // specifikaciji.

        // Svaki put kada primite poruku napravite `ratchet` korak u `receiving`
        // lanacu (i `root` lanacu ako je potrebno prema specifikaciji)
        // koristeći informacije dostupne u zaglavlju i dekriptirajte poruku uz
        // pomoć novog `receiving` ključa. Ako detektirate da je integritet
        // poruke narušen, zaustavite izvršavanje programa i generirajte
        // iznimku.

        // Metoda treba vratiti dekriptiranu poruku.
        public String receiveMessage(String peerUsername, byte[] message) throws GeneralSecurityException {
                // dummy implementation
                return new String(message, Charset.defaultCharset());
        }
}
