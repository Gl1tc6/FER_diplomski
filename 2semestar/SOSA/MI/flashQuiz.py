import random

flashcards = [
    # Izvor 01. TPM
    ("Tri namjene TPM-a", "Sigurna pohrana ključeva, mjerenje integriteta sustava tijekom pokretanja, autentifikacija uređaja i korisnika."),
    ("Kako TPM mjeri integritet podataka tijekom pokretanja?", "Kroz kriptografske sažetke koda i konfiguracije, pohranjene u PCR registre."),
    ("Kako TPM štiti kriptografske ključeve?", "Privatni ključ nije moguće pročitati; ključevi se generiraju u TPM-u i štite autorizacijom."),
    ("Kako TPM omogućuje sigurnu autentifikaciju?", "Putem attestation key (AK), koristi se za autentifikaciju uređaja/korisnika (npr. VPN, banke)."),
    ("Scenarij sprječavanja neovlaštenog pristupa s TPM-om?", "Disk je šifriran, a TPM otključava ključ samo ako je sustav validiran; USB boot ne pomaže."),

    # Izvor 02. SGX
    ("Koji problem SGX rješava?", "Nepouzdana okruženja za izvršavanje aplikacija, posebno u cloudu."),
    ("Što je SGX?", "Skup procesorskih instrukcija za stvaranje izoliranih enklava za zaštitu podataka."),
    ("Zašto SGX reverse sandbox?", "Štiti aplikaciju od OS-a i drugih komponenti, obrnuto od uobičajenog sandboxa."),
    ("Što je enklava i kako se potvrđuje?", "Izolirani kontejner; ispravnost se potvrđuje udaljenom atestacijom uz measurement hash."),
    ("Što je udaljena atestacija i čemu služi?", "Dokazivanje da SGX enklava sigurno izvršava kod, povećava povjerenje."),

    # Izvor 03. Android
    ("Principi Android sigurnosnog modela?", "Ograničavanje Intents, OWASP testiranja, scoped storage."),
    ("Scoped storage i razlog uvođenja?", "Aplikacije pristupaju samo vlastitim direktorijima; povećava privatnost."),
    ("Kako Android štiti komunikaciju između komponenti?", "Ograničavanje Intents, izbjegavanje implicitnih, validacija."),
    ("Glavni sigurnosni izazovi Androida?", "Statičke dozvole i manjak kontrole protoka podataka."),
    ("Prijetnje Android aplikacijama?", "Rizici od pristupa datotekama, rukovanja Intents, zlouporaba dozvola."),

    # Izvor 04. Docker
    ("Kako Docker izolira procese među kontejnerima?", "Kroz imenske prostore (namespaces) procesa."),
    ("Uloga nodev pri montiranju?", "Sprječava stvaranje uređajskih datoteka u kontejneru."),
    ("Kako se ostvaruje mrežna veza među kontejnerima?", "Preko bridge sučelja; bez filtriranja paketa, ARP/MAC napadi mogući."),
    ("Dodatni sustavi za zaštitu jezgre?", "Linux capabilities i LSM (AppArmour, SELinux)."),
    ("Kako radi AppArmour?", "Profili aplikacija s ograničenim mogućnostima; enforcement i complain modovi."),

    # Izvor 05. Modeliranje prijetnji
    ("Što je modeliranje prijetnji?", "Analiza potencijalnih napada i prijetnji, strukturirani pristup."),
    ("Zašto se koristi modeliranje prijetnji?", "Prepoznavanje ranjivosti prije produkcije."),
    ("Zadaci stručnjaka za modeliranje prijetnji?", "Vodi projekt, upoznaje dionike, dijeli sigurnosne rizike."),
    ("Koraci projekta modeliranja prijetnji?", "Ciljevi, model sustava, identifikacija i analiza prijetnji."),
    ("O čemu govori manifest modeliranja prijetnji?", "Kultura učenja, suradnja, aktivno modeliranje, stalno poboljšanje."),

    # Izvor 06. STRIDE
    ("Što je STRIDE?", "Model s 6 kategorija prijetnji: Spoofing, Tampering, Repudiation, Information Disclosure, DoS, EoP."),
    ("Koraci STRIDE modeliranja?", "Dizajn sustava, identifikacija prijetnji, implementacija zaštita."),
    ("Prednosti STRIDE-a?", "Strukturirana analiza, primjenjiva u svim fazama razvoja."),
    ("Zaštita od Spoofing i DoS napada?", "Autentifikacija za Spoofing; ograničenje zahtjeva, DDoS zaštita za DoS."),
    ("Uloga granica povjerenja u DFD?", "Označuju prijelaze između pouzdanih i nepouzdanih komponenti, važno za identifikaciju prijetnji."),

    # Izvor 07. Stabla napada
    ("Stablo napada vs. graf napada?", "Stablo je hijerarhija bez ciklusa; graf ima cikluse, više roditelja."),
    ("Prednosti stabla napada?", "Više ciljeva, preglednost, proširivost, obuhvatnost."),
    ("Struktura stabla napada?", "Korijen (cilj), grane (putovi), lišće (preduvjeti), veze 'ili' i 'i'."),
    ("Kako definirati metrike napada?", "Booleove ili kontinuirane vrijednosti za analizu putova i usporedbu."),
    
    # Izvor 08. Dizajn arhitekture s naglaskom na sigurnost
    ("Tri sigurnosna zahtjeva", "Autentifikacija, Autorizacija, Povjerljivost"),
    ("Važnost sigurnosti na početku SDLC-a", "Cijena popravka ranjivosti raste s napretkom sustava. Razmišljanje o sigurnosti rano smanjuje rizik i troškove budućih izmjena"),
    ("Razlika između bug-a i ranjivosti u arhitekturi", "Bug je greška u kodu koja ne mora biti vezana za sigurnost i popravlja se izmjenom koda. Ranjivost u arhitekturi je loša odluka u dizajnu sustava koja narušava sigurnost i zahtijeva redizajn dijela sustava"),
    ("Prednosti i mane korištenja eksternih komponenti", "Prednosti: Jeftiniji i brži razvoj.\nMane: Povećava se broj mjesta gdje napadači mogu pokušati kompromitirati sustav, zahtijeva validaciju, testiranje i praćenje promjena"),
    ("Tactic-Oriented Architectural Analysis (ToAA)","n/a"),
    
    # Izvor 09. Sigurnost mikroservisne arhitekture
    ("Veći sigurnosni izazovi mikroservisa", "Zbog distribuirane prirode sustava (više vanjskih sučelja), potrebe za autentifikacijom između servisa, ranjivosti komunikacije među servisima na presretanje, teže analize zapisa i detekcije anomalija, i mogućnosti napada na različitim razinama (mrežnoj, servisnoj, kontejnerskoj, orkestracijskoj, razini otkrivanja servisa)"),
    ("API Gateway", "Jedina ulazna točka u sustav mikroservisa. Značajan je za sigurnost jer centralizira proces autentifikacije, validira zahtjeve i ograničava njihov broj, čime smanjuje površinu napada"),
    ("Metode autentifikacije u mikroservisima", " OAuth 2.0 i OpenID Connect kao standardi, JWT (JSON Web Token) za sigurno upravljanje sesijama"),
    ("Princip najmanjih privilegija", "Korisnici ili procesi trebaju dobiti minimalne privilegije potrebne za obavljanje svojih zadataka"),
    ("Sigurnost u DevSecOps", "Integracija sigurnosnih testova u CI/CD pipeline (SAST, DAST, SCA), skeniranje kontejnera za ranjivosti, automatizacija sigurnosnih provjera u svakom koraku razvoja, i korištenje IaC (Infrastructure as Code) za sigurnu konfiguraciju"),
    
    
    # Izvor 11. Programski jezik Rust
    ("Vrste grešaka od kojih Rust štiti", "Rust štiti od grešaka vezanih uz nepravilno korištenje memorije kao što su use-after-free, double free, preljev spremnika, korištenje neinicijaliziranih varijabli, korištenje varijabli nepodudarajućih tipova, i podatkovne utrke (race condition)"),
    ("Situacije u kojima je Rust dobar odabir", "Kada je potreban presjek sigurnosti i performansi, za ugradbena računala, operacijske sustave, kritičnu opremu (medicinsku, svemirsku), i za učenje novog pogleda na programiranje"),
    ("Vlasništvo (Ownership) u Rustu", "Vlasništvo je svojstvo varijable nad memorijom. Postoji samo jedan vlasnik memorije u danom trenutku. Vlasništvo se prebacuje kada nova varijabla pokazuje na tu memoriju, a stari vlasnik se invalidira. Memorija se dealocira kada vlasnik izađe iz dosega"),
    ("Dozvole varijabli i referenci unutar borrow checkera", "Borrow checker internno koristi sustav od tri dozvole: čitanje (R), pisanje (W), i vlasništvo (O). Pravila uključuju: više nemutabilnih referenci (R) je dozvoljeno, samo jedna mutabilna referenca (W) je dozvoljena, i mutabilna referenca ne može postojati istovremeno s nemutabilnim referencama"),
    ("Što se događa s varijablom pri slanju izmjenjive reference", "Kada se izmjenjiva referenca (mutable reference) pošalje u funkciju, originalna varijabla gubi sve dozvole (RWO) dok referenca postoji. Kada referenca izađe iz dosega, originalna varijabla ponovno dobiva sve dozvole"),
    
]

def quiz():
    print("Dobrodošli u Flashcards kviz! (Za izlaz upiši 'exit')\n")
    random.shuffle(flashcards)
    for question, answer in flashcards:
        print(f"Pitanje: {question}")
        input("Pritisni Enter za odgovor...")
        print(f"Odgovor: {answer}\n")
        cont = input("Pritisni Enter za sljedeće pitanje ili 'exit' za izlaz: ")
        if cont.lower() == "exit":
            break
    print("Kraj kviza! Sretno na ispitu!")

if __name__ == "__main__":
    quiz()
