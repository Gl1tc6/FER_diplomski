import random

# Baza pitanja iz dokumenta
# Svako pitanje je rječnik s tekstom pitanja, opcijama i točnim odgovorom.
baza_pitanja = [
    {
        "pitanje": "Organizirani prema političkoj pripadnosti, a ne prema nacionalnosti, zastupnici u Europskom parlamentu djeluju unutar klubova zastupnika. Koliko trenutačno postoji klubova zastupnika u Europskom parlamentu?",
        "opcije": {"a": "8", "b": "6", "c": "9", "d": "5"},
        "tocan": "a"
    },
    {
        "pitanje": "U Republici Hrvatskoj je smještena i djeluje barem jedna agencija Europske unije. Da li je ova tvrdnja ispravna?",
        "opcije": {"a": "Točno", "b": "Netočno"},
        "tocan": "b"
    },
    {
        "pitanje": "Roberta Metsola zastupnica iz Malte izabrana je u siječnju 2022. godine za predsjednicu Europskog parlamenta. Od 1979. godine kad su održani prvi izravni izbori za Europski parlament koliko je bilo žena na toj čelnoj funkciji, uključujući sadašnju predsjednicu Metsolu?",
        "opcije": {"a": "1", "b": "4", "c": "3", "d": "2"},
        "tocan": "d"
    },
    {
        "pitanje": "Direktiva kao zakonodavni obvezujući akt je?",
        "opcije": {"a": "Tercijarni izvor prava EU", "b": "Sekundarni izvor prava EU", "c": "Primarni izvor prava EU"},
        "tocan": "b"
    },
    {
        "pitanje": "Godišnji proračun Europske unije predlaže:",
        "opcije": {"a": "Europsko vijeće", "b": "Europska komisija", "c": "Vijeće Europske unije", "d": "Europski parlament"},
        "tocan": "b"
    },
    {
        "pitanje": "Najviša razina političke suradnje između država članica Europske unije se ostvaruje u?",
        "opcije": {"a": "Vijeću Europske unije", "b": "Europskom parlamentu", "c": "Vijeću Europe", "d": "Europskom vijeću"},
        "tocan": "d"
    },
    {
        "pitanje": "Višegodišnji financijski okvir Europske unije donosi se za razdoblje od:",
        "opcije": {"a": "Pet godina", "b": "Deset godina", "c": "Sedam godina", "d": "Tri godine"},
        "tocan": "c"
    },
    {
        "pitanje": "Političko vodstvo Europske komisije čini kolegij od koliko povjerenika?",
        "opcije": {"a": "27", "b": "28", "c": "15", "d": "12"},
        "tocan": "a"
    },
    {
        "pitanje": "Kako se zove ugovor kojim je formalno osnovana Europska unija?",
        "opcije": {"a": "Ugovor iz Maastrichta", "b": "Ugovor iz Lisabona", "c": "Ugovor iz Nice", "d": "Ugovor iz Amsterdama"},
        "tocan": "a"
    },
    {
        "pitanje": "Zastupnici u Europskom parlamentu djeluju u organizacijskim jedinicama koje su u svom radu nadležne za pojedine javne politike Europske unije. Kako se zovu te organizacijske jedinice?",
        "opcije": {"a": "Delegacije", "b": "Klubovi", "c": "Međuodbori", "d": "Odbori", "e": "Koordinacije"},
        "tocan": "d"
    },
    {
        "pitanje": "Koliko je zastupnika iz Republike Hrvatske u Europskom parlamentu u mandatnom razdoblju 2019.-2024. (i 2024.-2029.)?",
        "opcije": {"a": "10", "b": "12", "c": "11"},
        "tocan": "b"
    },
    {
        "pitanje": "Koja od navedenih institucija je savjetodavno tijelo Europske unije?",
        "opcije": {"a": "Europska udruga regionalnih energetskih agencija", "b": "Europski ombudsman", "c": "Europski odbor regija", "d": "Europska udruga gradova"},
        "tocan": "c"
    },
    {
        "pitanje": "Političko vodstvo Europske komisije se naziva Kolegij i na njegovom čelu je u mandatu 2019.-2024. predsjednica Komisije Ursula von der Leyen. Gospođa von der Leyen je prva žena koja se nalazi na čelnoj funkciji Europske komisije. Da li je ova činjenica istinita?",
        "opcije": {"a": "Točno", "b": "Netočno"},
        "tocan": "a"
    },
    {
        "pitanje": "Koji zakonodavni akt je obvezujući u cijelosti i neposredno se primjenjuje u svim državama članicama?",
        "opcije": {"a": "Odluka", "b": "Uredba", "c": "Direktiva", "d": "Komunikacija"},
        "tocan": "b"
    },
    {
        "pitanje": "Izvršne agencije EU se osnivaju na ograničeno razdoblje u svrhu upravljanja posebnim zadaćama povezanima s provedbom programa EU. Osnivaju se odlukom koje institucije EU?",
        "opcije": {"a": "Europske komisije", "b": "Europskog revizorskog suda", "c": "Europskog parlamenta", "d": "Europskog vijeća", "e": "Vijeća Europske unije"},
        "tocan": "a"
    },
    {
        "pitanje": "Koliko zastupnika ima Europski parlament nakon izlaska Velike Britanije iz EU?",
        "opcije": {"a": "705", "b": "507", "c": "336", "d": "751", "e": "735"},
        "tocan": "a"
    },
    {
        "pitanje": "Europski revizorski sud provodi financijsku reviziju koje institucije?",
        "opcije": {"a": "Europske komisije", "b": "Europskog parlamenta", "c": "Vijeća EU", "d": "Suda Europske unije"},
        "tocan": "a"
    },
    {
        "pitanje": "Koja institucija upravlja proračunom Europske unije?",
        "opcije": {"a": "Europski parlament", "b": "Vijeće Europske unije", "c": "Europska komisija", "d": "Europski revizorski sud"},
        "tocan": "c"
    },
    {
        "pitanje": "Vijeće Europske unije ima sastave koji raspravljaju o određenim područjima politika. Vijeće se sastaje u koliko sastava?",
        "opcije": {"a": "27", "b": "5", "c": "10", "d": "8"},
        "tocan": "c"
    },
    {
        "pitanje": "Decentralizirane agencije EU su:",
        "opcije": {"a": "Povremene agencije EU", "b": "Nacionalne agencije u većinskom vlasništvu EU", "c": "Stalne agencije EU (osnovane na neodređeno vrijeme)"},
        "tocan": "c"
    },
    {
        "pitanje": "Sustav glasovanja u Vijeću koji se naziva kvalificirana većina definiran je kao:",
        "opcije": {"a": "55% država članica koje predstavljaju najmanje 65% stanovništva EU-a", "b": "65% država članica", "c": "Države članice koje predstavljaju najmanje 55% stanovništva", "d": "55% država članica"},
        "tocan": "a"
    },
    {
        "pitanje": "Koja institucija EU može izvršiti istragu, kazneni progon i iznošenje presude za zločine protiv financijskih interesa Europske unije?",
        "opcije": {"a": "Europski ombudsman", "b": "Sud Europske unije", "c": "Europski revizorski sud", "d": "Europski javni tužitelj"},
        "tocan": "d"
    },
    {
        "pitanje": "Europski gospodarski i socijalni odbor savjetodavno je tijelo EU-a koje čine predstavnici radnika, poslodavaca i...",
        "opcije": {"a": "Predstavnika studenata i profesora", "b": "Predstavnika sindikata i nevladinih organizacija", "c": "Predstavnika regija EU", "d": "Predstavnika državnih službenika"},
        "tocan": "b"
    },
    {
        "pitanje": "Koji od navedenih poreza je izvor prihoda za proračun Europske unije?",
        "opcije": {"a": "Korporativni porez na dobit", "b": "Trošarine", "c": "Dio poreza na dohodak", "d": "Porez na dodanu vrijednost"},
        "tocan": "d"
    },
    {
        "pitanje": "Kako se nazivaju radna tijela Europske komisije?",
        "opcije": {"a": "Sastavi", "b": "Glavni sektori", "c": "Odjeli", "d": "Odbori", "e": "Glavne (opće) uprave"},
        "tocan": "e"
    },
    {
        "pitanje": "U kojoj instituciji EU Predsjednik Vlade predstavlja Republiku Hrvatsku?",
        "opcije": {"a": "Europskom vijeću", "b": "Vijeću Europske unije", "c": "Europskom parlamentu", "d": "Europskoj komisiji"},
        "tocan": "a"
    },
    {
        "pitanje": "Sveobuhvatni paket javnih politika EU kojim se želi postići zelena tranzicija društva i gospodarstva Europske unije s krajnjim ciljem postizanja klimatske neutralnosti do 2050. godine zove se:",
        "opcije": {"a": "Energetska unija", "b": "InvestEU", "c": "Next Generation EU", "d": "Europski zeleni plan"},
        "tocan": "d"
    },
    {
        "pitanje": "Na koji datum je Republika Hrvatska postala članica Europske unije?",
        "opcije": {"a": "1. srpnja 2011.", "b": "1. siječnja 2013.", "c": "1. srpnja 2013.", "d": "1. srpnja 2014."},
        "tocan": "c"
    },
    {
        "pitanje": "Ugovorom iz Lisabona Europski parlament je dobio veće ovlasti i zajedno s kojom institucijom Europske unije je zadužen za donošenje zakonodavstva EU?",
        "opcije": {"a": "Vijeće Europske unije", "b": "Europska komisija", "c": "Vijeće Europe", "d": "Europsko vijeće", "e": "Europski odbor regija"},
        "tocan": "a"
    },
    {
        "pitanje": "Prema Ugovoru iz Lisabona koja institucija Europske unije predlaže nove propise?",
        "opcije": {"a": "Europski parlament", "b": "Europska komisija", "c": "Vijeće Europske unije", "d": "Europsko vijeće", "e": "Sud Europske unije"},
        "tocan": "b"
    },
    {
        "pitanje": "Hrvatski član/članica u Europskom revizorskom sudu je:",
        "opcije": {"a": "Karlo Ressler", "b": "Tonino Picula", "c": "Biljana Borzan", "d": "Dubravka Šuica", "e": "Ivana Maletić"},
        "tocan": "e"
    },
    {
        "pitanje": "Što od navedenog ne spada pod neobvezujuće akte?",
        "opcije": {"a": "Rezolucije", "b": "Direktiva", "c": "Deklaracija", "d": "Preporuke i mišljenja"},
        "tocan": "b"
    },
    {
        "pitanje": "Što ne spada među ovlasti Europskog parlamenta?",
        "opcije": {"a": "Zakonodavne ovlasti", "b": "Ovlasti u upravljanju proračunom", "c": "Ovlasti miješanja u pitanja iz nadležnosti država članica", "d": "Proračunske ovlasti", "e": "Nadzorne ovlasti"},
        "tocan": "c"
    },
    {
        "pitanje": "Tko u Hrvatskoj ne predlaže kandidate za članove/zamjenike Europskog odbora regija?",
        "opcije": {"a": "Sabor Republike Hrvatske", "b": "Hrvatske općine i gradovi", "c": "Udruga gradova u RH", "d": "Udruga općina u RH"},
        "tocan": "a"
    },
    {
        "pitanje": "Članovi Europskog revizorskog suda imenovani su na mandat od:",
        "opcije": {"a": "4 godine s mogućnošću obnavljanja", "b": "6 godina bez mogućnosti obnavljanja", "c": "6 godina s mogućnošću obnavljanja", "d": "4 godine bez mogućnosti obnavljanja"},
        "tocan": "c"
    },
    {
        "pitanje": "Tko u najvećem dijelu upravlja izvršavanjem proračuna Europske unije?",
        "opcije": {"a": "Europska komisija", "b": "Države članice", "c": "Međunarodne organizacije", "d": "Europsko vijeće"},
        "tocan": "b"
    },
    {
        "pitanje": "Tko do kraja 2024. godine predsjeda Vijećem Europske unije?",
        "opcije": {"a": "Poljska", "b": "Hrvatska", "c": "Belgija", "d": "Mađarska"},
        "tocan": "d"
    },
    {
        "pitanje": "Potpredsjednica Europske komisije Dubravka Šuica zadužena je za portfelj:",
        "opcije": {"a": "Održivi razvoj i turizam", "b": "Okoliš", "c": "Demokracija, pravednost, vladavina prava i zaštita potrošača", "d": "Mediteran", "e": "Ribarstvo i oceane"},
        "tocan": "c"
    },
    {
        "pitanje": "Za koliko portfelja je odgovoran svaki od povjerenika Europske komisije?",
        "opcije": {"a": "1", "b": "Za koliko god se odluči predsjednica", "c": "Niti jedan", "d": "2"},
        "tocan": "a"
    },
    {
        "pitanje": "Primarni izvori prava NISU definirani:",
        "opcije": {"a": "Poveljom EU o temeljnim pravima", "b": "Ugovorom o funkcioniranju EU", "c": "Pravnim aktima koje donose institucije Europske unije", "d": "Ugovorom o EU"},
        "tocan": "c"
    },
    {
        "pitanje": "Za dugoročni proračun EU 2021.-2027., koji od navedenih novih izvora prihoda je implementiran do listopada 2023.?",
        "opcije": {"a": "Porez na financijske transakcije", "b": "Porez na digitalne usluge", "c": "Nacionalni doprinos od nerecikliranog plastičnog ambalažnog otpada", "d": "Korporativni porez na dobit"},
        "tocan": "c"
    },
    {
        "pitanje": "Što od navedenog nije vrsta agencije Europske unije?",
        "opcije": {"a": "Izvršne agencije", "b": "Agencije i tijela EUROATOM", "c": "Agencije europskog odbora regija", "d": "Decentralizirane agencije"},
        "tocan": "c"
    },
    {
        "pitanje": "Zajednička poduzeća u EU (JPP) obično uključuju suradnju između Vlada i koga?",
        "opcije": {"a": "Nevladinih organizacija", "b": "Europske komisije", "c": "Istraživačkih institucija i privatnih tvrtki", "d": "Savjetodavnih tijela"},
        "tocan": "c"
    },
    {
        "pitanje": "Koji od sljedećih klubova zastupnika ne vežemo uz Europski parlament?",
        "opcije": {"a": "The Left", "b": "S&D", "c": "Renew Europe", "d": "EBC: Klub zastupnika za ekonomski boljitak", "e": "ECR"},
        "tocan": "d"
    }
]

def pokreni_kviz():
    random.shuffle(baza_pitanja) # Izmiješaj pitanja
    bodovi = 0
    ukupno_pitanja = len(baza_pitanja)

    print(f"Dobrodošli u kviz o Europskoj uniji! Imate {ukupno_pitanja} pitanja.\n")

    for i, podaci in enumerate(baza_pitanja, 1):
        print(f"Pitanje {i}/{ukupno_pitanja}:")
        print(podaci["pitanje"])
        
        # Ispis opcija
        for oznaka, tekst in sorted(podaci["opcije"].items()):
            print(f"  {oznaka}) {tekst}")
        
        # Unos odgovora
        odgovor = input("\nVaš odgovor (a/b/c/d...): ").lower().strip()
        
        # Provjera
        if odgovor == podaci["tocan"]:
            print("✅ Točno!\n")
            bodovi += 1
        else:
            tocan_tekst = podaci["opcije"][podaci["tocan"]]
            print(f"❌ Netočno. Točan odgovor je: {podaci['tocan']}) {tocan_tekst}\n")
    
    # Krajnji rezultat
    postotak = (bodovi / ukupno_pitanja) * 100
    print("-" * 30)
    print(f"Kviz završen! Vaš rezultat: {bodovi}/{ukupno_pitanja} ({postotak:.2f}%)")

if __name__ == "__main__":
    pokreni_kviz()