import random
import time

class Flashcards:
    def __init__(self):
        self.questions = [
            {
                "question": "Prema Benthamu izbaci ono što se ne koristi za račun sreće?",
                "answer": "Racionalnost"
            },
            {
                "question": "Koje pravilo slijedi načelo 'bolje spriječiti nego liječiti'?",
                "answer": "Predostrožnost"
            },
            {
                "question": "Tko bi trebao odlučivati o etičkim postavkama autonomnih vozila osim korisnika i države?",
                "answer": "Proizvođač vozila"
            },
            {
                "question": "Što je rekao Ross da koristimo za spoznaju moralnih dužnosti?",
                "answer": "Intuiciju"
            },
            {
                "question": "Koji utilitarizam se protivi utilitarizmu postupaka?",
                "answer": "Utilitarizam pravila"
            },
            {
                "question": "Prema utilitaristima koji kriterij mora biti zadovoljen da bi biće imalo moralni status?",
                "answer": "Osjećajnost - sva bića koja osjećaju ugodu i bol"
            },
            {
                "question": "Ruth Benedikt poistovjećuje moralnost s čime?",
                "answer": "Stvar navike"
            },
            {
                "question": "Kako se zove sposobnost prepoznavanja mentalnih stanja ljudi?",
                "answer": "Teorija uma"
            },
            {
                "question": "Kako je Aristotel definirao etičke vrline?",
                "answer": "Sredina između dvaju poroka"
            },
            {
                "question": "Kod problema mnogih ruku se raspršuje?",
                "answer": "Odgovornost"
            },
            {
                "question": "Aristotel je smatrao da se etičke vrline stječu?",
                "answer": "Navikavanjem, Poukom"
            },
            {
                "question": "Kod Kanta moralni zakon važi za?",
                "answer": "Sva umna bića"
            },
            {
                "question": "'Pojmovi moralnog napretka i moralne istine postaju nesuvišli'?",
                "answer": "Etički relativizam"
            },
            {
                "question": "Tko je zagovarao etički intelektualizam?",
                "answer": "Sokrat"
            },
            {
                "question": "Mill je rekao 'Bolje biti nezadovoljan Sokrat (ljudsko biće) nego zadovoljna budala (svinja)'. Što to znači?",
                "answer": "Ako budala ima drugačije mišljenje to je zato što zna samo svoju stranu. Dok Sokrat poznaje obje strane"
            },
            {
                "question": "Što je Millov stručni sudac iskusio, a budala ne?",
                "answer": "Obje strane iz izreke 'Bolje biti nezadovoljan Sokrat (ljudsko bice) nego zadovoljna budala (svinja)', a budala samo jednu"
            },
            {
                "question": "Inicijative protiv viška čega u robota zagovaraju Britanski institut za standardizaciju, Rathenau Institut i IEEE?",
                "answer": "Antropomorfizacije"
            },
            {
                "question": "Što je Kantov kategorički imperativ?",
                "answer": "Formula općeg zakona, formula zakona prirode, formula svrhe o sebi"
            },
            {
                "question": "Scenarij tempirane bombe se odnosi na kršenje čega?",
                "answer": "Ljudskih prava"
            },
            {
                "question": "Po jedan argument za i protiv korištenja robotskih ljubavnika?",
                "answer": "Za: bolji ljubavni život invalida, zatvorenika; protiv: poticanje loših seksualnih nagona i ponašanja (pedofilija, silovanja)"
            },
            {
                "question": "Osim aktivne i pasivne, kakva još podjela eutanazije postoji (obzirom na pristanak)?",
                "answer": "Dobrovoljna i ne-dobrovoljna"
            },
            {
                "question": "Što je navodno rasistički napravio Robert Moses?",
                "answer": "Nekoliko nadvožnjaka koje je projektirao bili su preniski da bi ispod njih mogli proći autobusi, a u njima su putovali Afroamerikanci i siromašni, dok su bogati imali vlastite automobile"
            },
            {
                "question": "Što znači grčka riječ deon na kojoj se bazira deontološka etika?",
                "answer": "Dužnost"
            },
            {
                "question": "Koju teoriju etike su zagovarali Hobbes i Rousseau?",
                "answer": "Kontraktualizam"
            },
            {
                "question": "Problemi koje teorije etike su nepostojanje moralne odgovornosti i slabost volje?",
                "answer": "Etički intelektualizam"
            },
            {
                "question": "Kako se zove sposobnost prepoznavanja mentalnih stanja ljudi, poput vjerovanja, osjecaja, namjera I zelja, koju bi trebali imati socijalni roboti?",
                "answer": "Teorija uma"
            },
            {
                "question": "Što je osnovno načelo negativnog utilitarizma?",
                "answer": "Najbolji postupak je onaj koji stvara najmanju količinu nesreće općenito"
            },
            {
                "question": "Osim sto Hansson kaze da nepravednost u tehonoloskom kontekstu cini razlika u pristupu tehnologije medu pojedincima I zemljama, za sto jos on misli da uzrokuje tehnolosku nepravednost?",
                "answer": "Sama tehnologija moze stvoriti I odrzavati……"
            },
            {
                "question": "Zaokruzi uljeza medu navedenim stvarima kojima se bavi distributicna pravednost?",
                "answer": "Smrtna kazna"
            },
            {
                "question": "Nejednakost različitih skupina ljudi s obzirom na pristup digitalnim, informacijskim tehnologijama u literaturi se naziva?",
                "answer": "Digitalna razdjelnica"
            },
            {
                "question": "Ako je Laissez-faire kapitalizam najbolji oblik ekonomske organizacije za moralno nesavršeno stanje ljudi, koji je oblik ekonomske organizacije najprikladniji za moralno usavršeno stanje ljudi?",
                "answer": "Socijalizam"
            },
            {
                "question": "Ekonomija za koju se zagovarao Marx je?",
                "answer": "Planska ekonomija"
            },
            {
                "question": "Što je Marxu smetalo kod kapitalističkog tržišta?",
                "answer": "Hiperprodukcija, otuđenost, specijaliziranost, rasipnost"
            },
            {
                "question": "Po Nozicku, kojim načinom se ne smije stjecati vlasništvo?",
                "answer": "Prevarom"
            },
            {
                "question": "Prema Nozicku kojim načinom se treba stjecati vlasništvo?",
                "answer": "Pravedno (NE PREVAROM)"
            },
            {
                "question": "Nozick misli da Chamberlain na kraju godine duguje društvu koliko svojeg prihoda od prodaje ulaznica za utakmicu?",
                "answer": "Ništa"
            },
            {
                "question": "Koju tradiciju odbacuje Rawls u svojoj socijalno-liberalnoj teoriji?",
                "answer": "Utilitarizam"
            },
            {
                "question": "Pojedinci koji u Rawlsovoj koncepciji odlučuju o načelima pravednosti ne znaju svoje mjesto u društvu i svoje prirodne talente. Zato kažemo da se načela pravednosti biraju iza?",
                "answer": "Vela neznanja"
            },
            {
                "question": "Po Rawlsu, što pojedinci na originalnoj poziciji iza vela neznanja ne znaju?",
                "answer": "Ne znaju svoje mjesto u društvu, svoje sposobnosti, snagu itd."
            },
            {
                "question": "Kako se zove stručna i kvalificirana radna snaga, s administrativnim ili uredskim poslom, višeg društvenog položaja i primanja?",
                "answer": "Bijeli ovratnici"
            },
            {
                "question": "Nestručna i nekvalificirana radna snaga; uglavnom fizički rad; niži društveni položaj i primanja nazivaju se?",
                "answer": "Plavi ovratnici"
            },
            {
                "question": "Kako se zove problem koji nastaje uslijed prevelikog i naglog gubitka poslova razvojem AI?",
                "answer": "Tranzicijski troškovi"
            },
            {
                "question": "Troškovi prekvalifikacije ili zbrinjavanja ljudi koji se neće na vrijeme prilagoditi tehnološki izmijenjenim okolnostima na tržištu rada su?",
                "answer": "Tranzicijski troškovi"
            },
            {
                "question": "Prijedlog rješenja problema 'tranzicijskih troškova'?",
                "answer": "Porez na robote"
            },
            {
                "question": "Prema Humeovom zakonu iz premisa koje opisuju činjenice ne može se na deduktivno valjan način izvesti kakva konkluzija?",
                "answer": "Normativna"
            },
            {
                "question": "Kako se zove kada nakon dugo vremena se počne podilaziti ugroženoj i ugnjetavanoj podskupini ljudi?",
                "answer": "Obrnuta diskriminacija"
            },
            {
                "question": "Ako je pseudonimizacija reverzibilan postupak zaštite podataka, kako se zove ireverzibilan?",
                "answer": "Anonimizacija"
            },
            {
                "question": "Ako je anonimizacija ireverzibilan postupak uklanjanja identifikacijskih informacija iz podataka kojim se osobni podaci pretvaraju u podatke koji ne mogu identificirati ispitanika, kako se zove reverzibilan postupak uklanjanja identifikacijskih informacija?",
                "answer": "Pseudonimizacija"
            },
            {
                "question": "U slučaju Loomis koristio se algoritam?",
                "answer": "COMPAS"
            },
            {
                "question": "'Arhitektura izbora' i 'nudge' su pojmovi koji se povezuju s libertarijanskim?",
                "answer": "Paternalizmom"
            },
            {
                "question": "Kako se zove automaton koji je izgubio bitku protiv Jazona i argonauta?",
                "answer": "Talos"
            },
            {
                "question": "Koja dva izma govore da je pravedni rat besmislen?",
                "answer": "Pacifizam i realizam"
            },
            {
                "question": "Jus in bello i Jus ad bellum oboje imaju?",
                "answer": "Razmjernost"
            },
            {
                "question": "Objasni razmjernost kod pravednosti u ratu?",
                "answer": "Svaka prouzročena šteta ili zlo moraju biti razmjerni vojnom cilju koji je ostvaren"
            },
            {
                "question": "Jus ad bellum?",
                "answer": "Pod kojim uvjetima je opravdano pribjegavanje ratovanju"
            },
            {
                "question": "Jus in bello?",
                "answer": "Kako moralno postupati tijekom ratovanja"
            },
            {
                "question": "Bostrom definira za postčovjeka potrebno je koliko postljudskih sposobnosti?",
                "answer": "Barem jednu"
            },
            {
                "question": "Ako je biopoboljšanje upotreba biotehničke moći da bi se poboljšale prirodene sposobnosti ljudi, kako se zove upotreba biotehničke moći kako bi se pojedince vratilo u normalno stanje zdravlja i sposobnosti?",
                "answer": "Liječenje"
            },
            {
                "question": "Navedi barem 2 vrste (bio)poboljšanja?",
                "answer": "Tjelesno poboljšanje, kognitivno poboljšanje, moralno poboljšanje, poboljšanje raspoloženja, produljenje životnog vijeka"
            },
            {
                "question": "Michael Sandel se protivi poboljšanju zbog brojčanih razloga, ali ne zbog gubitka?",
                "answer": "Znanosti"
            },
            {
                "question": "Prema utilitaristima koji kriterij mora biti zadovoljen da bi bice imalo moralni status?",
                "answer": "Osjećajnost"
            },
            {
                "question": "Agar progovara da nam nije u interesu stvarati 'postosobe' zbog?",
                "answer": "Dva praga"
            },
            {
                "question": "Zlo je dopustivo ako je kao sporedni učinak i nenamjeravano?",
                "answer": "Dvostruki učinak"
            },
            {
                "question": "Što je rekao Ross da koristimo za spoznaju moralnih dužnosti u etici dužnosti prima facie?",
                "answer": "Intuiciju"
            },
            {
                "question": "Kako se zove u medicinskoj praksi kada liječnici mogu uskratiti informacije ili čak obmanjivati pacijente radi njihova vlastita dobra.",
                "answer": "Paternalizam"
            },
            {
                "question": "Što je suprotno kod Immanuela Kanta od slobode.",
                "answer": "Determinizam"
            },
            {
                "question": "Tko je napravio \"Račun sreće\" / \"hedonički račun\"?",
                "answer": "Bentham"
            },
            {
                "question": "Screnarij tempirane bombe se odnosi na krsenje cega?",
                "answer": "Ljudskih prava"
            },
            {
                "question": "Dolazak u postojanje uvijek predstavlja ozbiljnu štetu (harm) ili tako nešto.",
                "answer": "Antinatalizam"
            },
            {
                "question": "Načelo predostrožnosti (\"jaka\" AI verzija): \"problem maksimizatora?",
                "answer": "Spajalica"
            },
            {
                "question": "Kako se zove problem rasprsivanja odgovornosti ili tako nesto je bio tekst zadatka?",
                "answer": "Problem mnogih ruku"
            },
            {
                "question": "Koga je George u dilema pješačkog mosta razmišlja gurnuti da spasi nekoliko ljudi?",
                "answer": "Debelog čovjeka"
            },
            {
                "question": "Sto se dogodilo obitelji Metzlera?",
                "answer": "Otmica"
            },
            {
                "question": "Istražuje pojmove i metode same etike, ne primjenu etike.",
                "answer": "Metaetika"
            },
            {
                "question": "Kod etickih relativista \"to je stvar navike\" i \"to je drustveno prihvatljivo\" što je to kod normativne etike?",
                "answer": "Moralno dobro ili ispravno"
            },
            {
                "question": "Što je kategorični imperativ kod Immanuela Kanta?",
                "answer": "Kategorični imperativ zabranjuje neke očigledno ispravne postupke (laž kojom spašavamo život)"
            },
            {
                "question": "Koje dvije \"stvari\"(pisalo nekako drugacije) su kod \"Jus in bello\"?",
                "answer": "Razmjernost i Zaštićenost"
            }
        ]

    def welcome(self):
        print("\n===== ETIKA FLASHCARDS =====")
        print("Testirajte svoje znanje iz etike!")
        print("Za svako pitanje pritisnite ENTER za prikaz odgovora.")
        print("Za izlaz iz programa u bilo kojem trenutku unesite 'exit'.\n")

    def run_flashcards(self, num_cards=None, randomize=True):
        self.welcome()
        
        if num_cards is None or num_cards > len(self.questions):
            num_cards = len(self.questions)
        
        cards = self.questions.copy()
        if randomize:
            random.shuffle(cards)
        else:
            cards.sort(key=lambda x: x['question'])
            
        cards = cards[:num_cards]
        
        for i, card in enumerate(cards):
            print(f"\nKartica {i+1}/{num_cards}:")
            print(f"Pitanje: {card['question']}")
            
            user_input = input("\nENTER/'exit': ")
            
            if user_input.lower() == 'exit':
                print("\nIzlaz iz programa. Hvala na učenju!")
                break
            
            print(f"\nOdgovor: {card['answer']}")
            print("-----------------------------------------------------\n")
            
            # if i < num_cards - 1:
            #     user_input = input("\nENTER/'exit: ")
            #     if user_input.lower() == 'exit':
            #         print("\nIzlaz iz programa. Hvala na učenju!")
            #         break
        
        if i == num_cards - 1:
            print("\n===== KRAJ KARTICA =====")
            print("Prošli ste kroz sve kartice!")

if __name__ == "__main__":
    flashcards = Flashcards()
    
    try:
        choice = input("Želite li nasumične kartice? (da/ne): ").lower()
        randomize = choice == 'da'
        
        num_str = input("Koliko kartica želite? (pritisnite ENTER za sve): ")
        if num_str.strip() == "":
            num_cards = None
        else:
            num_cards = int(num_str)
            if num_cards <= 0:
                print("Broj kartica mora biti pozitivan. Prikazujem sve kartice.")
                num_cards = None
    except ValueError:
        print("Neispravan unos. Prikazujem sve kartice.")
        num_cards = None
    
    flashcards.run_flashcards(num_cards, randomize)
