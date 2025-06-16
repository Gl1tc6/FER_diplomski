## Motivacija

_Slajd 3/24_
### Eskalacija cyber prijetnji

- **2022:** Sve najeksploatirane ranjivosti identificirane od CISA-e imale su mrežne potpise
- Povećanje ransomware za **41%** u 2022.
- Prosječni trošak data breach-a: **$4.45 milijuna**
### Kompleksnost modernih IT okruženja

- Hibridne cloud inf   Rad na daljinu
- IoT uređaji povećavaju attack surface

### Potreba za proaktivnom zaštitom
antivirus slab i potreba za "online" detekcijom

---

## Evolucija sigurnosnih tehnologija

_Slajd 4/24_

```
IDS/NIDS → EDR → NDR → XDR
```

### Ključni uvidi:

- **1990s:** Pasivni IDS sustavi
- **2000s:** Dodavanje IPS funkcionalnosti
- **2010s:** Fokus na krajnje točke (EDR)
- **2015+:** Mrežna analitika (NDR)
- **2020+:** Holistički pristup (XDR)

### Formula evolucije:

**XDR = EDR + NDR + Cloud Security + Email Security + Identity Protection**

|Tehnologija|Opis|
|---|---|
|**IDS/NIDS**|Pasivna detekcija i uzbunjivanje|
|**EDR**|Fokusirane endpoint response capabilities|
|**NDR**|Mrežno-centrirana detekcija|
|**XDR**|Holistička integracija svih slojeva|

---

## IDS/NIDS - Općenito
### Što su IDS/NIDS sustavi?

- **IDS** (Intrusion Detection System) 
- **NIDS** (Network IDS)
- Pasivno nadgledanje i alarmiranje

### Ključne karakteristike:

- **Signature-based detekcija** 
- **Anomaly-based detekcija**
- **Hybrid pristup**

### Ograničenja tradicionalnih IDS/NIDS:

- Visok broj false-positive alarma
- Ograničena mogućnost automatskog odgovora
- Slaba korelacija događaja

---

## IDS/NIDS - Ključni alati i performanse

_Slajd 6/24_

|Alat|Točnost detekcije|Performanse|Resursna potrošnja|Prosjek|
|---|---|---|---|---|
|**Suricata**|1|1|3|**1.67**|
|**Zeek**|3|2|1|**2.00**|
|**Snort**|2|3|2|**2.33**|

_1 = najbolji, 3 = najlošiji_

**Pobjednik:** Suricata - najbolji overall performer s prosječnom ocjenom 1.67

---

## IDS/NIDS - Zaključak evaluacije

_Slajd 7/24_

### Suricata pobjeđuje kao najbolji overall performer zbog:

- Paralelne obrade paketa
- Multi-threading podrške
- Naprednih IDS/IPS mogućnosti

---

## EDR - Općenito

_Slajd 8/24_

### Definicija i svrha

- **24/7** kontinuiran nadzor krajnjih točaka
- Analiza procesa, aplikacija, mrežnih veza i datoteka
- **"Walled garden"** pristup - fokus isključivo na krajnjim točkama

### Ključne funkcionalnosti:

- **Real-time monitoring** svih endpoint aktivnosti
- **Behavioral analysis** za detekciju anomalija
- **Incident response** s mogućnostima izolacije
- **Forensics** - detaljna analiza napada

### Prednosti EDR pristupa:

- Visoka granularnost na razini procesa
- Idealno za remote work scenarije
- Brza izolacija kompromitiranih uređaja

---

## EDR - Izazovi i ograničenja

_Slajd 9/24_

### Glavni izazovi:

- **Ovisnost o agentima** - potreba za instalaciju na svaki uređaj
- **Potrošnja resursa** - utjecaj na performanse
- **Ograničena mrežna vidljivost** između uređaja
- **Alert fatigue** - previše alarma

### Tipični scenariji:

- Udaljena zaštita radnika
- **BYOD** (Bring Your Own Device) okruženja
- **Compliance zahtjevi** (GDPR, HIPAA)
- Odgovor na incidente i forenzika

---

## EDR - Shema

_Slajd 10/24_

_Prikaz sheme EDR-a_

**Izvor:** https://medium.com/@kavib/endpoint-detection-and-response-edr-390b7eae4999  
**Pristupljeno:** 12.6.2025

---

## NDR - Pristup

_Slajd 11/24_

### Fokus na mrežnu analizu

- **East-West traffic** - komunikacija između internih sustava
- **North-South traffic** - komunikacija s vanjskim svijetom
- **Deep Packet Inspection (DPI)** za detaljnu analizu

### Napredne analitičke metode:

- **Machine Learning** za detekciju anomalija
- **Behavioral analytics** za prepoznavanje obrazaca
- **Statistical analysis** za otkrivanje odstupanja

### Ključne prednosti:

- **Agentless pristup** - nema utjecaja na performanse krajnjih točaka
- Otkrivanje napada **"poprečnog kretanja"** (lateral movement)
- Potpuna mrežna vidljivost

---

## NDR - Tehnološke mogućnosti

_Slajd 12/24_

### Metode detekcije:

- **Signature/Fingerprint** analiza poznatih prijetnji
- **Machine learning modeli** za nepoznate prijetnje
- **Analiza mrežnog ponašanja** (NBA)
- **Analitika korisničkog i entitetskog ponašanja** (UEBA/SIEM)

### Praktični primjeri detekcije:

- **Data exfiltration** - neobični odlazni promet
- **Command & Control** komunikacija
- **Poprečno kretanje** - neobična interna komunikacija
- **Cryptojacking** - specifični mrežni uzorci

---

## NDR - Shema

_Slajd 13/24_

_Prikaz rada Darktrace NDR-a_

**Izvor:** https://safetech.ro/en/solutions/security-incident-detection-and-response/darktrace-ndr/  
**Pristupljeno:** 12.6.2025

---

## XDR

_Slajd 14/24_

### XDR kao sljedeći korak

- **"Evoluirana EDR"** - proširuje fokus s endpoints na sve
- **Holistički pristup** cijelom IT okruženju
- **Centralizirani SOC** (Security Operations Center)

### Široki spektar nadzora:

- **Endpoints** (laptopi, serveri, mobile uređaji)
- **Network infrastructure** (routeri, switches, firewalls)
- **Cloud workloads** (AWS, Azure, GCP)
- **Email systems** (Office 365, Exchange)
- **Identity systems** (Active Directory, LDAP)

### Ključne prednosti:

- Smanjenje false-positive alarma kroz korelaciju
- Automatska korelacija između sigurnosnih slojeva
- Jednostavnije upravljanje kroz jednu platformu

---

## XDR - Tehnološke komponente

_Slajd 15/24_

### Data Collection Layer

- Endpoint agent
- Mrežni "senzori"
- Cloud APIs
- Agregatori logova

### Analytics Engine

- Machine Learning algoritmi
- Ponašajna analitika
- Procjena rizika

### Response and Orchestration

- Upravljanje incidentima
- Forensics capabilities

---

## Tržišni udjeli i pozicioniranje

_Slajd 16/24_

|Proizvod|Tržišni udio|Kategorija|Specijalizacija|
|---|---|---|---|
|**Darktrace**|19.5%|IDPS/AI|Otkrivanje prijetnji uz AI|
|**CrowdStrike Falcon**|15.5%|EDR/XDR|Cloud-native platforma|
|**Wazuh**|13%|XDR|SIEM/XDR hibrid, otvorenog koda|
|**Vectra AI**|11.3%|NDR/AI|Mrežna analiza|
|**Microsoft Defender XDR**|6.9%|XDR|Integracija u Microsoft ekosustav|
|**Palo Alto Cortex XDR**|5.6%|XDR|Mrežni sigurnosni sustav|

---

## Detaljna analiza - CrowdStrike Falcon

_Slajd 17/24_

**Tržišni udio:** 15.5% (vodeći u EDR/XDR kategoriji)  
**Korisničke ocjene:** 4.8/5 (1410 recenzija)

### Ključne prednosti:

- **Cloud-native arhitektura** s potpunom skalabilnošću
- **AI/ML tehnologija** za naprednu detekciju prijetnji
- **Lightweight agent** s minimalnim utjecajem na performanse
- **Real-time threat intelligence** iz globalnog oblaka
- **Advanced threat hunting** mogućnosti

### Glavni nedostaci:

- Viša cijena u odnosu na konkurenciju
- Složenost konfiguracije upozorenja i izvještaja
- Strma krivulja učenja za potpunu optimizaciju
- Ovisnost o internetskoj vezi za optimalne performanse

---

## Detaljna analiza - Darktrace

_Slajd 18/24_

**Tržišni udio:** 19.5% (dominira IDPS segment)  
**Fokus:** Samoučeći AI pristup cyber obrani

### Ključne prednosti:

- **Enterprise Immune System** - mimikra ljudskog imunosnog sustava
- **Antigena funkcionalnost** za instant automated response
- **Samoučeći algoritmi** koji se prilagođavaju organizaciji
- Minimalni false-positive šum nakon početnog učenja
- Izvrsno email i network monitoring

### Glavni nedostaci:

- **Vrlo visoka cijena** - jedan od najskupljih na tržištu
- Početni period učenja s visokim brojem false-positives
- **Ograničena endpoint vidljivost** - fokus na mrežu
- Složena integracija s drugim sigurnosnim alatima

---

## Detaljna analiza - Wazuh

_Slajd 19/24_

**Tržišni udio:** 13.0% (najveći open-source "igrač")  
**Model:** Potpuno besplatno, community-driven

### Ključne prednosti:

- **Zero licensing costs** - potpuno besplatno
- **Visoka customizabilnost** za specifične potrebe
- **SIEM + XDR** funkcionalnosti u jednom paketu
- **Multi-platform podrška** (Windows, Linux, macOS)
- Aktivna zajednica developera i korisnika

### Glavni izazovi:

- Zahtijeva **tehničku ekspertizu** za implementaciju
- **Ograničena komercijalna podrška**
- Kompleksna integracija za heterogene sustave
- Skalabilnost izazovi s velikim količinama podataka
- **DIY pristup** maintenance i optimizacije

---

## Analiza - ostali

_Slajd 20/24_

### Microsoft Defender XDR

**Prednosti:** Microsoft integracija, isplativo za M365 korisnike  
**Nedostaci:** Ograničene third-party integracije, kompleksno licenciranje

### SentinelOne Singularity

**Prednosti:** Autonomni AI agent, rollback mogućnosti, offline zaštita  
**Nedostaci:** Viši implementacijski troškovi, korisničko sučelje može biti poboljšano

### Vectra AI Platform

**Prednosti:** Izvrsno risk scoring, AI-driven detection, Microsoft integracija  
**Nedostaci:** Zahtijeva SIEM integraciju, ograničena host-level vidljivost

### Palo Alto Cortex XDR

**Prednosti:** Robusna multi-layer integracija, proaktivna detekcija  
**Nedostaci:** Kompleksna implementacija, visoka cijena, težak što se resursa tiče

---

## Preporuke prema veličini poduzeća

_Slajd 21/24_

### Mala poduzeća (1-100 zaposlenika)

**Budžet:** $5,000-$50,000 godišnje  
**Preporuke:** Microsoft Defender XDR, Darktrace, Wazuh

### Srednja poduzeća (100-1000 zaposlenika)

**Budžet:** $50,000-$500,000 godišnje  
**Preporuke:** Cisco Sourcefire SNORT, SentinelOne, Vectra AI

### Velika poduzeća (1000+ zaposlenika)

**Budžet:** $500,000+ godišnje  
**Preporuke:** CrowdStrike Falcon, Palo Alto Cortex XDR, Vectra AI

---

## Ključna otkrića

_Slajd 22/24_

### Tehnološki trendovi

- **AI/ML dominacija** u svim vodećim rješenjima
- **Cloud-first pristup** kao novi standard
- **Konsolidacija alata** - preference za integrirane platforme
- **Demokratizacija sigurnosti** za manje organizacije

### Nema "one-size-fits-all" rješenja

Optimalan izbor ovisan je o veličini organizacije, budžetu i tehničkim kapacitetima

### Buduće perspektive

- **Autonomna sigurnost** s potpuno automatiziranim odgovorom
- **Quantum-resistant security**
- Dublja **Zero Trust** integracija
- Proširenje na **IoT i edge computing**

---

## Zaključak

_Slajd 23/24_

### Sigurnosni krajolik kontinuirano evoluira

Od tradicionalnih IDS/NIDS sustava prema sofisticiranim XDR platformama

### Uspješne organizacije će biti one koje:

- Kombiniraju **tehnološku inovaciju** s promišljenim strategijskim planiranjem
- **Kontinuirano ulažu** u ljudski kapital
- Prilagođavaju odabir tehnologija **specifičnim organizacijskim potrebama**
- Fokusiraju se na **proper implementation** i **ongoing optimization**

---

## Literatura

_Slajd 24/24_

Unutar samog seminara!

---

## Hvala!

**Pitanja i diskusija**