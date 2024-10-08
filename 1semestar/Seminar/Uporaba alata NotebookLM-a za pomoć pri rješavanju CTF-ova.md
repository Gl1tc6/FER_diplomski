
## Datum: 6.10.2024

Procedura istraživanja:
- predstavljanje alata (što je, kakav je to alat, čiji je, kome je namijenjen)
- predstavljanje mogućnosti (što alat može/koje su mu mogućnosti, prednosti/mane)
- *edge* testing (testiranje rubnih slučajeva, mogućnosti)
- prezentacija (prolazak jednog CTF natjecanja i testiranje alata u stvarnoj situaciji - emuliranje CTF-a)

Valjalo bi objasniti što je **NotebookLM**.

Izvor: https://support.google.com/notebooklm/answer/14273541?hl=en
*NotebookLM gives you a personalized AI collaborator that helps you do your best thinking. After uploading your documents, NotebookLM becomes an instant expert in those sources so you can read, take notes, and collaborate with it to refine and organize your ideas.

Izvor: https://www.fastcompany.com/91143291/googles-notebooklm-is-a-great-tool-for-adding-ai-to-your-notes

*After querying NotebookLM, you can save its responses and build on them to make new notes. These reside on your noteboard, along with other AI queries and new notes you make. If you’re writing something new or preparing a presentation, you can use NotebookLM to assist you in exploring your materials. You can have a dialogue with your own notes.*

Osnovno o alatu:
- Project Tailwind (u zadnji trenutak je ime izmjenjeno)
- prototip
- izgrađen na osnovi Gemini-a (Googlov LLM, v.1.5 Pro)
- prva verzija u javnost "puštena" sredinom 2023 no samo za Američko tržište (https://www.theverge.com/23845856/google-notebooklm-tailwind-ai-notes-research)
- u Lipnju ove godine proširen je support za ostale zemlje (https://medium.com/@HacktheCost/googles-updated-ai-powered-notebooklm-now-available-across-india-uk-and-200-nations-8382a02a5395)
- namijenjen za studente i generalno ljude kojima treba pomoć izvući korisne informacije iz velike količine podataka

Značajke (https://support.google.com/notebooklm/answer/14276468?hl=en&ref_topic=14272891,14272180,&visit_id=638638406349495188-1158215411&rd=1):

- podržava 50 izvora
- parsira izvore do 500 000 riječi ili 200MB za uploadane datoteke
- izvori mogu biti:
	- Google Docs
	- Google Slides
	- PDF, Text i Markodown datoteke
	- linkovi (samo će tekst sa stranica biti parsiran, slike i videi neće, stranice koje nemaju web scrapping ili "*paywall*" neće biti parsirane)
	- Zalijepljen tekst
	- YouTube linkovi (transkript javnih videa koji ne krše ToS YT-a će biti parsiran i to tek za videe starije od 72 sata)
	- Audio datoteke
- odgovori NotebookLM-a mogu biti spremljeni zajedno u bilješke u samoj aplikaciji
- citati spremljeni u bilješke, citati vode do dijelova iz kojih su izvedeni u izvornom tekstu
- nakon uploada izvora i dalje možemo limitirati na što će se LM fokusirati
- bilježnice se mogu podijeliti sa drugim osobama (kao viewer/editor)
- LM može sugestirati pitanja na temelju izvora i prethodnih pitanja

U daljnjem testiranju:
- napraviti nekoliko svojih bilježnica te testirati koliko toga LM:
	- zna bez ikakvih izvora
	- što sve može uz pružene izvore
	- za kraj napraviti "needle-in-a-haystack" test (test u kojem se unutar velikog teksta vezanog za jednu temu ubaci informacija nevezana za tu temu - https://arize.com/blog-course/the-needle-in-a-haystack-test-evaluating-the-performance-of-llm-rag-systems/)

==Pri istraživanju teme se kao dobra alternativa pokazao alat AnythingLLM koji lokalno 
pokreće LM izbora (Llama 3 model bi bio dobar kandidat) https://anythingllm.com/ te u 
svojoj suštini radi sve isto što i NotebookLM bez potencijalnih ograničenja (naravno sada 
osobno računalo preuzima obrađivački dio posla)== 

---

## Datum: 7.10.2024

Prije svega valjalo bi testirati osnovno znanje LM-a kako na engleskom tako i na hrvatskom jeziku.


Nije moguce razgovarati sa LM-om bez postavljanja izvora pa sam prije svega postavio datoteku sa brojem 22 u nadi da cu dobiti nesto vise od samo toga. 
Testiranje predznanja na engleskom jeziku:
JA > "Disregard my source file. What can you tell me about yourself?"
LM > "The sources and our conversation history do not contain information about you, so I cannot answer your question."

JA > "Sto imas u svojim izvorima?"
LM > "The information available in the sources you have provided is limited to a single excerpt: 

*   The excerpt from "Pasted Text" contains only the number **22**. "

JA > "Odgovaraj mi na hrvatskom"
LM > "The sources you provided do not contain any information that would allow me to respond to your query in Croatian.
The available information in our conversation history only confirms that your sources contain a single numerical excerpt, "22", from a source labeled "Pasted Text"[1]"

Daljnji pokuaji da odgovori na hrvatskom su zavrsili neuspjehom iako razumije upite na hrvatskom.

U nastavku pokusavamo dobiti znanje koje nije eksplicitno upisano.

JA > Write a short script that results with said number.
LM > [kratki igrokaz o kapetanu i inzinjeru na svemirskom brodu]

JA > script in python and bash
LM >
""
## Python and Bash Scripts Outputting 22

Based on your previous input, the number you are interested in is **22** . Below are scripts in Python and Bash that will output this number:

**Python**

```python
print(22)
```

This script uses the `print()` function to output the number 22 directly.

**Bash**

```bash
echo 22
```

This script utilizes the `echo` command to print the number 22 to the console. 
""

Znaci bez dodatnih izvora LM ima nekakvo predznanje o jezicima.

JA - Can you offer some sources (links, books, videos...) that mention 22
LM -
The sources and our conversation history do not contain any links, books, or videos that mention 22. The only available information is the number **22** itself, originally provided by you. 

Osim toga ne može ništa više.



---



