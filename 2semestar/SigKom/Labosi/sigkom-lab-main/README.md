Prije rješavanja zadataka logirajte sa na virtualni stroj. 
Virtualni stroj namijenjen za macOS Apple Silicon (M1/M2) može se preuzeti na sljedećoj [poveznici](https://mrepro.tel.fer.hr/images/IMUNES-Ubuntu_20240305-M1.utm.zip). Za ostala računala preporučeno je preuzeti [VirtualBox](http://www.virtualbox.org/wiki/Downloads) te pomoću njega pokrenuti virtualni stroj na [poveznici](https://mrepro.tel.fer.hr/images/IMUNES-Ubuntu_20240305.ova).

Nakon pokretanja virtualnog stroja dohvatite najnoviju verziju laboratorijske vježbe:

```
$ git clone https://gitlab.tel.fer.hr/sigkom/sigkom-lab.git
$ cd sigkom-lab
```
ili (ako direktorij već postoji):
```
$ cd ~/sigkom-lab
$ git pull
```
<!---
Ako naredba "git pull", kojom dohvaćate najnoviju verziju zadataka za laboratorijske vježbe, javi grešku tipa:

```fatal: unable to access https://gitlab.tel.fer.hr/sigkom/sigkom-lab.git/: server certificate verification failed. CAfile: none CRLfile: none```

Pozovite:
```
$ sudo su
# echo $(echo -n | openssl s_client -showcerts -connect gitlab.tel.fer.hr:443 \
     2>/dev/null | sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p') \
     >> /etc/ssl/certs/ca-certificates.crt
```
Nakon toga bi `git pull` trebao ispravno dohvatiti sve datoteke.
-->
