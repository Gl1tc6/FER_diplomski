# Upute za pokretanje

## Struktura datoteka

```
.
├── docker-compose.yml
├── .env                              ← kredencijali (NE commitati u git!)
├── zeek/
│   └── local.zeek                    ← Zeek konfiguracija (JSON logging)
├── zeek-logs/                        ← Zeek logovi (automatski nastaje)
└── grafana/
    └── provisioning/
        └── datasources/
            └── influxdb.yaml         ← automatsko spajanje na InfluxDB
```

## Prije pokretanja

1. Uredi `.env` i promijeni sve lozinke i token:
   ```
   INFLUXDB_TOKEN=neki_dugi_random_string_ovdje
   INFLUXDB_PASSWORD=jaka_lozinka
   GRAFANA_PASSWORD=jaka_lozinka
   ```

2. Provjeri radi li `wlan1` sučelje:
   ```bash
   ip link show wlan1
   ```
   Ako sučelje ima drugačije ime, promijeni ga u `docker-compose.yml` pod Zeek `command`.

3. Postavi dozvole za Grafana volume direktorij (izbjegava permission greške):
   ```bash
   mkdir -p zeek-logs
   sudo chown -R 472:472 grafana/  # 472 je Grafana UID unutar kontejnera
   ```

## Pokretanje

```bash
docker compose up -d
```

Provjera logova:
```bash
docker compose logs -f          # svi servisi
docker compose logs -f zeek     # samo Zeek
docker compose logs -f grafana  # samo Grafana
```

## Pristup servisima

| Servis   | URL                    | Korisnik | Lozinka                    |
|----------|------------------------|----------|----------------------------|
| Grafana  | http://localhost:3000  | admin    | (iz .env GRAFANA_PASSWORD) |
| InfluxDB | http://localhost:8086  | admin    | (iz .env INFLUXDB_PASSWORD)|

## Provjera Zeek logova

Zeek logovi se zapisuju u `./zeek-logs/` u JSON formatu:
```bash
ls zeek-logs/
cat zeek-logs/conn.log | head -5   # connection log
cat zeek-logs/dns.log  | head -5   # DNS log
```

## Gašenje

```bash
docker compose down           # gasi kontejnere, čuva volumene
docker compose down -v        # gasi i briše sve volumene (RESET)
```
