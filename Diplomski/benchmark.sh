#!/bin/bash
# ============================================================
# benchmark.sh — Usporedba InfluxDB vs Elasticsearch
#   ./benchmark.sh           — pokreni mjerenja (odbija ako dataset nije fer)
#   ./benchmark.sh --force   — mjeri i kad se datasetovi NE poklapaju
#   ./benchmark.sh --clean   — obriši zeek podatke iz obje baze (PRIJE ingesta)
# ============================================================

set -euo pipefail

if [ ! -f .env ]; then
    echo "[ERROR] .env nije pronađen. Pokreni iz root direktorija projekta."
    exit 1
fi
source .env

RESULTS_FILE="benchmark_results.txt"
INFLUX_URL="http://localhost:8086"
ELASTIC_URL="http://localhost:9200"
INFLUX_CONTAINER="influxdb"          # naziv kontejnera iz docker-compose.yml
QUERY_REPEATS=10                     # mjerenja po upitu (+2 warmup koja se odbacuju)
RAM_SAMPLES=3                        # uzoraka RAM-a nakon smirivanja
DATASET_FAIR=1                       # postavlja benchmark_verify
FORCE=0

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

header()  { echo ""; echo -e "${BLUE}============================================================${NC}"; echo -e "${BLUE}  $1${NC}"; echo -e "${BLUE}============================================================${NC}"; }
info()    { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }
result()  { echo -e "  $1"; }
# ispiši na ekran I u datoteku rezultata
reslog()  { echo -e "  $1"; echo "  $1" >> "$RESULTS_FILE"; }

check_service() {
    if curl -sf --max-time 3 "$1" > /dev/null 2>&1; then
        info "$2 dostupan ($1)"; return 0
    else
        warn "$2 NIJE dostupan ($1) — preskačem."; return 1
    fi
}

# ------------------------------------------------------------
# Brojanje zapisa (za VERIFIKACIJU pariteta, ne za latenciju)
# InfluxDB Flux vraća ANOTIRANI CSV, ne JSON → zadnja kolona zadnjeg retka.
# ------------------------------------------------------------
influx_count() {
    curl -s -X POST "${INFLUX_URL}/api/v2/query?org=${INFLUXDB_ORG}" \
        -H "Authorization: Token ${INFLUXDB_TOKEN}" \
        -H "Content-Type: application/vnd.flux" \
        -d 'from(bucket:"'"$INFLUXDB_BUCKET"'") |> range(start: 0) |> filter(fn: (r) => r._measurement == "zeek_conn" and r._field == "uid") |> count() |> group() |> sum()' \
        2>/dev/null | sed '/^[[:space:]]*$/d' | tail -n1 | awk -F',' '{gsub(/\r/,""); print $NF}'
}
elastic_count() {
    curl -s "${ELASTIC_URL}/zeek-conn-*/_count" 2>/dev/null | grep -o '"count":[0-9]*' | head -1 | cut -d: -f2
}

# ------------------------------------------------------------
# STORAGE — like-for-like: samo zeek bucket/indeks, ne cijeli data dir.
# POPRAVAK: veličina InfluxDB bucketa mjeri se IZNUTRA kontejnera
# (`du` u /var/lib/influxdb2/...), jer host-bind putanja nije pouzdana.
# ------------------------------------------------------------
influx_bucket_size() {
    local bid
    bid=$(curl -s "${INFLUX_URL}/api/v2/buckets?name=${INFLUXDB_BUCKET}" \
            -H "Authorization: Token ${INFLUXDB_TOKEN}" 2>/dev/null \
            | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
    if [ -z "$bid" ]; then
        echo "N/A (bucket id nije dohvaćen — provjeri token/URL)"; return
    fi
    docker exec "$INFLUX_CONTAINER" du -sh "/var/lib/influxdb2/engine/data/$bid" 2>/dev/null | cut -f1 \
        || echo "N/A (du u kontejneru pao — provjeri naziv kontejnera/putanju)"
}

# ------------------------------------------------------------
# Upiti za LATENCIJU — vraćaju time_total (s), mjeri se end-to-end
# latencija kakvu vidi klijent na localhostu (mreža = zanemariva).
# ------------------------------------------------------------
q_influx() {
    curl -o /dev/null -s -w '%{time_total}' -X POST \
        "${INFLUX_URL}/api/v2/query?org=${INFLUXDB_ORG}" \
        -H "Authorization: Token ${INFLUXDB_TOKEN}" \
        -H "Content-Type: application/vnd.flux" \
        -d "$1"
}
q_es_search() {
    curl -o /dev/null -s -w '%{time_total}' -X GET \
        "${ELASTIC_URL}/zeek-conn-*/_search" \
        -H 'Content-Type: application/json' -d "$1"
}

# Pokreni naredbu (ispisuje time_total) N puta + 2 warmup.
# Vrati: "medijan ms  (min X, max Y)" — pošteniji od golog prosjeka.
measure_latency() {
    "$@" >/dev/null 2>&1 || true
    "$@" >/dev/null 2>&1 || true
    local t times=()
    for _ in $(seq 1 "$QUERY_REPEATS"); do
        t=$("$@" 2>/dev/null || echo 0)
        times+=("$t")
    done
    printf '%s\n' "${times[@]}" | sort -n | awk -v n="$QUERY_REPEATS" '
        { a[NR]=$1 }
        END {
            if (n % 2) med=a[(n+1)/2]; else med=(a[n/2]+a[n/2+1])/2;
            printf "%.1f ms  (min %.1f, max %.1f)", med*1000, a[1]*1000, a[n]*1000;
        }'
}

init_results() {
    cat > "$RESULTS_FILE" << EOF
============================================================
  BENCHMARK REZULTATI: InfluxDB vs Elasticsearch
  Datum: $(date '+%Y-%m-%d %H:%M:%S')
  Host:  $(hostname) | $(uname -m)
  OS:    $(uname -o) $(uname -r)
============================================================

EOF
}

# ------------------------------------------------------------
# 0. VERIFIKACIJA: usporedba je fer SAMO ako obje baze drže isti broj zapisa
# ------------------------------------------------------------
benchmark_verify() {
    header "0. VERIFIKACIJA JEDNAKOSTI DATASETA"
    echo "=== 0. VERIFIKACIJA ===" >> "$RESULTS_FILE"
    local ic ec raw
    if [ -f zeek-logs/conn.log ]; then raw=$(wc -l < zeek-logs/conn.log | tr -d ' '); else raw="N/A"; fi
    ic=$(influx_count);  [ -z "$ic" ] && ic="N/A"
    ec=$(elastic_count); [ -z "$ec" ] && ec="N/A"
    reslog "Redaka u conn.log (izvor):   $raw"
    reslog "Elasticsearch dokumenata:    $ec"
    reslog "InfluxDB točaka (zeek_conn):  $ic"
    # Paritet = ES je svaki redak indeksirao TOČNO JEDNOM (ES ne deduplicira).
    if [ "$ec" = "N/A" ] || [ "$raw" = "N/A" ]; then
        warn "Ne mogu provjeriti ES vs izvor — provjeri servise i conn.log."
        DATASET_FAIR=0
    elif [ "$ec" != "$raw" ]; then
        warn "ES ($ec) != broj redaka ($raw): Filebeat je čitao dvaput ili djelomično."
        warn "Resetiraj ES indeks + Filebeat kontejner i ingestaj fajl JEDNOM."
        DATASET_FAIR=0
    else
        info "ES = $raw: svaki redak indeksiran točno jednom."
    fi
    # InfluxDB < redaka je OČEKIVANO: točke s istim (ts, tagset) se stapaju.
    if [ "$ic" != "N/A" ] && [ "$raw" != "N/A" ] && [ "$ic" -gt "$raw" ] 2>/dev/null; then
        warn "InfluxDB ($ic) > redaka ($raw) — neočekivano, provjeri Telegraf."
        DATASET_FAIR=0
    elif [ "$ic" != "N/A" ] && [ "$ic" != "$raw" ]; then
        info "InfluxDB ($ic) < redaka ($raw): očekivano stapanje (ts+tagovi), nije gubitak."
    fi
    echo "" >> "$RESULTS_FILE"
}

# ------------------------------------------------------------
# 1. STORAGE (samo zeek podaci)
# ------------------------------------------------------------
benchmark_storage() {
    header "1. STORAGE (samo zeek podaci)"
    echo "=== 1. STORAGE ===" >> "$RESULTS_FILE"

    if [ -f "zeek-logs/conn.log" ]; then
        reslog "Sirovi conn.log (referenca): $(du -sh zeek-logs/conn.log 2>/dev/null | cut -f1)"
    fi
    if check_service "$INFLUX_URL/health" "InfluxDB"; then
        reslog "InfluxDB bucket na disku:    $(influx_bucket_size)"
    fi
    if check_service "$ELASTIC_URL" "Elasticsearch"; then
        reslog "Elasticsearch indeks:"
        local es; es=$(curl -s "${ELASTIC_URL}/_cat/indices/zeek-conn-*?h=index,docs.count,store.size" 2>/dev/null)
        echo "$es" | sed 's/^/    /'; echo "$es" | sed 's/^/    /' >> "$RESULTS_FILE"
    fi
    echo "" >> "$RESULTS_FILE"
}

# ------------------------------------------------------------
# 2. RAM (nakon smirivanja, više uzoraka) — Filebeat DODAN
# ------------------------------------------------------------
benchmark_ram() {
    header "2. RAM POTROŠNJA (nakon smirivanja)"
    echo "=== 2. RAM POTROŠNJA ===" >> "$RESULTS_FILE"
    info "Čekam 15s da se ingestija/upiti smire..."
    sleep 15

    # grafana namjerno izostavljena (ne radi tijekom benchmarka, vidi metodologiju)
    local containers=("influxdb" "telegraf" "elasticsearch" "filebeat" "zeek-ids")
    for s in $(seq 1 "$RAM_SAMPLES"); do
        reslog "--- uzorak $s ---"
        for c in "${containers[@]}"; do
            if docker ps --format "{{.Names}}" | grep -q "^${c}$"; then
                local st; st=$(docker stats --no-stream --format "{{.MemUsage}} | {{.CPUPerc}}" "$c" 2>/dev/null)
                printf "  %-16s %s\n" "$c" "$st"
                printf "  %-16s %s\n" "$c" "$st" >> "$RESULTS_FILE"
            fi
        done
        [ "$s" -lt "$RAM_SAMPLES" ] && sleep 5
    done
    echo "" >> "$RESULTS_FILE"
}

# ------------------------------------------------------------
# 3. QUERY LATENCY — EKVIVALENTAN posao u oba sustava
#    COUNT: ES radi value_count SKEN (ne keširani _count) ~ Flux count()
#    TOP10: terms agg ~ group+count+sort+limit
#    AGG:   date_histogram 1m ~ aggregateWindow(every:1m, count)
# ------------------------------------------------------------
benchmark_query_latency() {
    header "3. QUERY LATENCY (medijan od $QUERY_REPEATS, +2 warmup)"
    echo "=== 3. QUERY LATENCY ===" >> "$RESULTS_FILE"

    local B="$INFLUXDB_BUCKET"
    local IN_COUNT='from(bucket:"'"$B"'") |> range(start: 0) |> filter(fn: (r) => r._measurement == "zeek_conn" and r._field == "uid") |> count() |> group() |> sum()'
    local IN_TOP='from(bucket:"'"$B"'") |> range(start: 0) |> filter(fn: (r) => r._measurement == "zeek_conn" and r._field == "uid") |> group(columns: ["dst_ip"]) |> count() |> group() |> sort(columns: ["_value"], desc: true) |> limit(n: 10)'
    local IN_AGG='from(bucket:"'"$B"'") |> range(start: 0) |> filter(fn: (r) => r._measurement == "zeek_conn" and r._field == "uid") |> aggregateWindow(every: 1m, fn: count, createEmpty: false)'

    # NB: dst_ip.keyword — dinamičko mapiranje stringove daje kao text+.keyword.
    # Ako si mapirao dst_ip kao ip-tip, makni ".keyword".
    local ES_COUNT='{"size":0,"track_total_hits":true,"aggs":{"c":{"value_count":{"field":"uid"}}}}'
    local ES_TOP='{"size":0,"aggs":{"top":{"terms":{"field":"dst_ip.keyword","size":10}}}}'
    local ES_AGG='{"size":0,"aggs":{"per_min":{"date_histogram":{"field":"@timestamp","fixed_interval":"1m"}}}}'

    if check_service "$INFLUX_URL/health" "InfluxDB"; then
        reslog "InfluxDB: COUNT (sken)        $(measure_latency q_influx "$IN_COUNT")"
        reslog "InfluxDB: TOP 10 dst_ip       $(measure_latency q_influx "$IN_TOP")"
        reslog "InfluxDB: AGG po minutama     $(measure_latency q_influx "$IN_AGG")"
    fi
    echo "" >> "$RESULTS_FILE"
    if check_service "$ELASTIC_URL" "Elasticsearch"; then
        reslog "Elastic:  COUNT (sken)        $(measure_latency q_es_search "$ES_COUNT")"
        reslog "Elastic:  TOP 10 dst_ip       $(measure_latency q_es_search "$ES_TOP")"
        reslog "Elastic:  AGG po minutama     $(measure_latency q_es_search "$ES_AGG")"
    fi
    echo "" >> "$RESULTS_FILE"
}

# ------------------------------------------------------------
# Čišćenje obje baze PRIJE ponovnog ingesta (paritet kreće od nule)
# ------------------------------------------------------------
clean_state() {
    header "ČIŠĆENJE OBJE BAZE"
    if curl -s -X POST "${INFLUX_URL}/api/v2/delete?org=${INFLUXDB_ORG}&bucket=${INFLUXDB_BUCKET}" \
        -H "Authorization: Token ${INFLUXDB_TOKEN}" -H 'Content-Type: application/json' \
        -d '{"start":"1970-01-01T00:00:00Z","stop":"2100-01-01T00:00:00Z","predicate":"_measurement=\"zeek_conn\""}' >/dev/null 2>&1; then
        info "InfluxDB zeek_conn podaci obrisani."
    else
        warn "InfluxDB brisanje preskočeno (servis nedostupan?)."
    fi
    # ES 8 odbija wildcard DELETE (action.destructive_requires_name) -> razriješi imena
    local _idx _i
    _idx=$(curl -s "${ELASTIC_URL}/_cat/indices/zeek-conn-*?h=index" 2>/dev/null | tr -d ' ')
    if [ -n "$_idx" ]; then
        for _i in $_idx; do curl -s -X DELETE "${ELASTIC_URL}/$_i" >/dev/null 2>&1; done
        info "Elasticsearch indeksi obrisani: $(echo $_idx | tr '\n' ' ')"
    else
        info "Nema zeek-conn-* indeksa za brisanje (ili ES nedostupan)."
    fi
    warn "Filebeat registry NE leži u volumenu (tvoj compose ga ne montira) — živi u"
    warn "kontejneru. Resetiraj ga uklanjanjem kontejnera prije ponovnog ingesta:"
    warn "  docker compose -f docker-compose-elk.yml rm -sf filebeat"
}

# ------------------------------------------------------------
main() {
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   InfluxDB vs Elasticsearch Benchmark      ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"

    case "${1:-}" in
        --clean) clean_state; exit 0 ;;
        --force) FORCE=1 ;;
    esac

    init_results
    benchmark_verify

    if [ "$DATASET_FAIR" -ne 1 ] && [ "$FORCE" -ne 1 ]; then
        error "Dataset nije fer. Mjerenja prekinuta da ne proizvedeš lažne brojke."
        error "Riješi paritet (vidi korake) ili pokreni './benchmark.sh --force' svjesno."
        exit 1
    fi
    [ "$DATASET_FAIR" -ne 1 ] && warn "Mjerim uz --force: rezultati NISU fer, označi ih u radu."

    benchmark_storage
    benchmark_ram
    benchmark_query_latency

    echo ""
    info "Gotovo. Rezultati: ${RESULTS_FILE}"
}

main "$@"