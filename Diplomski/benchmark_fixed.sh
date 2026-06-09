#!/bin/bash
# ============================================================
# benchmark.sh — Usporedba InfluxDB vs Elasticsearch
#   ./benchmark.sh           — pokreni mjerenja
#   ./benchmark.sh --clean   — obriši zeek podatke iz obje baze
#                              (pokreni PRIJE ponovnog ingesta)
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
QUERY_REPEATS=10   # mjerenja po upitu (+2 warmup koji se odbacuju)
RAM_SAMPLES=3      # uzoraka RAM-a nakon smirivanja

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

header()  { echo ""; echo -e "${BLUE}============================================================${NC}"; echo -e "${BLUE}  $1${NC}"; echo -e "${BLUE}============================================================${NC}"; }
info()    { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }
result()  { echo -e "  $1"; }

check_service() {
    if curl -sf --max-time 3 "$1" > /dev/null 2>&1; then
        info "$2 dostupan ($1)"; return 0
    else
        warn "$2 NIJE dostupan ($1) — preskačem."; return 1
    fi
}

# ------------------------------------------------------------
# Upiti koji vraćaju time_total (sekunde) — mjeri se server-side
# vrijeme, ne pokretanje curl procesa.
# ------------------------------------------------------------
q_influx() {
    curl -o /dev/null -s -w '%{time_total}' -X POST \
        "${INFLUX_URL}/api/v2/query?org=${INFLUXDB_ORG}" \
        -H "Authorization: Token ${INFLUXDB_TOKEN}" \
        -H "Content-Type: application/vnd.flux" \
        -d "$1"
}
q_es_count()  { curl -o /dev/null -s -w '%{time_total}' "${ELASTIC_URL}/zeek-conn-*/_count"; }
q_es_search() { curl -o /dev/null -s -w '%{time_total}' -X GET "${ELASTIC_URL}/zeek-conn-*/_search" -H 'Content-Type: application/json' -d "$1"; }

# Pokreni naredbu (koja ispisuje time_total) N puta + 2 warmup, vrati prosjek u ms
measure_latency() {
    "$@" >/dev/null 2>&1 || true
    "$@" >/dev/null 2>&1 || true
    local sum=0 t
    for _ in $(seq 1 "$QUERY_REPEATS"); do
        t=$("$@" 2>/dev/null || echo 0)
        sum=$(awk -v s="$sum" -v t="$t" 'BEGIN{printf "%.6f", s+t}')
    done
    awk -v s="$sum" -v n="$QUERY_REPEATS" 'BEGIN{printf "%.1f", (s/n)*1000}'
}

# ------------------------------------------------------------
# Broj zapisa — InfluxDB Flux vraća ANOTIRANI CSV, ne JSON.
# (Stari grep '"_value"' je zato uvijek vraćao N/A.)
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
# Storage — like-for-like: samo zeek indeks/bucket, ne cijeli data dir.
# ------------------------------------------------------------
influx_bucket_size() {
    local bid
    bid=$(curl -s "${INFLUX_URL}/api/v2/buckets?name=${INFLUXDB_BUCKET}" \
            -H "Authorization: Token ${INFLUXDB_TOKEN}" 2>/dev/null \
            | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
    if [ -n "$bid" ] && [ -d "influx-data/engine/data/$bid" ]; then
        du -sh "influx-data/engine/data/$bid" 2>/dev/null | cut -f1
    else
        echo "N/A (bucket dir nije pronađen)"
    fi
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
# 0. VERIFIKACIJA: usporedba je fer samo ako obje baze drže isti broj zapisa
# ------------------------------------------------------------
benchmark_verify() {
    header "0. VERIFIKACIJA JEDNAKOSTI DATASETA"
    echo "=== 0. VERIFIKACIJA ===" >> "$RESULTS_FILE"
    local ic ec
    ic=$(influx_count);  [ -z "$ic" ] && ic="N/A"
    ec=$(elastic_count); [ -z "$ec" ] && ec="N/A"
    result "InfluxDB zapisa (zeek_conn): $ic"
    result "Elasticsearch dokumenata:    $ec"
    echo "  InfluxDB: $ic | Elasticsearch: $ec" >> "$RESULTS_FILE"
    if [ "$ic" = "N/A" ] || [ "$ec" = "N/A" ]; then
        warn "Jedna baza ne odgovara — provjeri da su oba stacka dignuta i napunjena."
    elif [ "$ic" != "$ec" ]; then
        warn "Broj zapisa se NE poklapa. Storage/RAM usporedba NIJE fer."
        warn "Pokreni './benchmark.sh --clean', obriši Filebeat registry, pa ingestaj isti dataset u oba."
    else
        info "Datasetovi jednaki ($ic zapisa) — usporedba je fer."
    fi
    echo "" >> "$RESULTS_FILE"
}

# ------------------------------------------------------------
# 1. STORAGE
# ------------------------------------------------------------
benchmark_storage() {
    header "1. STORAGE (samo zeek podaci)"
    echo "=== 1. STORAGE ===" >> "$RESULTS_FILE"

    if [ -f "zeek-logs/conn.log" ]; then
        local raw; raw=$(du -sh zeek-logs/conn.log 2>/dev/null | cut -f1)
        result "Sirovi conn.log (referenca): $raw"
        echo "  Sirovi conn.log: $raw" >> "$RESULTS_FILE"
    fi

    if check_service "$INFLUX_URL/health" "InfluxDB"; then
        local isz; isz=$(influx_bucket_size)
        result "InfluxDB bucket na disku:    $isz"
        echo "  InfluxDB bucket: $isz" >> "$RESULTS_FILE"
    fi

    if check_service "$ELASTIC_URL" "Elasticsearch"; then
        result "Elasticsearch indeksi:"
        local es; es=$(curl -s "${ELASTIC_URL}/_cat/indices/zeek-conn-*?h=index,docs.count,store.size" 2>/dev/null)
        echo "$es" | sed 's/^/    /'
        echo "  Elasticsearch (_cat/indices zeek-*):" >> "$RESULTS_FILE"
        echo "$es" | sed 's/^/    /' >> "$RESULTS_FILE"
    fi
    echo "" >> "$RESULTS_FILE"
}

# ------------------------------------------------------------
# 2. RAM (nakon smirivanja, više uzoraka)
# ------------------------------------------------------------
benchmark_ram() {
    header "2. RAM POTROŠNJA (nakon smirivanja)"
    echo "=== 2. RAM POTROŠNJA ===" >> "$RESULTS_FILE"
    info "Čekam 15s da se ingestija/upiti smire..."
    sleep 15

    local containers=("influxdb" "elasticsearch" "telegraf" "grafana" "zeek-ids")
    for s in $(seq 1 "$RAM_SAMPLES"); do
        result "--- uzorak $s ---"
        echo "  --- uzorak $s ---" >> "$RESULTS_FILE"
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
# 3. QUERY LATENCY (server-side, prosjek)
# ------------------------------------------------------------
benchmark_query_latency() {
    header "3. QUERY LATENCY (prosjek $QUERY_REPEATS, +2 warmup)"
    echo "=== 3. QUERY LATENCY ===" >> "$RESULTS_FILE"

    local F_COUNT='from(bucket:"'"$INFLUXDB_BUCKET"'") |> range(start: 0) |> filter(fn: (r) => r._measurement == "zeek_conn") |> count()'
    local F_TOP10='from(bucket:"'"$INFLUXDB_BUCKET"'") |> range(start: 0) |> filter(fn: (r) => r._measurement == "zeek_conn") |> group(columns: ["dst_ip"]) |> count() |> top(n:10, columns:["_value"])'
    local F_AGG='from(bucket:"'"$INFLUXDB_BUCKET"'") |> range(start: -90d) |> filter(fn: (r) => r._measurement == "zeek_conn") |> aggregateWindow(every: 1m, fn: count, createEmpty: false)'
    local ES_TOP10='{"size":0,"aggs":{"top_dst":{"terms":{"field":"dst_ip","size":10}}}}'
    local ES_AGG='{"size":0,"aggs":{"per_minute":{"date_histogram":{"field":"@timestamp","fixed_interval":"1m"}}}}'

    printf "  %-32s %-12s\n" "Upit" "Prosj. (ms)"
    printf "  %-32s %-12s\n" "----" "-----------"

    if check_service "$INFLUX_URL/health" "InfluxDB"; then
        for pair in "COUNT|$F_COUNT" "TOP 10 dst_ip|$F_TOP10" "AGG po minutama|$F_AGG"; do
            local label="${pair%%|*}" flux="${pair#*|}" ms
            ms=$(measure_latency q_influx "$flux")
            printf "  %-32s %-12s\n" "InfluxDB: $label" "$ms"
            printf "  %-32s %-12s\n" "InfluxDB: $label" "$ms" >> "$RESULTS_FILE"
        done
    fi
    echo "" >> "$RESULTS_FILE"

    if check_service "$ELASTIC_URL" "Elasticsearch"; then
        local ms
        ms=$(measure_latency q_es_count);            printf "  %-32s %-12s\n" "Elastic: COUNT" "$ms"; printf "  %-32s %-12s\n" "Elastic: COUNT" "$ms" >> "$RESULTS_FILE"
        ms=$(measure_latency q_es_search "$ES_TOP10"); printf "  %-32s %-12s\n" "Elastic: TOP 10 dst_ip" "$ms"; printf "  %-32s %-12s\n" "Elastic: TOP 10 dst_ip" "$ms" >> "$RESULTS_FILE"
        ms=$(measure_latency q_es_search "$ES_AGG");   printf "  %-32s %-12s\n" "Elastic: AGG po minutama" "$ms"; printf "  %-32s %-12s\n" "Elastic: AGG po minutama" "$ms" >> "$RESULTS_FILE"
    fi
    echo "" >> "$RESULTS_FILE"
}

# ------------------------------------------------------------
# CLEAN — obriši zeek podatke iz obje baze prije ponovnog ingesta
# ------------------------------------------------------------
clean_state() {
    header "CLEAN STATE"
    local idxs
    idxs=$(curl -s "${ELASTIC_URL}/_cat/indices/zeek-*?h=index" 2>/dev/null)
    if [ -n "$idxs" ]; then
        for idx in $idxs; do
            curl -s -X DELETE "${ELASTIC_URL}/$idx" >/dev/null 2>&1
        done
        info "Elasticsearch indeksi obrisani: $(echo "$idxs" | tr '\n' ' ')"
    else
        warn "ES: nema zeek-* indeksa za brisanje (ili servis nedostupan)."
    fi
    if curl -s -X POST "${INFLUX_URL}/api/v2/delete?org=${INFLUXDB_ORG}&bucket=${INFLUXDB_BUCKET}" \
        -H "Authorization: Token ${INFLUXDB_TOKEN}" -H 'Content-Type: application/json' \
        -d '{"start":"1970-01-01T00:00:00Z","stop":"2100-01-01T00:00:00Z","predicate":"_measurement=\"zeek_conn\""}' >/dev/null 2>&1; then
        info "InfluxDB zeek_conn podaci obrisani."
    else
        warn "InfluxDB brisanje preskočeno (servis nedostupan?)."
    fi
    warn "Filebeat registry MORAŠ očistiti ručno (inače re-indeksira iste logove):"
    warn "  docker compose down && docker volume rm <filebeat_data_volume>  (ili obriši registry dir)"
}

# ------------------------------------------------------------
main() {
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   InfluxDB vs Elasticsearch Benchmark      ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"

    if [ "${1:-}" = "--clean" ]; then
        clean_state
        exit 0
    fi

    init_results
    benchmark_verify
    benchmark_storage
    benchmark_ram
    benchmark_query_latency

    echo ""
    info "Gotovo. Rezultati: ${RESULTS_FILE}"
}

main "$@"