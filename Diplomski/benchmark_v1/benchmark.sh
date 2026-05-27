#!/bin/bash
# ============================================================
# benchmark.sh — Usporedba InfluxDB vs Elasticsearch
# Pokreni iz root direktorija projekta:
#   chmod +x benchmark.sh
#   ./benchmark.sh
# ============================================================

set -euo pipefail

# ------------------------------------------------------------
# Učitaj .env varijable
# ------------------------------------------------------------
if [ ! -f .env ]; then
    echo "[ERROR] .env file nije pronađen. Pokreni iz root direktorija projekta."
    exit 1
fi
source .env

# ------------------------------------------------------------
# Konstante
# ------------------------------------------------------------
RESULTS_FILE="benchmark_results.txt"
INFLUX_URL="http://localhost:8086"
ELASTIC_URL="http://localhost:9200"
QUERY_REPEATS=5   # Svaki upit se ponavlja N puta, uzima se prosjek

# Boje za terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ------------------------------------------------------------
# Helper funkcije
# ------------------------------------------------------------
header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

info()    { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }
result()  { echo -e "  $1"; }

# Mjeri prosječno vrijeme N ponavljanja (vraća ms)
measure_latency() {
    local cmd="$1"
    local total=0

    for i in $(seq 1 $QUERY_REPEATS); do
        local start end elapsed
        start=$(date +%s%N)
        eval "$cmd" > /dev/null 2>&1 || true
        end=$(date +%s%N)
        elapsed=$(( (end - start) / 1000000 ))  # ns -> ms
        total=$(( total + elapsed ))
    done

    echo $(( total / QUERY_REPEATS ))
}

# Provjeri je li servis dostupan
check_service() {
    local url="$1"
    local name="$2"
    if curl -sf --max-time 3 "$url" > /dev/null 2>&1; then
        info "$name je dostupan ($url)"
        return 0
    else
        warn "$name NIJE dostupan ($url) — preskačem taj dio benchmarka."
        return 1
    fi
}

# ------------------------------------------------------------
# Inicijalizacija rezultata
# ------------------------------------------------------------
init_results() {
    cat > "$RESULTS_FILE" << EOF
============================================================
  BENCHMARK REZULTATI: InfluxDB vs Elasticsearch
  Datum: $(date '+%Y-%m-%d %H:%M:%S')
  Host:  $(hostname) | $(uname -m)
  OS:    $(uname -o) $(uname -r)
============================================================

EOF
    info "Rezultati će biti zapisani u: $RESULTS_FILE"
}

# ------------------------------------------------------------
# 1. STORAGE BENCHMARK
# ------------------------------------------------------------
benchmark_storage() {
    header "1. STORAGE"
    echo "=== 1. STORAGE ===" >> "$RESULTS_FILE"
    echo "Datum: $(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"

    # Zeek logovi (referenca)
    if [ -d "zeek-logs" ]; then
        local zeek_size
        zeek_size=$(du -sh zeek-logs/ 2>/dev/null | cut -f1)
        result "Zeek logovi (sirovi):  $zeek_size"
        echo "  Zeek logovi (sirovi):  $zeek_size" >> "$RESULTS_FILE"
    fi

    # InfluxDB
    if [ -d "influx-data" ]; then
        local influx_size
        influx_size=$(du -sh influx-data/ 2>/dev/null | cut -f1)
        result "InfluxDB data dir:     $influx_size"
        echo "  InfluxDB data dir:     $influx_size" >> "$RESULTS_FILE"
    else
        warn "influx-data/ ne postoji — nije pokrenut InfluxDB stack?"
        echo "  InfluxDB: N/A (influx-data/ ne postoji)" >> "$RESULTS_FILE"
    fi

    # Elasticsearch
    if [ -d "elastic-data" ]; then
        local elastic_size
        elastic_size=$(du -sh elastic-data/ 2>/dev/null | cut -f1)
        result "Elasticsearch data:    $elastic_size"
        echo "  Elasticsearch data:    $elastic_size" >> "$RESULTS_FILE"
    else
        warn "elastic-data/ ne postoji — nije pokrenut ELK stack?"
        echo "  Elasticsearch: N/A (elastic-data/ ne postoji)" >> "$RESULTS_FILE"
    fi

    echo "" >> "$RESULTS_FILE"
}

# ------------------------------------------------------------
# 2. RAM BENCHMARK
# ------------------------------------------------------------
benchmark_ram() {
    header "2. RAM POTROŠNJA (idle)"
    echo "=== 2. RAM POTROŠNJA (idle) ===" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"

    # Čekaj 5s da se kontejneri stabiliziraju
    sleep 5

    local containers=("influxdb" "elasticsearch" "telegraf" "grafana" "zeek-ids")
    local found=0

    printf "  %-20s %-15s %-10s\n" "Kontejner" "RAM" "CPU"
    printf "  %-20s %-15s %-10s\n" "---------" "---" "---"
    printf "  %-20s %-15s %-10s\n" "Kontejner" "RAM" "CPU" >> "$RESULTS_FILE"
    printf "  %-20s %-15s %-10s\n" "---------" "---" "---" >> "$RESULTS_FILE"

    for container in "${containers[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
            local stats
            stats=$(docker stats --no-stream --format "{{.MemUsage}}|{{.CPUPerc}}" "$container" 2>/dev/null)
            local mem cpu
            mem=$(echo "$stats" | cut -d'|' -f1)
            cpu=$(echo "$stats" | cut -d'|' -f2)
            printf "  %-20s %-15s %-10s\n" "$container" "$mem" "$cpu"
            printf "  %-20s %-15s %-10s\n" "$container" "$mem" "$cpu" >> "$RESULTS_FILE"
            found=1
        fi
    done

    if [ "$found" -eq 0 ]; then
        warn "Nisu pronađeni aktivni kontejneri."
        echo "  Nisu pronađeni aktivni kontejneri." >> "$RESULTS_FILE"
    fi

    echo "" >> "$RESULTS_FILE"
}

# ------------------------------------------------------------
# 3. QUERY LATENCY BENCHMARK
# ------------------------------------------------------------
benchmark_query_latency() {
    header "3. QUERY LATENCY (prosjek od $QUERY_REPEATS upita)"
    echo "=== 3. QUERY LATENCY (prosjek od $QUERY_REPEATS upita) ===" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"

    printf "  %-35s %-15s\n" "Upit" "Prosj. latency"
    printf "  %-35s %-15s\n" "-----" "--------------"
    printf "  %-35s %-15s\n" "Upit" "Prosj. latency" >> "$RESULTS_FILE"
    printf "  %-35s %-15s\n" "-----" "--------------" >> "$RESULTS_FILE"

    # --- InfluxDB upiti ---
    if check_service "$INFLUX_URL/health" "InfluxDB"; then

        # Upit 1: Broj svih zapisa (cijeli dataset)
        local influx_count_ms
        influx_count_ms=$(measure_latency "curl -sf -X POST \
            '${INFLUX_URL}/api/v2/query?org=${INFLUXDB_ORG}' \
            -H 'Authorization: Token ${INFLUXDB_TOKEN}' \
            -H 'Content-Type: application/vnd.flux' \
            -d 'from(bucket:\"${INFLUXDB_BUCKET}\") |> range(start: -30d) |> filter(fn: (r) => r._measurement == \"zeek_conn\") |> count()'")
        printf "  %-35s %-15s\n" "InfluxDB: COUNT 30 dana unazad" "${influx_count_ms}ms"
        printf "  %-35s %-15s\n" "InfluxDB: COUNT 30 dana unazad" "${influx_count_ms}ms" >> "$RESULTS_FILE"

        # Upit 2: Top 10 destination IP-ova (cijeli dataset)
        local influx_top10_ms
        influx_top10_ms=$(measure_latency "curl -sf -X POST \
            '${INFLUX_URL}/api/v2/query?org=${INFLUXDB_ORG}' \
            -H 'Authorization: Token ${INFLUXDB_TOKEN}' \
            -H 'Content-Type: application/vnd.flux' \
            -d 'from(bucket:\"${INFLUXDB_BUCKET}\") |> range(start: -30d) |> filter(fn: (r) => r._measurement == \"zeek_conn\") |> group(columns: [\"dst_ip\"]) |> count() |> top(n:10, columns:[\"_value\"])'")
        printf "  %-35s %-15s\n" "InfluxDB: TOP 10 dst_ip" "${influx_top10_ms}ms"
        printf "  %-35s %-15s\n" "InfluxDB: TOP 10 dst_ip" "${influx_top10_ms}ms" >> "$RESULTS_FILE"

        # Upit 3: Agregacija prometa po minutama (cijeli dataset)
        local influx_agg_ms
        influx_agg_ms=$(measure_latency "curl -sf -X POST \
            '${INFLUX_URL}/api/v2/query?org=${INFLUXDB_ORG}' \
            -H 'Authorization: Token ${INFLUXDB_TOKEN}' \
            -H 'Content-Type: application/vnd.flux' \
            -d 'from(bucket:\"${INFLUXDB_BUCKET}\") |> range(start: -30d) |> filter(fn: (r) => r._measurement == \"zeek_conn\") |> aggregateWindow(every: 1m, fn: count)'")
        printf "  %-35s %-15s\n" "InfluxDB: AGG po minutama" "${influx_agg_ms}ms"
        printf "  %-35s %-15s\n" "InfluxDB: AGG po minutama" "${influx_agg_ms}ms" >> "$RESULTS_FILE"
    fi

    echo "" >> "$RESULTS_FILE"

    # --- Elasticsearch upiti ---
    if check_service "$ELASTIC_URL" "Elasticsearch"; then

        # Upit 1: Broj dokumenata
        local elastic_count_ms
        elastic_count_ms=$(measure_latency "curl -sf \
            '${ELASTIC_URL}/zeek-*/_count'")
        printf "  %-35s %-15s\n" "Elastic: COUNT sve" "${elastic_count_ms}ms"
        printf "  %-35s %-15s\n" "Elastic: COUNT sve" "${elastic_count_ms}ms" >> "$RESULTS_FILE"

        # Upit 2: Top 10 destination IP-ova
        local elastic_top10_ms
        elastic_top10_ms=$(measure_latency "curl -sf -X GET \
            '${ELASTIC_URL}/zeek-*/_search' \
            -H 'Content-Type: application/json' \
            -d '{\"size\":0,\"aggs\":{\"top_dst\":{\"terms\":{\"field\":\"dst_ip\",\"size\":10}}}}'")
        printf "  %-35s %-15s\n" "Elastic: TOP 10 dst_ip" "${elastic_top10_ms}ms"
        printf "  %-35s %-15s\n" "Elastic: TOP 10 dst_ip" "${elastic_top10_ms}ms" >> "$RESULTS_FILE"

        # Upit 3: Histogram po minutama
        local elastic_agg_ms
        elastic_agg_ms=$(measure_latency "curl -sf -X GET \
            '${ELASTIC_URL}/zeek-*/_search' \
            -H 'Content-Type: application/json' \
            -d '{\"size\":0,\"aggs\":{\"per_minute\":{\"date_histogram\":{\"field\":\"@timestamp\",\"fixed_interval\":\"1m\"}}}}'")
        printf "  %-35s %-15s\n" "Elastic: AGG po minutama" "${elastic_agg_ms}ms"
        printf "  %-35s %-15s\n" "Elastic: AGG po minutama" "${elastic_agg_ms}ms" >> "$RESULTS_FILE"
    fi

    echo "" >> "$RESULTS_FILE"
}

# ------------------------------------------------------------
# 4. INGESTION BENCHMARK
# ------------------------------------------------------------
benchmark_ingestion() {
    header "4. INGESTION BRZINA"
    echo "=== 4. INGESTION BRZINA ===" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"

    if [ ! -f "zeek-logs/conn.log" ]; then
        warn "conn.log ne postoji, preskačem ingestion benchmark."
        echo "  N/A (conn.log ne postoji)" >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
        return
    fi

    local lines
    lines=$(wc -l < zeek-logs/conn.log)
    local size
    size=$(du -sh zeek-logs/conn.log | cut -f1)

    result "conn.log: $lines linija ($size)"
    echo "  conn.log: $lines linija ($size)" >> "$RESULTS_FILE"

    # InfluxDB — broj točaka trenutno u bazi
    if check_service "$INFLUX_URL/health" "InfluxDB"; then
        local influx_points
        influx_points=$(curl -sf -X POST \
            "${INFLUX_URL}/api/v2/query?org=${INFLUXDB_ORG}" \
            -H "Authorization: Token ${INFLUXDB_TOKEN}" \
            -H "Content-Type: application/vnd.flux" \
            -d "from(bucket:\"${INFLUXDB_BUCKET}\") |> range(start: 1970-01-01T00:00:00Z) |> filter(fn: (r) => r._measurement == \"zeek_conn\") |> count()" \
            2>/dev/null | grep -o '"_value":[0-9]*' | head -1 | cut -d: -f2 || echo "N/A")
        result "InfluxDB točke (zeek_conn): $influx_points"
        echo "  InfluxDB točke (zeek_conn): $influx_points" >> "$RESULTS_FILE"
    fi

    # Elasticsearch — broj dokumenata
    if check_service "$ELASTIC_URL" "Elasticsearch"; then
        local elastic_docs
        elastic_docs=$(curl -sf "${ELASTIC_URL}/zeek-*/_count" \
            2>/dev/null | grep -o '"count":[0-9]*' | cut -d: -f2 || echo "N/A")
        result "Elasticsearch dokumenti: $elastic_docs"
        echo "  Elasticsearch dokumenti: $elastic_docs" >> "$RESULTS_FILE"
    fi

    echo "" >> "$RESULTS_FILE"
}

# ------------------------------------------------------------
# 5. SAŽETAK
# ------------------------------------------------------------
print_summary() {
    header "SAŽETAK"
    echo "=== SAŽETAK ===" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"

    local summary="Benchmark završen: $(date '+%Y-%m-%d %H:%M:%S')
Detaljni rezultati: $RESULTS_FILE"

    echo "$summary"
    echo "$summary" >> "$RESULTS_FILE"
}

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
main() {
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   InfluxDB vs Elasticsearch Benchmark      ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"

    init_results
    benchmark_storage
    benchmark_ram
    benchmark_query_latency
    benchmark_ingestion
    print_summary

    echo ""
    info "Gotovo! Rezultati su zapisani u: ${RESULTS_FILE}"
}

main