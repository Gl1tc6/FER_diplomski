#!/bin/bash

# ============================================================
# dipl.sh — pokretanje/gašenje kompletne okoline
# Pokreće: AP (net_setReset_rules.sh) + Docker servisi
#
# Korištenje:
#   sudo ./dipl.sh start   — postavlja AP i pokreće kontejnere
#   sudo ./dipl.sh stop    — gasi kontejnere i resetira AP
#   sudo ./dipl.sh restart  — resetira sve servise i AP
#   sudo ./dipl.sh status  — prikazuje stanje svih servisa
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NET_SCRIPT="$SCRIPT_DIR/net_setReset_rules.sh"

# Boje za output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ "$EUID" -ne 0 ]; then
  echo "Start as root (sudo)."
  exit 1
fi

if [ "$1" == "start" ]; then

    echo -e "${YELLOW}[1/2] Postavljanje AP okoline...${NC}"
    bash "$NET_SCRIPT" set
    if [ $? -ne 0 ]; then
        echo -e "${RED}Greška pri postavljanju AP-a. Prekidam.${NC}"
        exit 1
    fi
    echo -e "${GREEN}AP okolina postavljena.${NC}"

    echo -e "${YELLOW}[2/2] Pokretanje Docker servisa...${NC}"
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d
    if [ $? -ne 0 ]; then
        echo -e "${RED}Greška pri pokretanju Docker servisa. Resetiram AP...${NC}"
        bash "$NET_SCRIPT" reset
        exit 1
    fi
    echo -e "${GREEN}Docker servisi pokrenuti.${NC}"

    echo ""
    echo -e "${GREEN}=== Okolina je aktivna ===${NC}"
    echo "  Grafana:  http://localhost:3000"
    echo "  InfluxDB: http://localhost:8086"
    echo ""
    echo "Logovi: docker compose logs -f"
    echo "SSID: Dipl_test_FORBIDDEN  |  pass: ZaDipl_1248"
    firefox http://localhost:3000 http://localhost:8086 &
# ------------------------------------------------------

elif [ "$1" == "stop" ]; then

    echo -e "${YELLOW}[1/2] Gašenje Docker servisa...${NC}"
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" down
    echo -e "${GREEN}Docker servisi ugašeni.${NC}"

    echo -e "${YELLOW}[2/2] Resetiranje AP okoline...${NC}"
    bash "$NET_SCRIPT" reset
    echo -e "${GREEN}AP okolina resetirana.${NC}"

    pkill firefox

    echo ""
    echo -e "${GREEN}=== Okolina je ugašena ===${NC}"
# ------------------------------------------------------
elif [ "$1" == "restart" ]; then
    # zaustavljanje
    echo -e "${YELLOW}[1/2] Gašenje Docker servisa...${NC}"
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" down
    echo -e "${GREEN}Docker servisi ugašeni.${NC}"

    echo -e "${YELLOW}[2/2] Resetiranje AP okoline...${NC}"
    bash "$NET_SCRIPT" reset
    echo -e "${GREEN}AP okolina resetirana.${NC}"

    echo ""
    echo -e "${GREEN}=== Okolina je ugašena ===${NC}"

    # ponovno pokretanje
    echo -e "${YELLOW}[1/2] Postavljanje AP okoline...${NC}"
    bash "$NET_SCRIPT" set
    if [ $? -ne 0 ]; then
        echo -e "${RED}Greška pri postavljanju AP-a. Prekidam.${NC}"
        exit 1
    fi
    echo -e "${GREEN}AP okolina postavljena.${NC}"

    echo -e "${YELLOW}[2/2] Pokretanje Docker servisa...${NC}"
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d
    if [ $? -ne 0 ]; then
        echo -e "${RED}Greška pri pokretanju Docker servisa. Resetiram AP...${NC}"
        bash "$NET_SCRIPT" reset
        exit 1
    fi
    echo -e "${GREEN}Docker servisi pokrenuti.${NC}"

    echo ""
    echo -e "${GREEN}=== Okolina je aktivna ===${NC}"
    echo "  Grafana:  http://localhost:3000"
    echo "  InfluxDB: http://localhost:8086"
    echo ""
    echo "Logovi: docker compose logs -f"
    echo "SSID: Dipl_test_FORBIDDEN  |  pass: ZaDipl_1248"
    firefox http://localhost:3000 http://localhost:8086 &
# ------------------------------------------------------

elif [ "$1" == "status" ]; then

    echo "=== Status okoline ==="
    echo ""

    # AP status
    if systemctl is-active --quiet hostapd; then
        echo -e "AP (hostapd):  ${GREEN}aktivan${NC}"
    else
        echo -e "AP (hostapd):  ${RED}neaktivan${NC}"
    fi

    if systemctl is-active --quiet dnsmasq; then
        echo -e "DHCP/DNS (dnsmasq): ${GREEN}aktivan${NC}"
    else
        echo -e "DHCP/DNS (dnsmasq): ${RED}neaktivan${NC}"
    fi

    echo ""

    # Docker status
    echo "Docker servisi:"
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" ps --format "table {{.Name}}\t{{.Status}}"

else
    echo "Korištenje: sudo ./dipl.sh [start|stop|restart|status]"
    echo ""
    echo "  start   — postavlja AP i pokreće sve Docker servise"
    echo "  stop    — gasi Docker servise i resetira AP"
    echo "  restart — pokreće stop pa start"
    echo "  status  — prikazuje stanje AP-a i Docker servisa"
fi
