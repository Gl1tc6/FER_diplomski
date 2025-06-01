#!/bin/bash

# Jednostavnija verzija s /usr/bin/time
PROGRAM="./connect4_mpi"
INPUT_FILE="ploca.txt"
DEPTH=8

echo "=== Connect4 MPI Benchmark ==="
echo "Dubina: $DEPTH"
echo ""

# Stvori CSV
echo "#Proces,time_seconds" > results.csv

for p in 1 2 3 4 5 6; do
    echo "================================"
    echo "Testiram s $p procesora"
    echo "================================"
    
    # Jednostavan pristup s /usr/bin/time
    echo "Pokrećem mjerenje..."
    vrijeme=$(/usr/bin/time -f "%e" mpirun -np $p $PROGRAM $INPUT_FILE $DEPTH 2>&1 >/dev/null)
    
    echo ""
    echo ">>> $p procesora: ${vrijeme} sekundi <<<"
    echo ""
    
    # Spremi u CSV
    echo "$p,$vrijeme" >> results.csv
    
    # Čekaj Enter (osim za zadnji)
    if [ $p -lt 6 ]; then
        echo "Pritisnite Enter za sljedeći test..."
        read
    fi
done

echo ""
echo "GOTOVO! Rezultati u results.csv:"
cat results.csv