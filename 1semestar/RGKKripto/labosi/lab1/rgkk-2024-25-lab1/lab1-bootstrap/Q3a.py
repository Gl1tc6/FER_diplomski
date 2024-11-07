from sys import exit
from bitcoin.core.script import *
from bitcoin.wallet import CBitcoinSecret

from lib.utils import *
from lib.config import (my_private_key, my_public_key, my_address,
                        faucet_address, network_type)
from Q1 import send_from_P2PKH_transaction


# TODO: Generirajte privatne kljuceve od klijenata koristeci `lib/keygen.py`
# i dodajte ih ovdje.
cust1_private_key = CBitcoinSecret(
    'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
cust1_public_key = cust1_private_key.pub
cust2_private_key = CBitcoinSecret(
    'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
cust2_public_key = cust2_private_key.pub
cust3_private_key = CBitcoinSecret(
    'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
cust3_public_key = cust3_private_key.pub


######################################################################
# TODO: Implementirajte `scriptPubKey` za zadatak 3

# Pretpostavite da vi igrate ulogu banke u ovom zadatku na nacin da privatni
# kljuc od banke `bank_private_key` odgovara vasem privatnom kljucu
# `my_private_key`.

Q3a_txout_scriptPubKey = [
    # vas kod ide ovdje...
]
######################################################################

if __name__ == '__main__':
    ######################################################################
    # Postavite parametre transakcije
    # TODO: amount_to_send = {cjelokupni iznos BCY-a u UTXO-u kojeg otkljucavamo} - {fee}
    amount_to_send = None
    # TODO: Identifikator transakcije
    txid_to_spend = (
        'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    # TODO: indeks UTXO-a unutar transakcije na koju se referiramo
    # (indeksi pocinju od nula)
    utxo_index = None
    ######################################################################

    response = send_from_P2PKH_transaction(amount_to_send, txid_to_spend,
                                           utxo_index, Q3a_txout_scriptPubKey,
                                           my_private_key, network_type)
    print(response.status_code, response.reason)
    print(response.text)
