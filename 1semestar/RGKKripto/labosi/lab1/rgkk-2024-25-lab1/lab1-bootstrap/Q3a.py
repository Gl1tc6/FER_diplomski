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
    'cVLWM2LywPJemdBHDcTUqMvb5D9mwNjt85Gc3AmACUtCTSEaoC2o')
cust1_public_key = cust1_private_key.pub
cust2_private_key = CBitcoinSecret(
    'cUKZkHJjUxQCRug1Zczntrs7pgoRRiFpgyz72KV777tdDem9VQLL')
cust2_public_key = cust2_private_key.pub
cust3_private_key = CBitcoinSecret(
    'cPWjSYgx8o3SyhRwwdKd2sjD2FztpVnkbrdXDGENHhezW6L5NSb8')
cust3_public_key = cust3_private_key.pub


######################################################################
# TODO: Implementirajte `scriptPubKey` za zadatak 3

# Pretpostavite da vi igrate ulogu banke u ovom zadatku na nacin da privatni
# kljuc od banke `bank_private_key` odgovara vasem privatnom kljucu
# `my_private_key`.

Q3a_txout_scriptPubKey = [
    # vas kod ide ovdje...
    my_public_key,
    OP_CHECKSIGVERIFY,
    1,
    cust1_public_key,
    cust2_public_key,
    cust3_public_key,
    3,
    OP_CHECKMULTISIG
]
######################################################################

if __name__ == '__main__':
    ######################################################################
    # Postavite parametre transakcije
    # TODO: amount_to_send = {cjelokupni iznos BCY-a u UTXO-u kojeg otkljucavamo} - {fee}
    amount_to_send = 0.000165 - 0.00001
    # TODO: Identifikator transakcije
    txid_to_spend = (
        '1a19256a13c6eea10f83b4144a9d4c59dda11516d7f14b31f37145ead64ed280')
    # TODO: indeks UTXO-a unutar transakcije na koju se referiramo
    # (indeksi pocinju od nula)
    utxo_index = 3
    ######################################################################

    response = send_from_P2PKH_transaction(amount_to_send, txid_to_spend,
                                           utxo_index, Q3a_txout_scriptPubKey,
                                           my_private_key, network_type)
    print(response.status_code, response.reason)
    print(response.text)
