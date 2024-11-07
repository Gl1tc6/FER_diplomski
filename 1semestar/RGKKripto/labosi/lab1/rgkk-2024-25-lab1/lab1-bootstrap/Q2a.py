from sys import exit
from bitcoin.core.script import *

from lib.utils import *
from lib.config import (my_private_key, my_public_key, my_address,
                    faucet_address, network_type)
from Q1 import send_from_P2PKH_transaction


######################################################################
# TODO: Implementirajte `scriptPubKey` za zadatak 2
Q2a_txout_scriptPubKey = [
    # vas kod ide ovdje...
]
######################################################################

if __name__ == '__main__':
    ######################################################################
    # Postavite parametre transakcije
    # TODO: amount_to_send = {cjelokupni iznos BCY-a u UTXO-u kojeg saljemo} - {fee}
    amount_to_send = None
    # TODO: Identifikator transakcije
    txid_to_spend = (
        'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    # TODO: indeks UTXO-a unutar transakcije na koju se referiramo
    # (indeksi pocinju od nula)
    utxo_index = None
    ######################################################################

    response = send_from_P2PKH_transaction(
        amount_to_send, txid_to_spend, utxo_index,
        Q2a_txout_scriptPubKey, my_private_key, network_type)
    print(response.status_code, response.reason)
    print(response.text)
