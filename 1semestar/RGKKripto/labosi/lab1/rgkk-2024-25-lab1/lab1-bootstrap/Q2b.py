from sys import exit
from bitcoin.core.script import *

from lib.utils import *
from lib.config import (my_private_key, my_public_key, my_address,
                    faucet_address, network_type)
from Q1 import P2PKH_scriptPubKey
from Q2a import Q2a_txout_scriptPubKey


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

txin_scriptPubKey = Q2a_txout_scriptPubKey
######################################################################
# TODO: implementirajte skriptu `scriptSig` kojom cete otkljucati BCY zakljucan
# pomocu skripte `scriptPubKey` u zadatku 2a
txin_scriptSig = [
    # vas kod ide ovdje...
]
######################################################################
txout_scriptPubKey = P2PKH_scriptPubKey(faucet_address)

response = send_from_custom_transaction(
    amount_to_send, txid_to_spend, utxo_index,
    txin_scriptPubKey, txin_scriptSig, txout_scriptPubKey, network_type)
print(response.status_code, response.reason)
print(response.text)
