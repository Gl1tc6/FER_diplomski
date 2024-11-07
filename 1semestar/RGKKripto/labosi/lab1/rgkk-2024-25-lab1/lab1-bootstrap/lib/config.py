from bitcoin import SelectParams
from bitcoin.base58 import decode
from bitcoin.core import x
from bitcoin.wallet import CBitcoinAddress, CBitcoinSecret, P2PKHBitcoinAddress

SelectParams('testnet')

faucet_address = CBitcoinAddress('mohjSavDdQYHRYXcS3uS6ttaHP8amyvX78')

# Koristimo BlockCypher Test Chain
network_type = 'bcy-test'

########################################################
# TODO: Nadopunite skriptu sa svojim privatnim kljucem #
########################################################

my_private_key = CBitcoinSecret.from_secret_bytes(
    x('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'))

my_public_key = my_private_key.pub
my_address = P2PKHBitcoinAddress.from_pubkey(my_public_key)
