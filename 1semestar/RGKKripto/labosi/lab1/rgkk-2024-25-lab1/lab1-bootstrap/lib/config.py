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
    x('1d1751e638aa9061df434d4d2b37df68b753a53c15fced1ee133d20807bddf31'))

my_public_key = my_private_key.pub
my_address = P2PKHBitcoinAddress.from_pubkey(my_public_key)
