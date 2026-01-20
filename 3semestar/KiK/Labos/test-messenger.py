#!/usr/bin/env python3

import os
import pickle
import unittest
from messenger import (
    Messenger,
    gov_decrypt,
)
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import (
    hashes,
)

def generate_p384_key_pair():
    secret_key = ec.generate_private_key(ec.SECP384R1())
    public_key = secret_key.public_key()
    return (secret_key, public_key)

def sign_with_ecdsa(secret_key, data):
    signature = secret_key.sign(data, ec.ECDSA(hashes.SHA256()))
    return signature

class TestMessenger(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Par kljuceva od CA
        cls.ca_sk, cls.ca_pk = generate_p384_key_pair()
        # Par kljuceva od vlade
        cls.gov_sk, cls.gov_pk = generate_p384_key_pair()

    def test_import_certificate_without_error(self):

        alice = Messenger('Alice', self.ca_pk, self.gov_pk)
        bob = Messenger('Bob', self.ca_pk, self.gov_pk)

        alice_cert = alice.generate_certificate()
        bob_cert = bob.generate_certificate()

        alice_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(alice_cert))
        bob_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(bob_cert))

        alice.receive_certificate(bob_cert, bob_cert_sign)
        bob.receive_certificate(alice_cert, alice_cert_sign)

    def test_send_message_without_error(self):

        lautrec = Messenger('Lautrec', self.ca_pk, self.gov_pk)
        player = Messenger('Player', self.ca_pk, self.gov_pk)

        lautrec_cert = lautrec.generate_certificate()
        player_cert = player.generate_certificate()

        lautrec_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(lautrec_cert))
        player_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(player_cert))

        lautrec.receive_certificate(player_cert, player_cert_sign)
        player.receive_certificate(lautrec_cert, lautrec_cert_sign)

        message = "Our futures are murky. Let's not be too friendly, now."
        lautrec.send_message('Player', message)

    def test_encrypted_message_can_be_decrypted(self):

        newman = Messenger('Newman', self.ca_pk, self.gov_pk)
        jerry = Messenger('Jerry', self.ca_pk, self.gov_pk)

        newman_cert = newman.generate_certificate()
        jerry_cert = jerry.generate_certificate()

        newman_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(newman_cert))
        jerry_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(jerry_cert))

        newman.receive_certificate(jerry_cert, jerry_cert_sign)
        jerry.receive_certificate(newman_cert, newman_cert_sign)

        plaintext = 'Just remember, when you control the mail, you control... information.'
        message = newman.send_message('Jerry', plaintext)

        result = jerry.receive_message('Newman', message)
        self.assertEqual(plaintext, result)

    def test_government_can_decrypt(self):

        alice = Messenger('Alice', self.ca_pk, self.gov_pk)
        bob = Messenger('Bob', self.ca_pk, self.gov_pk)

        alice_cert = alice.generate_certificate()
        bob_cert = bob.generate_certificate()

        alice_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(alice_cert))
        bob_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(bob_cert))

        alice.receive_certificate(bob_cert, bob_cert_sign)
        bob.receive_certificate(alice_cert, alice_cert_sign)

        plaintext1 = 'Hi Bob!'
        message = alice.send_message('Bob', plaintext1)
        plaintext2 = gov_decrypt(self.gov_sk, message)
        self.assertEqual(plaintext1, plaintext2)

    def test_user_can_send_stream_of_messages_without_response(self):

        indigo = Messenger('Indigo', self.ca_pk, self.gov_pk)
        rugen = Messenger('Rugen', self.ca_pk, self.gov_pk)

        indigo_cert = indigo.generate_certificate()
        rugen_cert = rugen.generate_certificate()

        indigo_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(indigo_cert))
        rugen_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(rugen_cert))

        indigo.receive_certificate(rugen_cert, rugen_cert_sign)
        rugen.receive_certificate(indigo_cert, indigo_cert_sign)

        plaintext = 'Hello.'
        message = indigo.send_message('Rugen', plaintext)

        result = rugen.receive_message('Indigo', message)
        self.assertEqual(plaintext, result)

        plaintext = 'My name is Inigo Montoya.'
        message = indigo.send_message('Rugen', plaintext)

        result = rugen.receive_message('Indigo', message)
        self.assertEqual(plaintext, result)

        plaintext = 'You killed my father.'
        message = indigo.send_message('Rugen', plaintext)

        result = rugen.receive_message('Indigo', message)
        self.assertEqual(plaintext, result)

        plaintext = 'Prepare to die.'
        message = indigo.send_message('Rugen', plaintext)

        result = rugen.receive_message('Indigo', message)
        self.assertEqual(plaintext, result)

    def test_conversation(self):

        alice = Messenger('Alice', self.ca_pk, self.gov_pk)
        bob = Messenger('Bob', self.ca_pk, self.gov_pk)

        alice_cert = alice.generate_certificate()
        bob_cert = bob.generate_certificate()

        alice_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(alice_cert))
        bob_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(bob_cert))

        alice.receive_certificate(bob_cert, bob_cert_sign)
        bob.receive_certificate(alice_cert, alice_cert_sign)

        plaintext = 'Hi Bob!'
        message = alice.send_message('Bob', plaintext)

        result = gov_decrypt(self.gov_sk, message)
        self.assertEqual(result, plaintext)

        result = bob.receive_message('Alice', message)
        self.assertEqual(plaintext, result)

        plaintext = 'Hey Alice!'
        message = bob.send_message('Alice', plaintext)

        result = alice.receive_message('Bob', message)
        self.assertEqual(plaintext, result)

        plaintext = 'Are you studying for the exam tomorrow?'
        message = bob.send_message('Alice', plaintext)

        result = alice.receive_message('Bob', message)
        self.assertEqual(plaintext, result)

        plaintext = 'Yes. How about you?'
        message = alice.send_message('Bob', plaintext)

        result = bob.receive_message('Alice', message)
        self.assertEqual(plaintext, result)

    def test_conversation_between_multiple_users(self):

        alice = Messenger('Alice', self.ca_pk, self.gov_pk)
        bob = Messenger('Bob', self.ca_pk, self.gov_pk)
        eve = Messenger('Eve', self.ca_pk, self.gov_pk)

        alice_cert = alice.generate_certificate()
        bob_cert = bob.generate_certificate()
        eve_cert = eve.generate_certificate()

        alice_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(alice_cert))
        bob_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(bob_cert))
        eve_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(eve_cert))

        alice.receive_certificate(bob_cert, bob_cert_sign)
        alice.receive_certificate(eve_cert, eve_cert_sign)

        bob.receive_certificate(alice_cert, alice_cert_sign)
        bob.receive_certificate(eve_cert, eve_cert_sign)

        eve.receive_certificate(alice_cert, alice_cert_sign)
        eve.receive_certificate(bob_cert, bob_cert_sign)

        plaintext = 'Hi Alice!'
        message = bob.send_message('Alice', plaintext)
        result = alice.receive_message('Bob', message)
        self.assertEqual(plaintext, result)

        plaintext = 'Hello Bob'
        message = alice.send_message('Bob', plaintext)
        result = bob.receive_message('Alice', message)
        self.assertEqual(plaintext, result)

        plaintext = 'What are you doing?'
        message = bob.send_message('Alice', plaintext)
        result = alice.receive_message('Bob', message)
        self.assertEqual(plaintext, result)

        plaintext = "I'm woking on my homework"
        message = alice.send_message('Bob', plaintext)
        result = bob.receive_message('Alice', message)
        self.assertEqual(plaintext, result)

        plaintext = 'Alice is doing her homework. What are you doing Eve?'
        message = bob.send_message('Eve', plaintext)
        result = eve.receive_message('Bob', message)
        self.assertEqual(plaintext, result)

        plaintext = "Hi Bob! I'm studying for the exam"
        message = eve.send_message('Bob', plaintext)
        result = bob.receive_message('Eve', message)
        self.assertEqual(plaintext, result)

        plaintext = "How's the homework going Alice"
        message = eve.send_message('Alice', plaintext)
        result = alice.receive_message('Eve', message)
        self.assertEqual(plaintext, result)

        plaintext = "I just finished it"
        message = alice.send_message('Eve', plaintext)
        result = eve.receive_message('Alice', message)
        self.assertEqual(plaintext, result)


    def test_user_can_send_stream_of_messages_with_infrequent_responses(self):

        alice = Messenger('Alice', self.ca_pk, self.gov_pk)
        bob = Messenger('Bob', self.ca_pk, self.gov_pk)

        alice_cert = alice.generate_certificate()
        bob_cert = bob.generate_certificate()

        alice_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(alice_cert))
        bob_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(bob_cert))

        alice.receive_certificate(bob_cert, bob_cert_sign)
        bob.receive_certificate(alice_cert, alice_cert_sign)

        for i in range(0, 2):
            for j in range(0, 4):
                plaintext = f'{j}) Hi Bob!'
                message = alice.send_message('Bob', plaintext)
                result = bob.receive_message('Alice', message)
                self.assertEqual(plaintext, result)

            plaintext = f'{i}) Hello Alice!'
            message = bob.send_message('Alice', plaintext)
            result = alice.receive_message('Bob', message)
            self.assertEqual(plaintext, result)

    def test_reject_message_from_unknown_user(self):

        alice = Messenger('Alice', self.ca_pk, self.gov_pk)
        bob = Messenger('Bob', self.ca_pk, self.gov_pk)

        alice_cert = alice.generate_certificate()
        bob_cert = bob.generate_certificate()

        alice_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(alice_cert))
        bob_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(bob_cert))

        bob.receive_certificate(alice_cert, alice_cert_sign)

        plaintext = 'Hi Alice!'
        message = bob.send_message('Alice', plaintext)
        self.assertRaises(Exception, alice.receive_message, 'Bob', message)

    def test_replay_attacks_are_detected(self):

        alice = Messenger('Alice', self.ca_pk, self.gov_pk)
        bob = Messenger('Bob', self.ca_pk, self.gov_pk)

        alice_cert = alice.generate_certificate()
        bob_cert = bob.generate_certificate()

        alice_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(alice_cert))
        bob_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(bob_cert))

        alice.receive_certificate(bob_cert, bob_cert_sign)
        bob.receive_certificate(alice_cert, alice_cert_sign)

        plaintext = 'Hi Alice!'
        message = bob.send_message('Alice', plaintext)
        alice.receive_message('Bob', message)

        self.assertRaises(Exception, alice.receive_message, 'Bob', message)

    def test_out_of_order_messages(self):

        alice = Messenger('Alice', self.ca_pk, self.gov_pk)
        bob = Messenger('Bob', self.ca_pk, self.gov_pk)

        alice_cert = alice.generate_certificate()
        bob_cert = bob.generate_certificate()

        alice_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(alice_cert))
        bob_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(bob_cert))

        alice.receive_certificate(bob_cert, bob_cert_sign)
        bob.receive_certificate(alice_cert, alice_cert_sign)

        plaintext1 = 'Hi Bob!'
        message1 = alice.send_message('Bob', plaintext1)
        plaintext2 = 'Bob?'
        message2 = alice.send_message('Bob', plaintext2)
        plaintext3 = 'BOB'
        message3 = alice.send_message('Bob', plaintext3)

        result = bob.receive_message('Alice', message1)
        self.assertEqual(plaintext1, result)

        result = bob.receive_message('Alice', message3)
        self.assertEqual(plaintext3, result)

        result = bob.receive_message('Alice', message2)
        self.assertEqual(plaintext2, result)

    def test_more_out_of_order_messages(self):

        colonel = Messenger('Colonel', self.ca_pk, self.gov_pk)
        snake   = Messenger('Snake', self.ca_pk, self.gov_pk)

        colonel_cert = colonel.generate_certificate()
        snake_cert = snake.generate_certificate()

        colonel_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(colonel_cert))
        snake_cert_sign = sign_with_ecdsa(self.ca_sk, pickle.dumps(snake_cert))

        colonel.receive_certificate(snake_cert, snake_cert_sign)
        snake.receive_certificate(colonel_cert, colonel_cert_sign)

        plaintext1 = 'Snake?'
        message1 = colonel.send_message('Snake', plaintext1)

        plaintext2 = 'Snake!?'
        message2 = colonel.send_message('Snake', plaintext2)

        plaintext3 = 'SNAAAAAAAAAAAKE!'
        message3 = colonel.send_message('Snake', plaintext3)

        result = snake.receive_message('Colonel', message3)
        self.assertEqual(plaintext3, result)

        result = snake.receive_message('Colonel', message2)
        self.assertEqual(plaintext2, result)

        result = snake.receive_message('Colonel', message1)
        self.assertEqual(plaintext1, result)

if __name__ == "__main__":
    unittest.main(verbosity=2)
