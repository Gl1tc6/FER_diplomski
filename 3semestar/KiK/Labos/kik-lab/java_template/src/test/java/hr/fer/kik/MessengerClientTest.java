package hr.fer.kik;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import com.google.crypto.tink.Config;
import com.google.crypto.tink.KeysetHandle;
import com.google.crypto.tink.KeyTemplates;
import com.google.crypto.tink.PublicKeySign;


import com.google.crypto.tink.signature.SignatureConfig;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.function.ThrowingRunnable;

import java.nio.charset.Charset;

public class MessengerClientTest {

	private KeysetHandle caPublicKeysetHandle;
	private KeysetHandle govPrivateKeysetHandle;
	private MessengerClient alice;
	private MessengerClient bob;
	private MessengerClient eve;

	// TODO: Funkcija `govDecrypt` dekriptira poruku `message` i vraća njen sadržaj

	// Dekripcija poruke unutar kriptosustava javnog kljuca `Elgamal`
	// gdje, umjesto kriptiranja jasnog teksta množenjem u Z_p, jasni tekst
	// kriptiramo koristeci simetricnu sifru AES-GCM.

	// Procitati poglavlje `The Elgamal Encryption Scheme` u udzbeniku
	// `Understanding Cryptography` (Christof Paar , Jan Pelzl) te obratiti
	// pozornost na `Elgamal Encryption Protocol`

	// Dakle, funkcija treba:
	// 1. Izracunati `masking key` `k_M` koristeci privatni kljuc `govPrivateKeysetHandle` i javni
	//    kljuc `govPubKey` koji se nalazi u zaglavlju `header`.
	// 2. Iz `k_M` derivirati kljuc `k` za AES-GCM koristeci odgovarajucu funkciju za
	//    derivaciju kljuca.
	// 3. Koristeci `k` i AES-GCM dekriptirati `govCt` iz zaglavlja da se dobije
	//    `sending (message) key` `mk`
	// 4. Koristeci `mk` i AES-GCM dekriptirati sifrat `ciphertext` orginalne poruke.
	// 5. Vratiti tako dobiveni jasni tekst.

	// Naravno, lokalne varijable mozete proizvoljno imenovati.  Zaglavlje
	// poruke treba sadrzavati polja `govPubKey`, `govIv` i `govCt`.
	public String govDecrypt(byte[] message) {
		// Dummy implementation
		return new String(message, Charset.defaultCharset());
	}

	@Before
	public void setUp() throws Exception {

		Config.register(SignatureConfig.LATEST);

		KeysetHandle caPrivateKeysetHandle = KeysetHandle.generateNew(KeyTemplates.get("ECDSA_P256"));
		caPublicKeysetHandle = caPrivateKeysetHandle.getPublicKeysetHandle();
		govPrivateKeysetHandle = KeysetHandle.generateNew(KeyTemplates.get("ECDSA_P256"));
		KeysetHandle govPublicKeysetHandle = govPrivateKeysetHandle.getPublicKeysetHandle();

		alice = new MessengerClient("Alice", caPublicKeysetHandle, govPublicKeysetHandle);
		bob = new MessengerClient("Bob", caPublicKeysetHandle, govPublicKeysetHandle);
		eve = new MessengerClient("Eve", caPublicKeysetHandle, govPublicKeysetHandle);

		byte[] aliceCert = alice.generateCertificate();
		byte[] bobCert = bob.generateCertificate();
		byte[] eveCert = eve.generateCertificate();

		PublicKeySign signer = caPrivateKeysetHandle.getPrimitive(PublicKeySign.class);
		byte[] aliceCertSignature = signer.sign(aliceCert);
		byte[] bobCertSignature = signer.sign(bobCert);
		byte[] eveCertSignature = signer.sign(eveCert);

		alice.receiveCertificate(bobCert, bobCertSignature);
		alice.receiveCertificate(eveCert, eveCertSignature);

		bob.receiveCertificate(aliceCert, aliceCertSignature);
		bob.receiveCertificate(eveCert, eveCertSignature);

		eve.receiveCertificate(aliceCert, aliceCertSignature);
		eve.receiveCertificate(bobCert, bobCertSignature);
	}

	@After
	public void tearDown() throws Exception {
	}

	@Test
	public void testSendMessageWithoutError() throws Exception {
		alice.sendMessage("Bob", "Hi Bob!");
	}

	@Test
	public void testEncryptedMessageCanBeDecrypted() throws Exception {
		String plaintext = "Hi Bob!";
		byte[] message = alice.sendMessage("Bob", plaintext);
		String result = bob.receiveMessage("Alice", message);
		assertEquals(plaintext, result);
	}

	@Test
	public void testConversationBetweenMultipleUsers() throws Exception {
		String plaintext = "Hi Alice!";
		byte[] message = bob.sendMessage("Alice", plaintext);
		String result = alice.receiveMessage("Bob", message);
		assertEquals(plaintext, result);

		plaintext = "Hello Bob";
		message = alice.sendMessage("Bob", plaintext);
		result = bob.receiveMessage("Alice", message);
		assertEquals(plaintext, result);

		plaintext = "What are you doing?";
		message = bob.sendMessage("Alice", plaintext);
		result = alice.receiveMessage("Bob", message);
		assertEquals(plaintext, result);

		plaintext = "I'm woking on my homework";
		message = alice.sendMessage("Bob", plaintext);
		result = bob.receiveMessage("Alice", message);
		assertEquals(plaintext, result);

		plaintext = "Alice is doing her homework. What are you doing Eve?";
		message = bob.sendMessage("Eve", plaintext);
		result = eve.receiveMessage("Bob", message);
		assertEquals(plaintext, result);

		plaintext = "Hi Bob! I'm studying for the exam";
		message = eve.sendMessage("Bob", plaintext);
		result = bob.receiveMessage("Eve", message);
		assertEquals(plaintext, result);

		plaintext = "How's the homework going Alice";
		message = eve.sendMessage("Alice", plaintext);
		result = alice.receiveMessage("Eve", message);
		assertEquals(plaintext, result);

		plaintext = "I just finished it";
		message = alice.sendMessage("Eve", plaintext);
		result = eve.receiveMessage("Alice", message);
		assertEquals(plaintext, result);
	}

	@Test
	public void testUserCanSendStreamOfMessagesWithoutResponse() throws Exception {
		String plaintext = "Hi Bob!";
		byte[] message = alice.sendMessage("Bob", plaintext);
		String result = bob.receiveMessage("Alice", message);
		assertEquals(plaintext, result);

		plaintext = "Hi Bob!";
		message = alice.sendMessage("Bob", plaintext);
		result = bob.receiveMessage("Alice", message);
		assertEquals(plaintext, result);

		plaintext = "Hi Bob!";
		message = alice.sendMessage("Bob", plaintext);
		result = bob.receiveMessage("Alice", message);
		assertEquals(plaintext, result);

		plaintext = "Hi Bob!";
		message = alice.sendMessage("Bob", plaintext);
		result = bob.receiveMessage("Alice", message);
		assertEquals(plaintext, result);
	}

	@Test
	public void testUserCanSendStreamOfMessagesWithInfrequentResponses() throws Exception {
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 4; j++) {
				String plaintext = Integer.toString(j) + " Hi Bob!";
				byte[] message = alice.sendMessage("Bob", plaintext);
				String result = bob.receiveMessage("Alice", message);
				assertEquals(plaintext, result);
			}
			String plaintext = Integer.toString(i) + " Hello Alice!";
			byte[] message = bob.sendMessage("Alice", plaintext);
			String result = alice.receiveMessage("Bob", message);
			assertEquals(plaintext, result);
		}
	}

	@Test
	public void testRejectMessageFromUnknownUser() throws Exception {
		String plaintext = "Hi Alice!";
		final byte[] message = bob.sendMessage("Alice", plaintext);
		assertThrows(Exception.class, new ThrowingRunnable() {
			public void run() throws Throwable {
				alice.receiveMessage("Unknown", message);
			}
		});
	}

	@Test
	public void testReplayAttacksAreDetected() throws Exception {
		String plaintext = "Hi Alice!";
		final byte[] message = bob.sendMessage("Alice", plaintext);
		String result = alice.receiveMessage("Bob", message);
		assertThrows(Exception.class, new ThrowingRunnable() {
			public void run() throws Throwable {
				alice.receiveMessage("Bob", message);
			}
		});
	}

	@Test
	public void testGovernmentCanDecrypt() throws Exception {
		String plaintext = "Hi Alice!";
		final byte[] message = bob.sendMessage("Alice", plaintext);
		String result = govDecrypt(message);
		assertEquals(plaintext, result);
	}

	@Test
	public void testOutOfOrderMessages() throws Exception {
		String plaintext1 = "Hi Alice!";
		String plaintext2 = "Alice?";
		String plaintext3 = "ALICE!?";
		final byte[] message1 = bob.sendMessage("Alice", plaintext1);
		final byte[] message2 = bob.sendMessage("Alice", plaintext2);
		final byte[] message3 = bob.sendMessage("Alice", plaintext3);

		String result1 = bob.receiveMessage("Alice", message1);
		assertEquals(plaintext1, result1);

		String result3 = bob.receiveMessage("Alice", message3);
		assertEquals(plaintext3, result3);

		String result2 = bob.receiveMessage("Alice", message2);
		assertEquals(plaintext2, result2);
	}

}
