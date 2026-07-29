# AES128_V2 — Automation Anywhere Package

Secure encryption and decryption custom package for **Automation Anywhere 360** using AES-128-CBC + PBKDF2.

---

## What It Does

Two commands for AA360 bot workflows:

| Command   | Inputs                                          | Output          |
|-----------|-------------------------------------------------|-----------------|
| **Encrypt** | `String to encrypt` (Credential) + `Password`   | Base64 ciphertext string |
| **Decrypt** | `Encrypted string` (Text) + `Password`          | Original plaintext string |

Use it to protect sensitive data — API keys, PII, tokens, credentials — inside bot pipelines.

---

## Algorithm

- **Cipher**: AES/CBC/PKCS5Padding (128-bit)
- **Key Derivation**: PBKDF2-HMAC-SHA1
  - Salt: 20 random bytes (generated per encryption)
  - Iterations: 50
- **IV**: 16 random bytes (generated per encryption)
- **Output Layout** (Base64-encoded): `[salt 20B][iv 16B][ciphertext]`

---

## Build the JAR

Java 11+ required. Gradle wrapper is included.

```powershell
# Windows
cd AES128
.\gradlew.bat shadowJar

# macOS / Linux
cd AES128
./gradlew shadowJar
```

Output JAR → **`build/libs/AES128_V2-2.7.0.jar`**

---

## Install in Automation Anywhere

1. Open **Control Room → Packages**
2. Click **Upload package**
3. Select `AES128_V2-2.7.0.jar`
4. **Encrypt** and **Decrypt** actions appear in your Action Library

---

## Usage

### Encrypt a Value
1. Drag **Encrypt** into your bot flow
2. Select a Credential Vault entry for **String to encrypt (Credential)**
3. Type a strong **Password** (used to derive the AES key)
4. Assign the result to a string variable — it will hold the Base64-encoded encrypted data

### Decrypt a Value
1. Drag **Decrypt** into your bot flow
2. Paste the Base64 output from the **Encrypt** step
3. Enter the **same password** used during encryption
4. Assign the result to a string variable to get back the original plaintext

---

## Error Handling

If something fails (wrong password, malformed input, etc.), both commands return a string prefixed with `ER001:`. You can check for this prefix in your bot flow to catch errors.

---

## Project Structure

```
AES128/
├── src/main/java/com/automationanywhere/botcommand/samples/commands/basic/
│   ├── Encrypt.java
│   └── Decrypt.java
├── src/main/resources/        # icons, locales, i18n messages
├── libs/                      # AA SDK jars (compile-only + runtime)
├── build.gradle               # Gradle build config (Shadow + AA codegen)
└── gradlew / gradlew.bat      # Gradle wrapper
```
