# AES128_V2 — Automation Anywhere Package

Secure encryption and decryption custom package for **Automation Anywhere 360** using AES-128-CBC with PBKDF2 key derivation.

---

## ✨ What It Does

Provides two bot commands that you can drop directly into your AA360 workflows:

| Command     | Description                                                            |
| ----------- | ---------------------------------------------------------------------- |
| **Encrypt** | Encrypts a plaintext string using a password → returns a Base64 string |
| **Decrypt** | Decrypts the Base64 string back to plaintext using the same password   |

Use it to protect sensitive data (API keys, PII, credentials, etc.) inside your bot pipelines.

---

## 🔐 Algorithm Details

- **Cipher**: AES/CBC/PKCS5Padding (128-bit key)
- **Key Derivation**: PBKDF2 with HMAC-SHA1
  - Salt: 20 random bytes (per encryption)
  - Iterations: 50
- **IV**: 16 random bytes (generated per encryption)
- **Output Format**: Base64-encoded `[salt(20)] [iv(16)] [ciphertext]`

---

## 🚀 Build the JAR

Requires **Java 11+**. Gradle wrapper is included.

```powershell
# Windows
cd AES128
.\gradlew.bat shadowJar

# macOS / Linux
cd AES128
./gradlew shadowJar
```

Output: **`build/libs/AES128_V2-2.7.0.jar`**

---

## 📥 Install in Automation Anywhere

1. Go to **Control Room → Packages**
2. Click **Upload package**
3. Select the generated `AES128_V2-2.7.0.jar`
4. The **Encrypt** and **Decrypt** actions appear under the package in your Action Library

---

## 🎯 Usage in a Bot

### Encrypt Action

1. Drag **Encrypt** into your workflow
2. Fill in:
   - **String to encrypt** — the text you want to protect
   - **Password** — secret key (used to derive the AES key)
3. Assign the output to a string variable (stores Base64 ciphertext)

### Decrypt Action

1. Drag **Decrypt** into your workflow
2. Fill in:
   - **Encrypted string** — the Base64 output from Encrypt
   - **Password** — **must match** the password used during encryption
3. Assign the output to a string variable (recovers original plaintext)

---

## 📁 Project Structure

```
AES128/
├── src/main/java/.../commands/basic/
│   ├── Encrypt.java   # Encryption command
│   └── Decrypt.java   # Decryption command
├── src/main/resources/   # icons, locales, messages
├── libs/                 # AA SDK jars
├── build.gradle          # Shadow Jar + AA codegen config
└── gradlew / gradlew.bat # Gradle wrapper
```

---

## ⚠️ Error Handling

Both commands return `ER001: <message>` as the string value when an exception occurs (wrong password, malformed input, etc.), so you can check for the `ER001` prefix in your bot flow to handle failures gracefully.

---

## 🛠 Tech Stack

- Java 11
- Gradle 7.2 (wrapper included)
- Gradle Shadow Plugin — creates a single fat JAR with dependencies
- Automation Anywhere Command SDK (compile-only + runtime)
- Apache Commons Codec (Base64)
