## **Challenge Name**: Echo Chamber

**Category**: Web Exploitation  
**Points**: 200  
**Description**:  
The challenge provides a URL to a web application. The application accepts user input through a form and echoes it back. The goal is to exploit the application and retrieve the flag.

---

## **Task Description**

The URL provided is:

```
http://challenge-site.com/echo
```

When submitting an input like "Hello, World!" through the form, the website responds with:

```tcolorbox
Input: Hello, World!  
Output: You said: Hello, World!  
```

---

## **Progress So Far**

### **Step 1: Capturing and Understanding the Request**

Using **Burp Suite**, I intercepted the request sent to the server when submitting input. Here's the captured HTTP request:

```http
POST /echo HTTP/1.1
Host: challenge-site.com
Content-Type: application/x-www-form-urlencoded
Content-Length: 15

input=Hello+World
```

The server echoes the user input back as part of the response.

### **Step 2: Testing for Input Validation**

I tested for various forms of input to see how the application handles them:

- Regular text: `test` -> Echoes back as expected.
- Special characters: `<>` -> Also echoed back without filtering.
- HTML tags: `<script>alert(1)</script>` -> Echoed back as plain text, no script execution.

From this, it seems the application does not sanitize user input.

---

### **Current Hypothesis**

Since the application echoes back user input, I suspect it might be vulnerable to server-side injection (possibly command injection or template injection).

To test this, I submitted the following payloads:

1. `{{7*7}}` (testing for server-side template injection)
    - Output: `You said: {{7*7}}` (no evaluation).
2. `` `ls` `` (testing for command injection with backticks).
    - Output: `You said: \`ls`` (no execution).
3. `; ls` (testing for command injection with a semicolon).
    - Output: `You said: ; ls` (no execution).

---

### **Stuck at This Point**

I'm unsure how to proceed, as none of the initial injection attempts seem to trigger any vulnerability. My next steps could include:

- Investigating if the application interacts with a backend database (potential SQL injection).
- Exploring alternative payloads for escaping and executing commands.
- Analyzing server-side behavior with further testing of HTTP headers or other methods.