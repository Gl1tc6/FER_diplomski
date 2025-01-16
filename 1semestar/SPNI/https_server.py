import http.server
import ssl

# Lozinka za pristup
AUTH_PASSWORD = "1234"

class AuthHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Provjeri postoji li lozinka u URL-u
        query = self.path.split('?')[-1]
        params = dict(param.split('=') for param in query.split('&') if '=' in param)
        password = params.get('password', '')

        if password == AUTH_PASSWORD:
            # Ako je lozinka točna, omogući pristup
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Dobrodošli na sigurnu stranicu!")
        else:
            # Ako lozinka nije točna, traži unos lozinke
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b"Unauthorized. Dodajte ?password=<lozinka> u URL.")

# Postavljanje servera
httpd = http.server.HTTPServer(('0.0.0.0', 8443), AuthHTTPRequestHandler)
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(certfile="server.pem", keyfile="server.pem")
httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

print("Server pokrenut na https://0.0.0.0:8443")
httpd.serve_forever()
