@load base/protocols/conn
@load base/protocols/dns
@load base/protocols/http
@load base/protocols/ssl
@load base/frameworks/notice

# prebacuje sve logove u JSON format
# (umjesto defaultnog TSV formata)
@load policy/tuning/json-logs

# za detekciju skenova i bf
# @load policy/misc/scan
@load policy/protocols/ssh/detect-bruteforcing