import urllib.request, json

# Test health endpoint
r = urllib.request.urlopen("http://127.0.0.1:5000/health")
print("Health:", json.dumps(json.loads(r.read()), indent=2))