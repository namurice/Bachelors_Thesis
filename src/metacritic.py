import http.client

conn = http.client.HTTPSConnection("metacriticapi.p.rapidapi.com")

headers = { 'x-rapidapi-host': "metacriticapi.p.rapidapi.com" }

conn.request("GET", "/games/silent-hill-2-2001/user-reviews/?platform=playstation-2&reviews=true", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))