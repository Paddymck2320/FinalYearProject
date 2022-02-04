import requests
from bs4 import BeautifulSoup
from csv import writer

url = "https://www.pro-football-reference.com/years/2021/#team_stats"
response = requests.get(url).text

soup = BeautifulSoup(response, "html.parser").table
tabl = soup.find("table", attrs={'id':'team_stats'})
print(tabl)

header = []

for i in tabl.find_All('th'):
    header.append(i.text)

with open("team_offensive_table.csv", "wt", newline='', encoding='utf-8') as csv_file:
    csv_writer = writer(csv_file, delimiter='|')
    csv_writer.writerow(header)

    for row in tabl.find_All('tr')[1:]:
        td = row.find_All('td')
        r = [i.text.replace('\n','') for i in td]
        csv_writer.writerow(r)