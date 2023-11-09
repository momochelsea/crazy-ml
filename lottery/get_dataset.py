from bs4 import BeautifulSoup
import requests
import csv
import os
import pandas as pd
import codecs


url = "http://datachart.500.com/dlt/history/newinc/history.php?start=07001&end=23128"
r = requests.get(url)
r.encoding = "utf-8"
text = r.text
soup = BeautifulSoup(text, "html.parser")
tbody = soup.find("tbody", id="tdata")
tr = tbody.find_all("tr")
td = tr[0].find_all("td")

lst = []
for page in range(len(tr)):
    td = tr[page].find_all("td")

    lst.append(
        [
            td[0].text,
            td[1].text,
            td[2].text,
            td[3].text,
            td[4].text,
            td[5].text,
            td[6].text,
            td[7].text,
            td[8].text,
            td[9].text,
            td[10].text,
            td[11].text,
            td[12].text,
            td[13].text,
            td[14].text,
        ]
    )

with open("Lottery_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        [
            "RoundID",
            "Red1",
            "Red2",
            "Red3",
            "Red4",
            "Red5",
            "Red6",
            "Blue",
            "PoolCash",
            "Tickets1",
            "Cash1",
            "Ticket2",
            "Cash2",
            "NewCash",
            "datetime",
        ]
    )
    writer.writerows(lst)
    csvfile.close()


# see: https://zhuanlan.zhihu.com/p/352588186
