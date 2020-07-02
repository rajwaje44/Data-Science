
# importing necessary libraries
from urllib.request import urlopen as ureq
from bs4 import BeautifulSoup as soup
import re

# url name to fetch data from
url = "https://www.flipkart.com/beauty-and-grooming/fragrances/perfume/pr?sid=g9b,0yh,jhz&p[]=facets.ideal_for%255B%255D%3DMen%2B%2526%2BWomen&p[]=facets.ideal_for%255B%255D%3DMen&p[]=facets.serviceability%5B%5D%3Dtrue&otracker=categorytree&otracker=nmenu_sub_Men_0_Perfumes"

# Reading as HTML file
uclient = ureq(url)
page_html = uclient.read()
uclient.close()

# using beautiful soup to parse
bsobject = soup(page_html,"html.parser")

# Selecting count tag to find total number of items
object1 = bsobject.select(".eGD5BM")
obj = object1[0].text
total_products = (obj.split(" ")[-2])
print(total_products)

# Storing details(tags) of each product in a object named container
containers = bsobject.findAll("div",{"class":"_3liAhj"})
print(containers[0])
ob1 = print(soup.prettify(containers[0]))

# first finding details in container 1 : name, price, discount%, discounted price, rating, link
container = containers[0]

# Rough Work
'''
name = container.select("._2cLu-l")       # name = container.findAll("a")
print(name[0].text)                       # print(name[1].text)

price = container.select("._3auQ3N")       # while using select make sure to put ".  <------------------ before class"
price = price[0].text
price = price.split("₹")
price = "Rs" + price[1]
print(price)

discounted_price = container.select("._1vC4OE")
discounted_price = discounted_price[0].text
discounted_price = discounted_price.split("₹")
discounted_price = "Rs" + discounted_price[1]
print(discounted_price)

rating = container.select(".hGSR34")
rating = rating[0].text
print(rating)

link = container.a["href"]
print(url + link)
'''

filename = "D:/dataforpython/flipkartdata.csv"
f = open(filename,"w")

headers = "Name,Price,Discounted_price,Rating,Link\n"
f.write(headers)

for container in containers:
    name = container.select("._2cLu-l")  # name = container.findAll("a")

    price = container.select("._3auQ3N")
    price = str(price)
    price = price[-12:-7]
    price = (re.sub("->","",price)).strip()
    price = "Rs" + price

    discounted_price = container.select("._1vC4OE")
    discounted_price = discounted_price[0].text
    discounted_price = discounted_price.split("₹")
    discounted_price = "Rs" + discounted_price[1]


    rating = container.select(".hGSR34")
    rating = rating[0].text

    link = container.a["href"]

    print(name[0].text)
    print(price)
    print(discounted_price)
    print(rating)
    print(url + link)

    f.write(name[0].text + "," + price + "," + discounted_price + "," + rating + "," + link + "\n")
f.close()







