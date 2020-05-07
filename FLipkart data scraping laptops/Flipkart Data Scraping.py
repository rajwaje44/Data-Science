
# scraping data from flipkart site (laptop data)

# import necessary libraries
from urllib.request import urlopen as ureq
from bs4 import BeautifulSoup as soup

url = "https://www.flipkart.com/search?q=laptop&sid=6bo%2Cb5g&as=on&as-show=on&otracker=AS_QueryStore_OrganicAutoSuggest_2_3_na_na_na&otracker1=AS_QueryStore_OrganicAutoSuggest_2_3_na_na_na&as-pos=2&as-type=RECENT&suggestionId=laptop%7CLaptops&requestId=81e6694e-5755-432a-9781-40424963c919&as-searchtext=lap"

uclient = ureq(url)
page_html = uclient.read()
uclient.close()

# creating soup object
object = soup(page_html,"html.parser")

# create obeject (like container) to store data of laptops
containers = object.findAll("div",{"class":"_1UoZlX"})
print(len(containers))

# we want info from html doc but our url is a plain string so to give it a structure we will use prettify function in beautiful soup
print(soup.prettify(containers[0]))

# lets first extract information of one laptop
container = containers[0]

# extract product name
product_name = container.findAll("div",{"class":"_3wU53n"})
print(product_name[0].text)

# extract price
price = container.findAll("div",{"class":"_1vC4OE _2rQ-NK"})
print(price[0].text)

# extract rating
rating = container.findAll("div",{"class":"hGSR34"})
print(rating[0].text)

# creating excel file to store the data
filename = "D:/dataforpython/laptop_data.csv"
f = open(filename,"w")

headers = "Laptop_name,price,rating"
f.write(headers)

for container in containers:
    name = container.findAll("div", {"class": "_3wU53n"})
    product_name = name[0].text

    pricing = container.findAll("div", {"class": "_1vC4OE _2rQ-NK"})
    price = pricing[0].text

    rate = container.findAll("div", {"class": "hGSR34"})
    rating = rate[0].text

    trim_product_name = product_name.split("-")     # some string manipulation
    rupee = ''.join(price.split(','))
    Rs_r = rupee.split("₹")
    Rs = "Rs" + Rs_r[1]

    print(product_name + "," + Rs + "," + rating + "\n")
    f.write(product_name + "," + Rs + "," + rating + "\n")

f.close()




















