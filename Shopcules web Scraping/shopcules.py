# shopcules data scraping

from urllib.request import urlopen as ureq
from bs4 import BeautifulSoup as soup
import io

Dictonary = {0:"-formal-shoes",1:"-casual-shoes",2:"-sandals",3:"boots",4:"-loafers-moccasins"}
string = input("Select number from dictonary:")

#url = "https://www.shopclues.com/footwear-mens-formal-shoes.html?page=1"
url = "https://www.shopclues.com/footwear-mens"+Dictonary[int(string)]+".html?page=1"

uclient = ureq(url)
html_page = uclient.read()
uclient.close()
bsobject = soup(html_page,"html.parser")


'''
Rough work: To see if one record is being scarped or not
## below commented code is only used to see if with only one product if the data is fetched correctly or not

containers = bsobject.find_all("div",{"class":"column col4"})
container = containers[0]
#print(soup.prettify(container))  #<-----------this is only to visualize html code in notepad


name = container.select(".prod_name")
name1 = name[0].text
print(name1)

img = container.findAll("div",{"class":"img_section img_320new"})
img = (container.div.img["data-img"])
print(img)


img = container.findAll("img")
image = img[0].get("src")
print(image)
containers = bsobject.find_all("div",{"class":"column col4"})
container = containers[0]
img = (container.div)
img = img.img["data-img"]
print(img)
'''

# count of number of products
url1 = "https://www.shopclues.com/footwear-mens"+Dictonary[int(string)]+".html?page=1"

uclient1 = ureq(url1)
html_page1 = uclient1.read()
uclient1.close()
bsobject1 = soup(html_page1,"html.parser")

containers1 = bsobject1.select(".product_found")
count = containers1[0].text
count = (count[0:5])
print(count)

loops = int(count) //24
print(loops)

# giving headers to csv columns
#headers = "Name,actual_price,discount_in_%,discounted_price,\n"
headers = "Names,Actual_price,Discount,Discounted_price,Link\n"
with io.open("D:/dataforpython/shopcules.csv","w",encoding="utf8") as f1:
    f1.write(headers)
    f1.close()


i = 1
for i in range(loops + 1):
    link = "https://www.shopclues.com/footwear-mens"+Dictonary[int(string)]+".html?page="
    mainlink = link + str(i)
    client = ureq(mainlink)
    html_page = client.read()
    client.close()
    bsobject = soup(html_page, "html.parser")
    containers = bsobject.find_all("div",{"class":"column col4"})

    for j in containers:
        name = j.select(".prod_name")
        if len(name) > 0 :
            name1 = name[0].text
            name1 = "".join(name1.split(","))
        else:
            name1 = " "


        discount = j.select(".prd_discount")
        if len(discount) == 0:
            discount1 = " "
        else:
            discount1 = discount[0].text
            discount1 = "".join(discount1.split("%"))
            print(discount1)
            discount1 = discount1[0:2]

        actual_price = j.findAll("div",{"class":"old_prices"})
        if len(actual_price) > 0 :
            actual_price = actual_price[0].text
            actual_price = "".join(actual_price.split(" "))
            actual_price = actual_price[0:6]
            actual_price = actual_price.strip()  # this is imp as if we dont do this step we will get actual price on next line
        else:
            actual_price = " "


        discounted_price = j.select(".p_price")
        if len(discounted_price) > 0 :
            discounted_price = discounted_price[0].text
            discounted_price = "".join(discounted_price.split("."))
        else:
            discounted_price = " "

        img = (j.div)
        if len(img) > 0 :
            img = img.img["data-img"]
            print(img)
            # img = (j.div.img["data-img"])
        else:
            img = " "

        dataline = str(name1) + "," + str(actual_price) + "," + str(discount1 + " %") + "," + str(discounted_price) + "," + str(img) +"\n"
        print(dataline)
        with io.open("D:/dataforpython/shopcules.csv", "a", encoding="utf8") as f1:
            f1.write(dataline)
            f1.close()
print("completed")
















