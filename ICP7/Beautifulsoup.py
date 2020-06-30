import re
import requests
from bs4 import BeautifulSoup
link=requests.get("https://en.wikipedia.org/wiki/Google")   #opening the link using requests.get
bs=BeautifulSoup(link.content,"html.parser")                #getting html part of link
#texts = bs.body.findAll(text=True)
texts = bs.body.get_text()                                  #finding text content in the complete file
lines = [i.strip() for i in texts.splitlines()]             #spliting a string into list and strip the data
chunks = [word.strip() for line in lines for word in line.split(" ")]
text = ' '.join(chunk for chunk in chunks if chunk)
file=open("texts.txt","a+")                                 #writing to file
file.write(str(text.encode("utf-8")))
file.close()
print("writing to file completed")

