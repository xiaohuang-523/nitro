import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import PyPDF2
import types


def findInDict(needle,haystack):
    for key in haystack.keys():
        try:
            value = haystack[key]
        except:
            continue
        if key == needle:
            return value
        if type(value) == types.DictType or isinstance(value,PyPDF2.generic.DictionaryObject):
            x = findInDict(needle,value)
            if x is not None:
                return x

# creating a pdf file object
file = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\Edentulous registration error analysis\\Case report\\Case Report #2128   Dr. Fish   2020-09-01 ( Plan-2 Drill-2 Guided-2 ).pdf"
pdfFileObj = open(file, 'rb')

# creating a pdf reader object
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

# printing number of pages in pdf file
#print(pdfReader.numPages)
#print(pdfReader.resolvedObjects)
#pageObj = pdfReader.getPage(1)
print(pdfReader.getPage(2))
print(pdfReader.getPage(2)['/Resources']['/Font']['/F11'])

print('show data')
print(pdfReader.getPage(2)['/Contents'].getData())

#pdf = PyPDF2.PdfFileReader(open('file.pdf'))
#pages = list(pdfReader.pages)


#answer = findInDict('/MYOBJECT',pdfReader.resolvedObjects).getData()
# creating a page object
#pageObj = pdfReader.getPage(1)

# extracting text from page
#print(pageObj.extractText())
#print(pageObj)

# closing the pdf file object
pdfFileObj.close()



