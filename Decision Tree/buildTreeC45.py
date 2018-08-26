from  xml.dom import  minidom
from xml.etree import ElementTree as ET
import math

def prettify(elem, level = 0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            prettify(e, level + 1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i
    return elem

def isnum(attr):
    for x in set(attr):
        if not x == "?":
            try:
                x = float(x)
                return isinstance(x, float)
            except ValueError:
                return False
    return True

def entropy(x):
    ent = 0
    for k in set(x):
        p_i = float(x.count(k)) / len(x)
        ent = ent - p_i * math.log(p_i, 2)
    return ent
    
def infoGain_ratio(classes, attr):
    s = 0
    cat = []
    att = []
    for i in range(len(attr)):
        if not attr[i] == "?":
            cat.append(classes[i])
            att.append(attr[i])
    for i in set(att):      
        p_i = float(att.count(i)) / len(att)
        cat_i = []
        for j in range(len(cat)):
            if att[j] == i:
                cat_i.append(cat[j])
        s = s + p_i * entropy(cat_i)
    infoGain = entropy(cat) - s
    ent_att = entropy(att)
    if ent_att == 0:
        return 0
    else:
        return infoGain / ent_att

def infoGain(classes, attr):
    cats = []
    for i in range(len(attr)):
        if not attr[i] == "?":
            cats.append([float(attr[i]), classes[i]])
    cats = sorted(cats, key = lambda x:x[0])
    
    cat = [cats[i][1] for iCat in range(len(cats))]
    att = [cats[i][0] for iCat in range(len(cats))]
    if len(set(att)) == 1:
        return 0
    else:
        infoGains = []
        divPoint = []
        for i in range(1, len(cat)):
            if not att[i] == att[i-1]:
                infoGains.append(entropy(cat[:i]) * float(i) / len(cat) + entropy(cat[i:]) * (1-float(i) / len(cat)))
                divPoint.append(i)
        infoGain = entropy(cat) - min(infoGains)
    
        p_1 = float(divPoint[infoGains.index(min(infoGains))]) / len(cat)
        ent_attr = -p_1 * math.log(p_1, 2)-(1 - p_1) * math.log((1 - p_1), 2)
        return infoGain / ent_attr

def divisionPoint(classes, attr):
    cats = []
    for i in range(len(attr)):
        if not attr[i] == "?":
            cats.append([float(attr[i]), classes[i]])
    cats = sorted(cats, key = lambda x:x[0])
    
    cat = [cats[i][1] for iCat in range(len(cats))]
    att = [cats[i][0] for iCat in range(len(cats))]
    infoGains = []
    divPoint = []
    for i in range(1, len(cat)):
        if not att[i] == att[i-1]:
            infoGains.append(entropy(cat[:i]) * float(i) / len(cat) + entropy(cat[i:]) * (1-float(i) / len(cat)))
            divPoint.append(i)
    return att[divPoint[infoGains.index(min(infoGains))]]
    


def buildTree(dataSet, classes, parent, attrNames):
    if len(set(classes)) > 1:
        
        division = []
        for i in range(len(dataSet)):
            if set(dataSet[i]) == set("?"):
                division.append(0)
            else:
                if (isnum(dataSet[i])):
                    division.append(infoGain(classes, dataSet[i]))           
                else:
                    division.append(infoGain_ratio(classes, dataSet[i]))
        if max(division) == 0:
            numMax = 0
            for cat in set(classes):
                numCat = classes.count(cat)
                if numCat > numMax:
                    numMax = numCat
                    mostCat = cat                
            parent.text = mostCat
        else:
            indexSelected = division.index(max(division))
            nameSelected = attrNames[indexSelected]
            if isnum(dataSet[indexSelected]):
                divPoint = divisionPoint(classes, dataSet[indexSelected])
                rChildData = [[] for i in range(len(dataSet))]
                rChildClasses = []
                lChildData = [[] for i in range(len(dataSet))]
                lChildClasses = []
                for i in range(len(classes)):
                    if not dataSet[indexSelected][i] == "?":
                        if float(dataSet[indexSelected][i]) < float(divPoint):
                            lChildClasses.append(classes[i])
                            for j in range(len(dataSet)):
                                lChildData[j].append(dataSet[j][i])     
                        else:
                            rChildClasses.append(classes[i])
                            for j in range(len(dataSet)):
                                rChildData[j].append(dataSet[j][i])  
                if len(lChildClasses)>0 and len(rChildClasses) > 0:
                    p_l = float(len(lChildClasses)) / (len(dataSet[indexSelected]) - dataSet[indexSelected].count("?"))
                    child = ET.SubElement(parent,nameSelected, {'value' : str(divPoint), "flag" : "l", "p" : str(round(p_l,3))})
                    buildTree(lChildData, lChildClasses, child, attrNames)
                    child = ET.SubElement(parent, nameSelected, {'value' : str(divPoint),"flag":"r","p" : str(round(1-p_l,3))})
                    buildTree(rChildData, rChildClasses, child, attrNames)
                else:
                    numMax = 0
                    for cat in set(classes):
                        numCat = classes.count(cat)
                        if numCat > numMax:
                            numMax = numCat
                            mostCat = cat                
                    parent.text = mostCat
            else:
                for k in set(dataSet[indexSelected]):
                    if not k == "?":
                        childData = [[] for i in range(len(dataSet))]
                        childClasses = []
                        for i in range(len(classes)):
                            if dataSet[indexSelected][i] == k:
                                childClasses.append(classes[i])
                                for j in range(len(dataSet)):
                                    childData[j].append(dataSet[j][i])
                        child = ET.SubElement(parent, nameSelected, {'value' : k, "flag" : "m", 'p' : str(round(float(len(childClasses)) / (len(dataSet[indexSelected]) - dataSet[indexSelected].count("?")),3))}) 
                        buildTree(childData,childClasses,child,attrNames)   
    else:
        parent.text = classes[0]
        

def trainTree(trainDataSet, trainClass, xmlFileName, displayTree = False):
    if not len(trainDataSet) == len(trainClass):
        return False
    attrNames = trainDataSet[0]
    dataSet = [[] for iName in range(len(attrNames))]
    classes = []

    for iSamples in range(1,len(trainDataSet)):
        classes.append(trainClass[iSamples])
        for jNames in range(len(attrNames)):
            dataSet[jNames].append(trainDataSet[iSamples][jNames])
    root = ET.Element('DecisionTree')
    tree = ET.ElementTree(root)
    buildTree(dataSet, classes, root, attrNames)
    tree.write(xmlFileName)
    if displayTree:
        ET.dump(prettify(root))
    return True

def add(a, b):
    t = a
    for i in b:
        if i in t:
            t[i] = t[i] + b[i]
        else:
            t[i] = b[i]
    return t

def decision(root, sample, attrNames, p):
    if root.hasChildNodes():
        attrName = root.firstChild.nodeName
        if attrName == "#text":
            
            return decision(root.firstChild, sample, attrNames, p)  
        else:
            attr = sample[attrNames.index(attrName)]
            if attr == "?":
                d = {}
                for child in root.childNodes:                    
                    d = add(d, decision(child, sample, attrNames, p * float(child.getAttribute("p"))))
                return d
            else:
                for child in root.childNodes:
                    if child.getAttribute("flag") == "m" and child.getAttribute("value") == attr or \
                        child.getAttribute("flag") == "l" and float(attr) < float(child.getAttribute("value")) or \
                        child.getAttribute("flag") == "r" and float(attr) >= float(child.getAttribute("value")):
                        return decision(child,sample,attrNames,p)    
    else:
        return {root.nodeValue:p}

def testTree(xmlFileName, testDataSet):
    doc = minidom.parse(xmlFileName)
    root = doc.childNodes[0]
    prediction = []
    attrNames = testDataSet[0]
    for iSamples in range(1,len(testDataSet)):
        resultList = decision(root, testDataSet[iSamples], attrNames, 1)
        resultList = sorted(iter(resultList.items()), key=lambda x:x[1], reverse = True )
        result = resultList[0][0]
        prediction.append(result)
    return prediction



    
