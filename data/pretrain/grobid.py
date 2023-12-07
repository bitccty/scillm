import xmltodict
import json

def parse_grobid_json(gJSON):
    if not gJSON:
        return
    pdfStruct = {}
    pdfHeader = gJSON.get('TEI', {}).get('teiHeader')
    pdfText = gJSON.get('TEI', {}).get('text')
    
    # Title
    try:
        pdfStruct['title'] = pdfHeader.get('fileDesc', {}).get('titleStmt', {}).get('title', {}).get('#text', "") or ""
    except:
        pdfStruct['title'] = ""
    
    # Publication
    pdfStruct['publication'] = {}
    try:
        pdfStruct['publication']['publisher'] = pdfHeader.get('fileDesc', {}).get('publicationStmt', {}).get('publisher', "") 
        pdfStruct['publication']['date'] = pdfHeader.get('fileDesc', {}).get('publicationStmt', {}).get('date', {}).get('@when', "")
    except:
        pass
        
    # Authors
    try:
        pdfStruct['author'] = get_grobid_author(pdfHeader.get('fileDesc', {}).get('sourceDesc', {}).get('biblStruct', {}).get('analytic', {}).get('author'))
    except:
        pdfStruct['author'] = []

    # Abstract
    try:
        pdfStruct['abstract'] = get_grobid_abstract(pdfHeader.get('profileDesc', {}).get('abstract', {}).get('div'))
    except:
        pdfStruct['abstract'] = []
    
    # Body
    try:
        pdfStruct['body'] = get_grobid_body(pdfText.get('body', {}).get('div'))
    except:
        pdfStruct['body'] = []
    
    # Reference
    try:
        pdfStruct['reference'] = get_grobid_reference(pdfText.get('back', {}).get('div'))
    except:
        pdfStruct['reference'] = []
    
    return pdfStruct


def get_grobid_author(authorSet):
    if not authorSet:
        return []
    authors = []
    if isinstance(authorSet, list):
        for authorItem in authorSet:
            try:
                auth = get_author_info(authorItem)
                if auth:
                    authors.append(auth)
            except:
                pass
    else:
        auth = get_author_info(authorSet)
        if auth:
            authors.append(auth)
    return authors

def get_author_info(authorItem):
    author = {}
    # Fetch item information
    try:
        foreName = authorItem.get('persName', {}).get('forename')
        if isinstance(foreName, list):
            author['forename'] = ""
            for name in foreName:
                currtext = name.get('#text')
                nextText = foreName[foreName.index(name) + 1].get('#text') if foreName.index(name) + 1 < len(foreName) else None
                if currtext:
                    if len(currtext) == 1 and 'A' <= currtext <= 'Z':
                        author['forename'] += currtext + "."
                        if nextText and not (len(nextText) == 1 and 'A' <= nextText <= 'Z'):
                            author['forename'] += " "
                    else:
                        author['forename'] += currtext + " "
                else:
                    author['forename'] += ""
        else:
            forname_text = foreName.get('#text')
            author['forename'] = forname_text + "." if forname_text and len(forname_text) == 1 and 'A' <= forname_text <= 'Z' else forname_text
    except:
        author['forename'] = ""
    
    try:
        author['surname'] = authorItem.get('persName', {}).get('surname') or ""
    except:
        author['surname'] = ""
    
    author['name'] = (author['forename'] + " " + author['surname']).strip()
    try:
        author['email'] = authorItem.get('email') or ""
    except:
        author['email'] = ""
    # Return None if forename or surname is empty
    if not author['forename'] or not author['surname']:
        return None
    return author

def get_grobid_paragraph(paraSet, isAbs=False):
    if not paraSet:
        return []
    paras = []
    if isinstance(paraSet, list):
        for paraItem in paraSet:
            try:
                p = get_para_info(paraItem, isAbs)
                if p:
                    paras.append(p)
            except:
                pass
    else:
        p = get_para_info(paraSet, isAbs)
        if p:
            paras.append(p)
    return paras

def get_para_info(paraItem, isAbs):
    if not paraItem:
        return None
    
    if isinstance(paraItem, str):
        paraTexts = paraItem
        return paraTexts
    else:
        paraTexts = paraItem.get('#text', "")
    # Return None if text is empty
    if not paraTexts:
        return None
    # Return text directly if abstract extracting
    if isAbs:
        return paraTexts
    # Fetch quote information
    paraQuotes = get_grobid_quote(paraItem.get('ref'))
    # Fetch paragraph information
    para = None
    if isinstance(paraTexts, list):
        texts = paraTexts[0]
        i = 1
        j = 0
        hasLeft = False
        while i < len(paraTexts):
            if j >= len(paraQuotes):
                texts += paraTexts[i]
                i += 1
                continue
            quoteText = paraQuotes[j].get('text', "")
            # context of 10 characters before
            paraQuotes[j]['context'] = texts[-10:]
            paraQuotes[j]['index'] = len(texts)
            if quoteText[0] == "[" or quoteText[0] == "(":
                if hasLeft:
                    texts += paraTexts[i]
                    i += 1
                if quoteText[-1] == "]" or quoteText[-1] == ")":
                    hasLeft = False
                else:
                    hasLeft = True
            elif quoteText[-1] == ";" or quoteText[-1] == ",":
                hasLeft = True
            else:
                hasLeft = False
            if hasLeft:
                texts += quoteText + " "
                j += 1
            else:
                texts += quoteText
                texts += paraTexts[i]
                i += 1
                j += 1
        para = texts
    else:
        para = paraTexts
    return para

def get_grobid_quote(quoteSet):
    if not quoteSet:
        return []
    quotes = []
    if isinstance(quoteSet, list):
        for quoteItem in quoteSet:
            q = get_quote_info(quoteItem)
            if q['type']:
                quotes.append(q)
    else:
        q = get_quote_info(quoteSet)
        if q['type'] == "bibr":
            quotes.append(q)
    return quotes

def get_quote_info(quoteItem):
    quote = {}
    # Fetch item information
    quote['text'] = quoteItem.get('#text', "")
    quote['target'] = quoteItem.get('@target', "")
    quote['type'] = quoteItem.get('@type', "")
    return quote

def get_grobid_abstract(abstractSet):
    if not abstractSet:
        return []
    abstracts = []
    if isinstance(abstractSet, list):
        for abstractItem in abstractSet:
            p = get_grobid_paragraph(abstractItem.get('p'), True)
            abstracts.append(p)
    else:
        p = get_grobid_paragraph(abstractSet.get('p'), True)
        abstracts.append(p)
    return abstracts

def get_grobid_body(bodySet):
    if not bodySet:
        return []
    bodies = []
    if isinstance(bodySet, list):
        for bodyItem in bodySet:
            b = get_section_info(bodyItem)
            bodies.append(b)
            # Check Section -1
            # if b['section']['index'] == -1:
            #     if len(bodies) > 0:
            #         # merge with last
            #         bodies[-1]['p'] += [b['section']['name']] + b['p']
            # else:
            #     bodies.append(b)
    else:
        b = get_section_info(bodySet)
        bodies.append(b)
    return bodies

def get_section_info(bodyItem):
    if not bodyItem:
        return None
    
    if isinstance(bodyItem.get("head"), str):
        IndexSet = ["VIII.", "VII.", "VI.", "V.", "IV.", "III.", "II.", "I."]
        sectIndex = -1
        sectName = bodyItem["head"]
        for i in range(len(IndexSet)):
            if IndexSet[i] in bodyItem["head"]:
                sectIndex = i + 1
                sectName = bodyItem["head"].replace(IndexSet[i], "").strip()
                break
    else:
        sectIndex = bodyItem.get('head', {}).get('@n', -1)
        sectName = bodyItem.get('head', {}).get('#text', "")
    p = get_grobid_paragraph(bodyItem.get('p'))
    return {'section': {'index': sectIndex, 'name': sectName}, 'p': p}


def get_grobid_reference(refDiv):
    if not refDiv:
        return []
    refs = []
    if isinstance(refDiv, list):
        for div in refDiv:
            if div.get('@type') == "references":
                refs = get_reference_struct(div.get('listBibl', {}).get('biblStruct'))
    else:
        if refDiv.get('@type') == "references":
            refs = get_reference_struct(refDiv.get('listBibl', {}).get('biblStruct'))
    return refs

def get_reference_struct(refSet):
    if not refSet:
        return []
    refs = []
    if isinstance(refSet, list):
        for refItem in refSet:
            try:
                ref = get_reference_info(refItem)
                if ref['title'] and (len(ref['author']) or ref['date'] or ref['doi']) and not judge_garbled(ref['title']):
                    refs.append(ref)
            except:
                pass
    else:
        ref = get_reference_info(refSet)
        if ref['title'] and (len(ref['author']) or ref['date'] or ref['doi']) and not judge_garbled(ref['title']):
            refs.append(ref)
    return refs

def judge_garbled(string):
    garbledCnt = 0
    garbledDic = {
        "!": False,
        "@": False,
        "#": False,
        "$": False,
        "%": False,
        "^": False,
        "&": False,
        "*": False,
        "+": False,
        "=": False,
        "/": False,
    }
    for char in string:
        if char in garbledDic and garbledDic[char] is False:
            garbledDic[char] = True
            garbledCnt += 1
    return garbledCnt >= 3

def get_reference_info(refItem):
    index = refItem.get('@xml:id', -1)
    title = refItem.get('analytic', {}).get('title', {}).get('#text') or refItem.get('monogr', {}).get('title', {}).get('#text') or ""
    author = get_grobid_author(refItem.get('analytic', {}).get('author') or refItem.get('monogr', {}).get('author'))
    doi = refItem.get('analytic', {}).get('idno', {}).get('#text') or refItem.get('monogr', {}).get('idno', {}).get('#text') or ""
    venue = refItem.get('monogr', {}).get('title', {}).get('#text') or ""
    date = refItem.get('monogr', {}).get('imprint', {}).get('date', {}).get('@when', "")
    return {
        'index': index,
        'title': title,
        'author': author,
        'doi': doi,
        'venue': venue,
        'date': date
    }


def parseGrobidXML(xml, out_file):
    # Check if xml is available
    if not xml:
        return None
    # convert xml to dict
    parsed_data = xmltodict.parse(xml)
    with open("z.json", "w") as fw:
        json.dump(parsed_data, fw, indent=4)         
    # parse the structure element
    extract_data = parse_grobid_json(parsed_data)
    # # write to file
    # fw.write(json.dumps(extract_data) + "\n")


def init_metric():
    metric = {
        't_acc': 0,
        'p_acc': 0,
        'au_acc': 0,
        'abs_acc': 0,
        'b_acc': 0,
        'r_acc': 0,
        'main_acc': 0,
        'all_acc': 0,
        'count': 0,
        'body_size': 0
    }
    return metric


def eval_extract(item, metric):
    metric['t_acc'] += int(bool(item['title']))
    metric['p_acc'] += int(bool(item['publication']))
    metric['au_acc'] += int(bool(item['author']))
    metric['abs_acc'] += int(bool(item['abstract']))
    metric['b_acc'] += int(bool(item['body']))
    metric['r_acc'] += int(bool(item['reference']))
    metric['main_acc'] += int(bool(item['title']) or bool(item['abstract']) or bool(item['body']))
    metric['all_acc'] += int(bool(item['title']) or bool(item['abstract']) or bool(item['body']) \
        or bool(item['publication']) or bool(item['author']) or bool(item['reference']))
    metric['count'] += 1
    metric['body_size'] = len(item['body'])
    
    return metric
    

if __name__ == "__main__":
    with open("pretrain/xml/ba96cbb5778c24ff548af07fe8855532.xml", "r") as fw:
        data = fw.read()
    parseGrobidXML(data, "")
    
