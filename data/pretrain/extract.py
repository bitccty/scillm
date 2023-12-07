from loguru import logger
import os
import shutil
import requests
import json
import traceback
from tqdm import tqdm
import xmltodict
import argparse
from grobid import parse_grobid_json, init_metric, eval_extract
import hashlib
import copy


MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


def download(url, document_dir):
    # Check if the file has been downloaded
    filename = os.path.join(document_dir, os.path.basename(url) + ".pdf")
    if os.path.exists(filename):
        return filename
    
    # Get the URL parameter from the request
    logger.info('----- Download Start -----')
    logger.info(f'URL -> {url}')

    # Check if the url ends with '.pdf'
    if not url.lower().endswith('.pdf'):
        url = url.replace("abs", "pdf") + ".pdf"

    # Get the file size
    r = requests.head(url)
    content_length = int(r.headers.get('Content-Length', 0))

    if content_length == 0:
        logger.warning(
            'Content-Length header not found, file size cannot be determined before download'
        )

    # Check if the file size is within limit
    if content_length > MAX_FILE_SIZE:
        message = f'File size exceeds the limit of {MAX_FILE_SIZE} bytes.'
        logger.error(message)

    # Create the 'document' subdirectory if it doesn't exist
    if not os.path.exists(document_dir):
        os.makedirs(document_dir)

    try:
        download_count = 0
        while True:
            # Download the file
            download_count += 1
            with requests.get(url, stream=True) as r, open(filename, 'wb') as f:
                with tqdm(total=content_length, unit='B', unit_scale=True) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Check if the downloaded file size is correct
            if download_count == 3 or os.path.getsize(filename) == content_length:
                return filename

    except Exception as e:
        traceback.print_exc()
        logger.error(str(e))
        return None


def grobid(pdf_file, xml_dir):
    # Check if pdf is available
    if not pdf_file:
        return None
    
    with open(pdf_file, "rb") as fw:
        source = fw.read()
    
    docHash = get_md5_of_file(copy.deepcopy(source))
    
    # Create the 'xml' subdirectory if it doesn't exist
    if not os.path.exists(xml_dir):
        os.makedirs(xml_dir)
    
    cache_file = xml_dir + f'/{docHash}.xml'
    
    if not os.path.exists(cache_file):
        logger.info('Try to extract new document')
        
        # Extract structure
        xml, status = fetch_xml_from_grobid(source)
        
        if status != 200:
            xml = None
            print('Grobid failed.')
        else:
            # Cache structure
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(xml)
                logger.info(f'Result {docHash} cached')
    else:
        logger.info('Try to fetch cached result')

        # Cache structure
        with open(cache_file, 'r', encoding='utf-8') as f:
            xml = f.read()
            logger.info(f'Cached {docHash} fetched')
    
    return xml
        

def fetch_xml_from_grobid(pdf_file):
    files = {
        'input': ('placeholder.pdf', pdf_file, 'application/pdf', {
            'Expires': '0'
        })
    }

    url = f'http://127.0.0.1:8070/api/processFulltextDocument'

    headers = {'Accept': 'application/xml'}
    params, data = {}, {}

    r = requests.request(
        'POST',
        url,
        headers=headers,
        params=params,
        files=files,
        data=data,
        timeout=60,
    )
    return r.text, r.status_code


def get_md5_of_file(file):
    blockSize = 64 * 1024
    md5 = hashlib.md5()
    while file:
        md5.update(file[:blockSize])
        file = file[blockSize:]
    fileHash = md5.hexdigest()
    return fileHash


def parseGrobidXML(xml):
    # Check if xml is available
    if not xml:
        return None
    # convert xml to json
    parsed_data = xmltodict.parse(xml)                
    # parse the structure element
    extract_data = parse_grobid_json(parsed_data)
    
    return extract_data


def main(args):
    # load urls
    print(f"[!]Load Paper Urls")
    if os.path.exists(args["cache_file"]):
        with open(args["cache_file"], "r") as fw:
            urls = json.load(fw)
    else:
        with open(args["input_file"], "r") as fw:
            urls = [json.loads(line)["url"] for line in tqdm(fw.readlines())]
        with open(args["cache_file"], "w") as fw:
            json.dump(urls, fw, indent=4)
    
    # Create the 'output' subdirectory if it doesn't exist
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])
    f_out = open(os.path.join(args["output_dir"], "arxiv_structure_" + str(args["start_pos"]) + ".json"), "w")
    f_error = open(os.path.join(args["output_dir"], "error" + str(args["start_pos"]) + ".json"), "w")
    metric = init_metric()
    print(f"[!]Extract Sturture Paper")
    for i in tqdm(range(args["start_pos"], len(urls))):
        try:
            # download pdf
            pdf_file = download(urls[i], args["pdf_dir"])
            # convert to xml
            xml = grobid(pdf_file, args["xml_dir"])
            # convert to paper structure with json
            st_data = parseGrobidXML(xml)
        except:
            f_error.write(urls[i] + "\n")
            continue
        # save
        if st_data:
            f_out.write(json.dumps(st_data) + "\n")
            f_out.flush()
            # eval
            metric = eval_extract(st_data, metric)
            logger.info(metric)
        
        # clean cache
        if i % args["batch_size"] == 0:
            clean_cache(args)


    clean_cache(args)
    # save
    with open(os.path.join(args["output_dir"], "result.json"), "w") as f:
        json.dump(metric, f)


def clean_cache(args):
    shutil.rmtree(args["pdf_dir"])
    shutil.rmtree(args["xml_dir"])
    os.makedirs(args["pdf_dir"])
    os.makedirs(args["xml_dir"])
    

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./pretrain/redpajama_arxiv.json")
    parser.add_argument("--cache_file", type=str, default="./pretrain/arxiv_urls.json")
    parser.add_argument("--pdf_dir", type=str, default="./pretrain/document")
    parser.add_argument("--xml_dir", type=str, default="./pretrain/xml")
    parser.add_argument("--output_dir", type=str, default="./pretrain/output")
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--start_pos", type=int, default=0)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parseArgs()
    args = vars(args)
    main(args)
