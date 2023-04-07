import os
from tqdm import tqdm
import progressbar
from urllib.request import urlretrieve
import zipfile

pbar = None

def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

def RetrieveData(URL, rootpath="./data"):
    if os.path.isdir(rootpath)==False:
        os.mkdir('./data')
        
    if os.path.isfile(os.path.join(rootpath,'data.zip'))==False:
        print('Downloading archive')
        urlretrieve(URL,os.path.join(rootpath,'data.zip'),show_progress)
        with zipfile.ZipFile(os.path.join(rootpath,'data.zip'),'r') as Zip:
            for member in tqdm(Zip.infolist(), desc='Extracting '):
                try:
                    Zip.extract(member, rootpath)
                except zipfile.error as e:
                    pass