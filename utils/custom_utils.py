import os
from tqdm import tqdm
import progressbar
from urllib.request import urlretrieve
import zipfile
from torchvision import transforms
import matplotlib.pyplot as plt

pbar = None

def show_progress(block_num, block_size, total_size):
    '''
    will show a progress bar during the download
    '''
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
    '''
    Download the zip file from URL if not already downloaded
    and put it in the rootpath directory.
    It is then extracted in the rootpath directory.
    
    '''
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

                
                
def show_images(img):
    plt.imshow(transforms.functional.to_pil_image(img))
    plt.show()