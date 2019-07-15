from PIL import Image
import os.path
import glob
<<<<<<< HEAD
def convertjpg(jpgfile,outdir,width=600,height=600):
=======
def convertjpg(jpgfile,outdir,width=128,height=128):
>>>>>>> 8fb0135ae072070df8fd4db81cf92544d8de7ecf
    img=Image.open(jpgfile).convert('RGB')
    try:
        new_img=img.resize((width,height),Image.BILINEAR)
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)
for jpgfile in glob.glob("/Users/zac/5-Algrithm/algrithm-data/ImageCluster/images/*.jpg"):
    convertjpg(jpgfile,"/Users/zac/5-Algrithm/algrithm-data/ImageCluster/images_resized")
