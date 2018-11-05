from PIL import Image
import os

data_path = 'C:\\Users\\FY\\Desktop\\TongueColorDetectYfang\\Tongues'
img_list = os.listdir(data_path)

for img_name in img_list:
    img=Image.open(os.path.join(data_path, img_name))

    out = img.resize((227, 227))  # resize成299*299像素大小。

    out.save(os.path.join('C:\\Users\\FY\\Desktop\\TongueColorDetectYfang\\Tongues\\resize', img_name))
